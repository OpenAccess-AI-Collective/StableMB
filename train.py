import re
from pathlib import Path

from accelerate import Accelerator
from MEGABYTE_pytorch import MEGABYTE
import bitsandbytes as bnb

import math
import random
import signal
import tqdm
import gzip
import numpy as np
import os
import tempfile
import torch
import wandb
import torch.optim as optim
from datasets import load_dataset, Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from bitsandbytes.optim.adam import Adam8bit
from sophiag import SophiaG


BATCH_SIZE = 6
GRADIENT_ACCUMULATE_EVERY = 128
PRIME_LEN = 100
SEQ_LEN = 8192
LEARNING_RATE = 4e-4
BATCH_EST = 1
TOTAL_BATCH_EST = 622614  # estimate if available
# redpajama sample -> 622614
RESUME_FROM_CHECKPOINT = None
MODEL_BASE = None
INFERENCE = None  # set this to a text file for completion, use with MODEL_BASE

def calculate_sizes(total_batches):
    # constants
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    num_batches = total_batches // BATCH_SIZE // WORLD_SIZE // GRADIENT_ACCUMULATE_EVERY
    # validate_every = 1600 // BATCH_SIZE // WORLD_SIZE // GRADIENT_ACCUMULATE_EVERY
    # generate_every = 8000 // BATCH_SIZE // WORLD_SIZE // GRADIENT_ACCUMULATE_EVERY
    # checkpoint_every = 3200 // BATCH_SIZE // WORLD_SIZE // GRADIENT_ACCUMULATE_EVERY
    validate_every = 10
    generate_every = 0
    checkpoint_every = 5
    return num_batches, validate_every, generate_every, checkpoint_every

# helpers

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable: {100 * trainable_params / all_param}")


def cycle(loader, infinite=True):
    while True:
        for data in loader:
            yield data
        if not infinite:
            break

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


class WrappedDataset(IterableDataset):
    def __init__(self, huggingface_dataset, seq_len, infinite=True):
        self.huggingface_dataset = huggingface_dataset
        self.seq_len = seq_len
        self.infinite = infinite

    def __iter__(self):
        buffer = torch.tensor([], dtype=torch.long)
        while True:  # Infinite loop over the dataset
            for row in self.huggingface_dataset:
                formatted_text = row['text']
                x = np.frombuffer(formatted_text.encode(), dtype=np.uint8).copy()
                buffer = torch.cat((buffer, torch.from_numpy(x)), dim=0)
                while len(buffer) >= self.seq_len:
                    yield buffer[:self.seq_len].long()
                    buffer = buffer[self.seq_len:]
            if not self.infinite:
                if len(buffer):
                    yield buffer
                break


def get_ds_len(ds, seq_len):
    length = 0
    for row in ds:
        length += len(row["text"])
    return math.ceil(length / seq_len)


def generate(model, text):
    x = np.frombuffer(text.encode(), dtype=np.uint8).copy()
    input = torch.from_numpy(x)

    print(f'%s \n\n %s', (text, '*' * 100))
    sample = model.generate(input[None, :].cuda())
    sample = sample.flatten(1)
    output_str = decode_tokens(sample[0][len(input):])
    print(output_str)


def main():
    raw_ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")

    # instantiate GPT-like decoder model

    # device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
    device_properties = torch.cuda.get_device_properties(torch.device(f'cuda:{os.environ["LOCAL_RANK"]}'))

    if device_properties.major == 8 and device_properties.minor == 0:
        flash_attn = False  # set to false if using A100
    else:
        flash_attn = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # global: D: 1024, L: 14
    # local: D: 1024, L: 18
    model = MEGABYTE(
        num_tokens = 256,
        dim = 384,
        depth = (24, 12),
        max_seq_len = (2048, 1024),
        dim_head = 64,
        heads=16,
        flash_attn = flash_attn
    )
    # model = MEGABYTE(
    #     num_tokens = 8192,
    #     dim = (768, 1024),  # embeddings dimenstion -> d_head
    #     max_seq_len = (2048, 1024),  # number of embeddings -> d_model
    #     depth = (14, 18),  # numof layers => #L
    #     dim_head = 92,
    #     heads = 16,
    #     ff_dropout=0.1,
    #     attn_dropout=0.1,
    #     flash_attn = True,
    # ).to(accelerator.device, dtype=torch.bfloat16)
    if MODEL_BASE and isinstance(MODEL_BASE, str) and Path(MODEL_BASE).is_file():
        """
        this is code to strip the "module." prefix off module names if it's incorrect
        """
        # from collections import OrderedDict
        # state_dict = torch.load(MODEL_BASE)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove module.
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict)

        model.load_state_dict(torch.load(MODEL_BASE))

    if INFERENCE and isinstance(INFERENCE, str) and Path(INFERENCE).is_file():
        with open(INFERENCE, 'r') as fin:
            text = "\n".join(fin.readlines())
        model.cuda()
        model.eval()
        generate(model, text)
        return

    print_trainable_parameters(model)
    # prepare enwik8 data

    ds = raw_ds["train"].train_test_split(test_size=0.001)
    # train_dataset = WrappedDataset(ds["train"], SEQ_LEN)
    # val_dataset = WrappedDataset(ds["test"], SEQ_LEN)
    # train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    # val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

    train_dataset = WrappedDataset(ds["train"], SEQ_LEN, infinite=False)
    val_dataset   = WrappedDataset(ds["test"], SEQ_LEN, infinite=False)
    if not TOTAL_BATCH_EST:
        TOTAL_BATCHES = get_ds_len(ds["train"], SEQ_LEN)
    else:
        TOTAL_BATCHES = TOTAL_BATCH_EST

    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE), infinite=False)
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE), infinite=False)

    # optimizer
    num_batches, validate_every, generate_every, checkpoint_every = calculate_sizes(TOTAL_BATCHES)

    # optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = Adam8bit(model.parameters(), lr=LEARNING_RATE)
    optimizer = SophiaG(model.parameters(), lr=LEARNING_RATE)

    wandb.login()

    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY,
    )
    accelerator.init_trackers(
        project_name="smb-wikipedia",
        config={"learning_rate": LEARNING_RATE},
    )

    model.to(accelerator.device, dtype=torch.bfloat16)

    # training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    signal.signal(
        signal.SIGINT,
        lambda signal, frame: (torch.save(model.state_dict(), 'model_out_sigint.pt'), exit(0)),
    )

    pbar = tqdm.tqdm(range(num_batches), mininterval=10., desc='training')
    device = torch.cuda.current_device()
    global RESUME_FROM_CHECKPOINT
    if RESUME_FROM_CHECKPOINT:
        if isinstance(RESUME_FROM_CHECKPOINT, int):
            model.load_state_dict(torch.load(f"./checkpoints/model_out.chkpt_{RESUME_FROM_CHECKPOINT}.pt"))
        elif RESUME_FROM_CHECKPOINT is True:
            # Get all checkpoint files
            files = list(Path("./checkpoints/").glob("model_out.chkpt_*.pt"))

            # Extract indices from filenames and find the max
            indices = [int(re.search('model_out.chkpt_(\d+).pt', f.name).group(1)) for f in files]
            max_index = max(indices)

            # Get the file with max index
            max_index_file = Path(f"./checkpoints/model_out.chkpt_{max_index}.pt")
            model.load_state_dict(torch.load(str(max_index_file)))
            RESUME_FROM_CHECKPOINT = max_index  # set this so it can properly skip later

    val_loss_str = ""
    for i in pbar:
        if RESUME_FROM_CHECKPOINT and i <= RESUME_FROM_CHECKPOINT:
            continue

        model.train()
        # for __ in range(GRADIENT_ACCUMULATE_EVERY):
        #     train_loss = model(next(train_loader), return_loss = True)

        with accelerator.accumulate(model):
            train_loss = model(next(train_loader), return_loss = True)
            accelerator.backward(train_loss)
            # loss.backward()

            train_loss_str = train_loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # what does this do?
            optimizer.step()
            optimizer.zero_grad()

        reserved = torch.cuda.memory_reserved(device)
        reserved_gb = reserved / 1024 / 1024 / 1024
        pbar.set_description(f'reserved_gb: {reserved_gb}, training loss: {train_loss_str}, validation loss: {val_loss_str}')

        if validate_every and i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader), return_loss = True)
                val_loss_str = loss.item()
                pbar.set_description(f'reserved_gb: {reserved_gb}, training loss: {train_loss_str}, validation loss: {val_loss_str}')
                accelerator.log({"train_loss": train_loss.item(), "valid_loss": loss.item()})
        else:
            accelerator.log({"train_loss": train_loss.item()})

        if i % checkpoint_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                with tempfile.NamedTemporaryFile(dir='./checkpoints/', delete=False) as f:
                    accelerator.save(model.state_dict(), f.name)
                    temp_name = f.name
                    os.rename(temp_name, f"./checkpoints/model_out.chkpt_{i}.pt")

    torch.save(model.state_dict(), 'model_out.pt')


if __name__ == "__main__":
    main()
