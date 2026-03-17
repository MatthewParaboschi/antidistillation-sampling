# -*- coding: utf-8 -*-
# ================================================================================
# SAVE_GRAD.PY - PROXY STUDENT GRADIENT COMPUTATION
# ================================================================================
# Computes and saves gradients from a proxy student model on holdout traces
# for use in antidistillation sampling (ADS).
#
# These gradients represent how the student model would be affected by changes
# to the teacher's outputs. The antidistillation mechanism uses them in a finite
# difference approximation to modify the teacher's sampling distribution.
# ================================================================================

import argparse
import os
import sys

import datasets
import torch
import yaml
import socket
from accelerate import Accelerator
from datasets import load_from_disk
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from trl import DataCollatorForCompletionOnlyLM

from utils import init

accelerator = Accelerator()

if not accelerator.is_main_process:
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    datasets.disable_progress_bar()
    tqdm = lambda x, *args, **kwargs: x


def log_color(content, title=""):
    console = Console(highlight=True, file=sys.stdout)
    console.print(Panel(content, title=title, border_style="cyan", title_align="left"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute proxy student gradients for antidistillation sampling.")
    parser.add_argument("holdout_config", type=str, help="Path to the holdout config.yaml file")
    parser.add_argument("--proxy_student", type=str, help="Proxy student model to use for gradient computation")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer model to use (should match proxy student)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--trace_colname", type=str, help="Column name for reasoning traces in dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for gradient computation")
    args = parser.parse_args()

    with open(args.holdout_config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    init(os.getenv("USER"), args.seed, "babel" in socket.gethostname())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, padding_side="left", trust_remote_code=True)

    if "llama" in args.tokenizer.lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token
    else:
        eos_token = tokenizer.eos_token
        special_tokens = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        args.proxy_student, trust_remote_code=True,
        torch_dtype=torch.float32, use_cache=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    for param in model.parameters():
        param.requires_grad = True
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    if accelerator.is_main_process:
        print(f"Student model {args.proxy_student} loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    if accelerator.is_main_process:
        ds = load_from_disk(args.trace_path)

        def preprocessor(examples):
            traces = examples[args.trace_colname]
            if tokenizer.bos_token:
                traces = [text.replace(tokenizer.bos_token, "", 1) for text in traces]
            tokenized = tokenizer(
                traces, padding=False, truncation=True,
                return_attention_mask=True, return_tensors=None,
            )
            return tokenized

        input_ds = ds.map(
            preprocessor, batched=True, num_proc=96,
            remove_columns=ds.column_names,
            desc="Preprocessing dataset", load_from_cache_file=True
        )
        dataset_size = len(input_ds)
        input_ds.save_to_disk("/tmp/cached_ds")
        print(f"Loaded {args.trace_path} with {dataset_size} samples")
        print(f"Example trace: {tokenizer.decode(input_ds[0]['input_ids'])}")

    accelerator.wait_for_everyone()
    input_ds = load_from_disk("/tmp/cached_ds")

    response_str = "<|im_start|>assistant\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(response_str, add_special_tokens=False),
        tokenizer=tokenizer, mlm=False
    )

    dataloader = DataLoader(
        input_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=data_collator, num_workers=1, pin_memory=True
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grads["module." + name] = torch.zeros_like(param.data)

    local_samples = 0
    model.train()

    for batch in tqdm(dataloader, desc="Accumulating gradients", disable=not accelerator.is_main_process):
        local_samples += batch["input_ids"].size(0)
        outputs = model(**batch)
        loss = outputs.loss * batch["input_ids"].size(0)
        accelerator.backward(loss)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads["module." + name].add_(param.grad)

        model.zero_grad()

    local_tensor = torch.tensor([local_samples], device=accelerator.device)
    accelerator.wait_for_everyone()
    reduced_tensor = accelerator.reduce(local_tensor, reduction="sum")
    total_samples = reduced_tensor.item()

    if accelerator.is_main_process:
        print(f"Processed a total of {total_samples} samples across all processes")

    for name in grads:
        accelerator.reduce(grads[name], reduction="sum")
        if accelerator.is_main_process:
            grads[name] = grads[name] / total_samples

    if accelerator.is_main_process:
        grad_save_path = os.path.join(args.exp_dir, "student_grads.pt")
        torch.save(grads, grad_save_path)
        print(f"Saved average gradients to {grad_save_path}")

        total_grad_norm = sum(torch.norm(grad).item() ** 2 for grad in grads.values()) ** 0.5
        print(f"Total gradient norm: {total_grad_norm:.2e}")
        print(f"Number of parameters with gradients: {len(grads)}")

    accelerator.end_training()
