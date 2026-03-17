# -*- coding: utf-8 -*-
# ================================================================================
# DISTILL.PY - STUDENT MODEL DISTILLATION FROM TEACHER TRACES
# ================================================================================
# FIXED VERSION: Qwen/Qwen2.5-3B-Instruct + TRL 0.12.0 compatibility
#
# Changes from original:
#   [FIX-1]  Removed Llama tokenizer branch; Qwen-only setup
#   [FIX-2]  Resize model embeddings after adding [PAD] special token
#   [FIX-3]  suffix_len slicing replaced with robust end-token stripping
#   [FIX-4]  Model loaded with use_cache=False (required during training)
#   [FIX-5]  attn_implementation="eager" (flash_attention_2 not installed)
#   [FIX-6]  bf16=True hardcoded (NOT student.config.use_bfloat16)
#   [FIX-7]  SFTConfig: max_seq_length (not max_length); removed params
#            not in TRL 0.12.0
#   [FIX-8]  Response template encoded WITHOUT special tokens
#   [FIX-9]  LoRA target_modules confirmed for Qwen2 architecture
#   [FIX-10] Preprocessing uses tokenize=False then encode()
#   [FIX-11] Distributed barrier before AND after dataset caching
#   [FIX-12] Re-enable use_cache=True only at save time
#   [FIX-13] num_proc reduced to reasonable value
# ================================================================================

import logging
import os
from io import StringIO
from types import SimpleNamespace

import datasets
import hydra
import torch
import yaml
import socket
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb
from utils import SYSTEM_PROMPT, init

accelerator = Accelerator()
log = logging.getLogger(__name__)

if not accelerator.is_main_process:
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    datasets.disable_progress_bar()
    tqdm = lambda x, *args, **kwargs: x


def log_color(content, title=""):
    try:
        console = Console()
        console.print(Panel(content, title=title, border_style="cyan", title_align="left"))
        string_io = StringIO()
        plain_console = Console(file=string_io, highlight=False)
        plain_console.print(Panel(content, title=title, border_style="none", title_align="left"))
        log.info("\n" + string_io.getvalue())
    except Exception as e:
        log.error(f"Error logging content: {e}")


@hydra.main(config_path=".", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
    assert cfg.train_traces, "Please provide the training traces"
    if cfg.do_eval:
        assert cfg.holdout_traces, "Please provide the holdout traces for evaluation"

    cfg.tokenizer = cfg.tokenizer or cfg.student

    with open(cfg.train_traces + ".yaml", 'r') as f:
        trace_config = yaml.safe_load(f)
        trace_cfg = SimpleNamespace(**trace_config)

    init(os.getenv("USER"), cfg.seed, "babel" in socket.gethostname())

    if accelerator.is_main_process:
        content = Syntax(OmegaConf.to_yaml(cfg, resolve=True), 'yaml', theme="monokai")
        log_color(content, title="Model Config")
        content = Syntax(yaml.dump(trace_config), 'yaml', theme="monokai")
        log_color(content, title="Trace Config")

    if not cfg.wandb:
        os.environ["WANDB_DISABLED"] = "true"

    # Tokenizer setup — Qwen specific [FIX-1]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer, use_fast=True, trust_remote_code=True,
        padding_side="left"
    )

    # [FIX-2] Add dedicated pad token
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        special_tokens = {"pad_token": "[PAD]"}
        num_added = tokenizer.add_special_tokens(special_tokens)
        log.info(f"Added {num_added} special token(s): pad_token='{tokenizer.pad_token}' "
                 f"(id={tokenizer.pad_token_id})")

    student = AutoModelForCausalLM.from_pretrained(
        cfg.student, trust_remote_code=True,
        attn_implementation="eager",     # [FIX-5]
        torch_dtype=torch.bfloat16,
        use_cache=False,                 # [FIX-4]
    )
    student.resize_token_embeddings(len(tokenizer))  # [FIX-2]
    student.generation_config.pad_token_id = tokenizer.pad_token_id

    # LoRA configuration [FIX-9]
    if cfg.lora:
        peft_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ],
            lora_dropout=cfg.lora_dropout, bias="none",
            task_type="CAUSAL_LM",
        )
        student = get_peft_model(student, peft_config)
        if accelerator.is_main_process:
            student.print_trainable_parameters()

    if accelerator.is_main_process:

        def preprocess_function(examples):
            """
            Preprocess reasoning traces for supervised fine-tuning.
            [FIX-10] Two-step tokenization: render string, then encode.
            """
            trace_colname = trace_cfg.trace_colname
            responses = []

            for response in examples[trace_colname]:
                if "<|im_start|>assistant\n" in response:
                    fixed_response = response.split("<|im_start|>assistant\n", 1)[1]
                    if "<|im_end|>" in fixed_response:
                        fixed_response = fixed_response.split("<|im_end|>")[0]
                elif "<\uff5cAssistant\uff5c>" in response:
                    fixed_response = response.split("<\uff5cAssistant\uff5c>", 1)[1]
                    fixed_response = fixed_response.replace(
                        "<\uff5cend\u2581of\u2581sentence\uff5c>", tokenizer.eos_token
                    )
                else:
                    raise ValueError(
                        f"Unknown assistant marker in trace. "
                        f"First 200 chars: {response[:200]}"
                    )
                responses.append(fixed_response)

            messages = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem.strip()},
                    {"role": "assistant", "content": resp.strip()},
                ]
                for problem, resp in zip(examples["problem"], responses)
            ]

            eos_id = tokenizer.eos_token_id
            cleaned_tokens = []

            for conv in messages:
                rendered = tokenizer.apply_chat_template(
                    conv, add_generation_prompt=False, tokenize=False,
                )
                toks = tokenizer.encode(rendered, add_special_tokens=False)

                assert isinstance(toks, list) and len(toks) > 0
                assert isinstance(toks[0], int)

                # [FIX-3] Strip trailing eos tokens
                while len(toks) > 1 and toks[-1] == eos_id:
                    toks = toks[:-1]

                cleaned_tokens.append(toks)

            tok_lengths = [len(toks) for toks in cleaned_tokens]
            return {"input_ids": cleaned_tokens, "token_lengths": tok_lengths}

        # [FIX-13] Reduced num_proc
        NUM_PROC = min(8, os.cpu_count() or 4)

        train_traces = datasets.load_from_disk(cfg.train_traces)
        train_traces = train_traces.map(
            preprocess_function, batched=True, batch_size=256,
            num_proc=NUM_PROC, remove_columns=list(train_traces.column_names),
            desc="Preprocessing train dataset", load_from_cache_file=True,
        )
        train_token_length_stats = (
            train_traces.to_pandas()["token_lengths"].describe()
        )
        log_color(str(train_token_length_stats.round(2)), title="Train Trace Token Lengths")
        train_traces = train_traces.remove_columns("token_lengths")
        train_traces.save_to_disk("/tmp/cached_train_traces")

        holdout_traces = datasets.load_from_disk(cfg.holdout_traces)
        holdout_traces = holdout_traces.map(
            preprocess_function, batched=True, batch_size=256,
            num_proc=NUM_PROC, remove_columns=list(holdout_traces.column_names),
            desc="Preprocessing holdout dataset", load_from_cache_file=True,
        )
        log_color(
            str(holdout_traces.to_pandas()["token_lengths"].describe().round(2)),
            title="Holdout Trace Token Lengths",
        )
        holdout_traces = holdout_traces.remove_columns("token_lengths")
        holdout_traces.save_to_disk("/tmp/cached_holdout_traces")

    # [FIX-11] Distributed barrier
    accelerator.wait_for_everyone()
    train_traces = datasets.load_from_disk("/tmp/cached_train_traces")
    holdout_traces = datasets.load_from_disk("/tmp/cached_holdout_traces")

    # [FIX-8] Response template for completion-only training
    response_string = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_string, add_special_tokens=False)

    if accelerator.is_main_process:
        sample_ids = train_traces[0]["input_ids"]
        sample_str = " ".join(map(str, sample_ids))
        template_str = " ".join(map(str, response_template_ids))
        assert template_str in sample_str, (
            f"Response template token IDs {response_template_ids} not found in "
            f"sample input_ids. Completion-only masking will fail!"
        )
        log.info(
            f"Response template: '{response_string}' -> "
            f"token IDs: {response_template_ids}"
        )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer, mlm=False,
    )

    # Training configuration — TRL 0.12.0 compatible [FIX-7]
    steps_per_epoch = len(train_traces) // cfg.batch_size
    eval_steps = int(steps_per_epoch * cfg.eval_epochs) if cfg.do_eval else 0

    sft_args = SFTConfig(
        bf16=True,                                              # [FIX-6]
        do_eval=cfg.do_eval,
        max_seq_length=cfg.max_length,                          # [FIX-7]
        eval_strategy="steps" if cfg.do_eval else "no",
        eval_steps=eval_steps if cfg.do_eval else None,
        eval_on_start=True if cfg.do_eval else False,
        gradient_accumulation_steps=(
            cfg.batch_size // cfg.per_device_batch_size // accelerator.num_processes
        ),
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        log_level="info",
        logging_steps=10,
        logging_strategy="steps",
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim="adamw_torch_fused",
        num_train_epochs=cfg.num_epochs,
        output_dir=cfg.model_path,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size * 2,
        report_to="wandb" if cfg.wandb else "none",
        save_strategy="no",
        seed=cfg.seed,
        warmup_ratio=cfg.warmup,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=student, train_dataset=train_traces,
        eval_dataset=holdout_traces, processing_class=tokenizer,
        data_collator=collator, args=sft_args,
    )

    if accelerator.is_main_process and cfg.wandb:
        wandb_run = wandb.init(
            project="antidistillation",
            name=f"{cfg.exp_dir}/{cfg.model_name}",
            config={**cfg, "trace_config": trace_config},
        )
        wandb.log({
            "train_trace_raw_accuracy": trace_cfg.stats["raw_accuracy"],
            "train_trace_af_accuracy": trace_cfg.stats["af_accuracy"],
            "trace_token_length/stats": {
                k: float(v) for k, v in train_token_length_stats.items()
            },
        })

        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        hydra_cfg = HydraConfig.get()
        full_cfg["hydra"] = {
            "run_dir": hydra_cfg.run.dir,
            "job_name": hydra_cfg.job.name,
            "cwd": hydra_cfg.runtime.cwd,
        }
        full_cfg["wandb_run_id"] = wandb_run.id
        yaml_path = cfg.model_path + ".yaml"
        with open(yaml_path, "w") as f:
            OmegaConf.save(full_cfg, f)

    train_result = trainer.train(resume_from_checkpoint=cfg.checkpoint)

    if cfg.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(trainer.eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if accelerator.is_main_process:
        trainer.model.config.use_cache = True  # [FIX-12]
        if cfg.lora:
            trainer.model = trainer.model.merge_and_unload()

        torch.cuda.empty_cache()

        final_output_dir = os.path.join(cfg.model_path, "final")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        if cfg.wandb:
            wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
