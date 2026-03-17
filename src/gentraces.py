# -*- coding: utf-8 -*-
# ================================================================================
# GENTRACES.PY - ANTIDISTILLATION SAMPLING TRACE GENERATION
# ================================================================================
# This script generates reasoning traces from language models with optional
# antidistillation sampling (ADS) for protecting against model distillation attacks.
#
# Key functionality:
# 1. Generate clean reasoning traces (when lam=0)
# 2. Generate "poisoned" traces with ADS (when lam>0) that maintain teacher utility
#    but reduce effectiveness for student distillation
# 3. Evaluate model performance on reasoning tasks
# 4. Support for answer forcing to improve trace quality
# 5. Continuous rejection sampling by perplexity threshold
#
# The antidistillation mechanism works by:
# - Using gradients from a proxy student model
# - Modifying the teacher's sampling distribution via finite difference approximation
# - Sampling in directions that would hurt student performance if it learns from traces
# ================================================================================

import ast
import json
import logging
import os
import socket
import random
import shutil
import tempfile
from io import StringIO
from pathlib import Path
import pandas as pd
import datasets
import hydra
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import gather_object
from hydra.core.hydra_config import HydraConfig
from math_verify import parse, verify
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithPadding, LogitsProcessor,
                          LogitsProcessorList)
from transformers import logging as hf_logging

import wandb
from utils import (ANSWER_FORCE_STRING, SYSTEM_PROMPT, init, load_gsm8k,
                   load_hendrycks_math_dataset, load_mmlu)

accelerator = Accelerator()
log = logging.getLogger(__name__)

if not accelerator.is_main_process:
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    datasets.disable_progress_bar()
    tqdm = lambda x, *args, **kwargs: x


def log_perplexity_from_ids(model, full_ids, prompt_len):
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    with torch.no_grad():
        outputs = model(input_ids=full_ids, labels=labels)
    return outputs.loss.item()


def log_color(content, title=""):
    try:
        console = Console()
        console.print(Panel(content, title=title, border_style="cyan", title_align="left"))
        string_io = StringIO()
        plain_console = Console(file=string_io, highlight=False)
        plain_console.print(Panel(content, title=title, border_style="none", title_align="left"))
        log.info("\n" + string_io.getvalue())
    except Exception as e:
        log.info(f"Error logging content: {e}")


def is_correct(example, trace_colname):
    """
    Evaluate if a generated trace produces the correct answer for a math problem.
    Uses math_verify to parse and compare solutions.
    """
    trace = example[trace_colname]
    try:
        soln = parse(example["solution"])
        if ANSWER_FORCE_STRING in trace:
            parts = trace.split(ANSWER_FORCE_STRING)
            alt_ans1 = ANSWER_FORCE_STRING.join(parts[:-1])
            alt_ans2 = parts[-1]
            res = any(verify(soln, parse(ans)) for ans in [trace, alt_ans1, alt_ans2])
        else:
            res = verify(soln, parse(trace))
    except:
        print(f"Error parsing trace: {trace} and comparing with solution: {example['solution']}")
        res = False
    return {"is_correct": res}


class CachedModelWrapper:
    """
    Wrapper for language models that implements KV-cache optimization.
    Avoids recomputing attention for previously processed tokens during
    incremental generation — crucial for antidistillation sampling where
    we call the student models multiple times per token.
    """
    def __init__(self, model):
        self.model = model
        self.past_key_values = None
        self.last_position = 0

    def reset_cache(self):
        self.past_key_values = None
        self.last_position = 0

    def __call__(self, input_ids, attention_mask=None):
        if self.past_key_values is None or input_ids.shape[1] <= self.last_position:
            outputs = self.model(
                input_ids, attention_mask=attention_mask,
                use_cache=True, return_dict=True
            )
            self.past_key_values = outputs.past_key_values
            self.last_position = input_ids.shape[1]
            return outputs.logits

        new_token = input_ids[:, -1:]
        outputs = self.model(
            new_token, attention_mask=attention_mask,
            use_cache=True, past_key_values=self.past_key_values,
            return_dict=True
        )
        self.past_key_values = outputs.past_key_values
        self.last_position += 1
        return outputs.logits


@hydra.main(config_path=".", config_name="gen_config", version_base="1.3")
def main(cfg: DictConfig):

    cfg.antidistillation = cfg.lam != 0
    cfg.wandb_lam = 1e-8 if cfg.lam == 0 else cfg.lam

    rejection_sampling = cfg.get("rejection_sampling", False)
    rejection_threshold = cfg.get("rejection_threshold", 8.017)
    target_samples = cfg.get("target_samples", 500)

    if cfg.antidistillation:
        assert cfg.proxy_student is not None, "Proxy student model must be specified for antidistillation"
        assert cfg.grad_path is not None, "Grad path must be specified for antidistillation"

    if cfg.trace_name == "REPLACE_ME":
        raise ValueError("Trace name must be specified")

    init(os.getenv("USER"), cfg.seed, "babel" in socket.gethostname())

    if accelerator.is_main_process:
        content = Syntax(OmegaConf.to_yaml(cfg, resolve=True), 'yaml', theme="monokai")
        log_color(content, title="Config")
        if rejection_sampling:
            log_color(
                f"Rejection sampling ENABLED\n"
                f"  threshold (log-PPL): {rejection_threshold}\n"
                f"  target accepted:     {target_samples}",
                title="Rejection Sampling Config"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer, use_fast=True, fast_tokenizer=True,
        trust_remote_code=True, padding_side="left",
    )

    bos_token = tokenizer.bos_token or ""
    eos_token = tokenizer.eos_token or ""

    if "llama" in cfg.tokenizer.lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token
    else:
        eos_token = tokenizer.eos_token
        bos_token = tokenizer.bos_token or ""
        special_tokens = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens)

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher, trust_remote_code=True,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        use_cache=True,
    ).to(accelerator.device)
    teacher.generation_config.pad_token_id = tokenizer.pad_token_id
    teacher.resize_token_embeddings(len(tokenizer))

    if cfg.antidistillation:
        student = CachedModelWrapper(AutoModelForCausalLM.from_pretrained(
            cfg.proxy_student, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.float16,
            use_cache=True,
        ).to(accelerator.device))

        dstudent = CachedModelWrapper(AutoModelForCausalLM.from_pretrained(
            cfg.proxy_student, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.float16,
            use_cache=True,
        ).to(accelerator.device))

        student.model.resize_token_embeddings(len(tokenizer))
        dstudent.model.resize_token_embeddings(len(tokenizer))

        grads = torch.load(cfg.grad_path, map_location='cpu')
        if accelerator.is_main_process:
            log.info(f"Using eps: {cfg.eps}")

        used_grads = set()
        param_sq, grad_sq, num_params = 0, 0, 0
        for name, param in student.model.named_parameters():
            module_name = 'module.' + name
            if module_name in grads:
                grad = grads[module_name].to(param.device, dtype=torch.float32)
                param.data = (param.data.to(torch.float32) + cfg.eps * grad).to(param.data.dtype)
                param_sq += torch.sum(param.data.to(torch.float32) ** 2).item()
                grad_sq += torch.sum(grad ** 2).item()
                num_params += torch.numel(param.data)
                used_grads.add(module_name)

        assert used_grads == set(grads.keys())
        if accelerator.is_main_process:
            log_color(f"{param_sq ** 0.5 / num_params ** 0.5:.2e}", title="Param RMSNorm")
            log_color(f"{grad_sq ** 0.5 / num_params ** 0.5:.2e}", title="Grad RMSNorm")

        used_grads = set()
        for name, param in dstudent.model.named_parameters():
            module_name = 'module.' + name
            if module_name in grads:
                grad = grads[module_name].to(param.device, dtype=torch.float32)
                param.data = (param.data.to(torch.float32) - cfg.eps * grad).to(param.data.dtype)
                used_grads.add(module_name)

        assert used_grads == set(grads.keys())
        del grads
        if accelerator.is_main_process:
            log.info('Calculated grads')

    if "gsm8k" in cfg.data_split:
        dataset = load_gsm8k(split=cfg.data_split.split("_")[1])
    elif "math" in cfg.data_split:
        dataset = load_hendrycks_math_dataset(split=cfg.data_split.split("_")[1])
    elif "mmlu" in cfg.data_split:
        dataset = load_mmlu(split=cfg.data_split.split("_")[1])
    else:
        raise ValueError(f"Unknown dataset and split: {cfg.data_split}")

    if cfg.max_samples is not None:
        dataset = dataset.take(min(cfg.max_samples, len(dataset)))

    if accelerator.is_main_process:
        def preprocess_function(examples):
            prompt_texts = []
            for problem in examples["problem"]:
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem.strip()+"\n"}]
                prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt_texts.append(prompt_text)
            tokenized = tokenizer(prompt_texts, add_special_tokens=False, padding=False, truncation=False)
            input_ids = tokenized["input_ids"]
            seq_lengths = [len(ids) for ids in input_ids]
            return {"input_ids": input_ids, "seq_lengths": seq_lengths}

        proc_dataset = dataset.map(preprocess_function, batched=True, num_proc=1, desc="Preprocessing dataset", load_from_cache_file=False)
        log_color(tokenizer.decode(proc_dataset[0]['input_ids']), title="Example Input")
        seq_length_stats = proc_dataset.to_pandas()["seq_lengths"].describe()
        log_color(str(seq_length_stats.round(2)), title="Sequence Lengths")
        proc_dataset = proc_dataset.remove_columns("seq_lengths")
        proc_dataset.save_to_disk("/tmp/cached_proc_dataset")

    accelerator.wait_for_everyone()
    proc_dataset = datasets.load_from_disk("/tmp/cached_proc_dataset")

    num_shards = accelerator.num_processes
    shard_id = accelerator.process_index
    dataset_shard = proc_dataset.shard(num_shards=num_shards, index=shard_id)
    ptds_shard = dataset_shard.remove_columns(dataset.column_names)

    class LogprobsModifier(LogitsProcessor):
        """
        Logits processor implementing antidistillation sampling.
        Modifies the teacher's token distribution via finite difference
        approximation using perturbed proxy student models.
        """
        def __init__(self, lam, eps, attention_mask, repetition_penalty):
            super().__init__()
            self.lam = lam
            self.eps = eps
            self.attention_mask = attention_mask
            self.repetition_penalty = repetition_penalty

        def __call__(self, input_ids, scores):
            attention_mask = F.pad(self.attention_mask, pad=(0, input_ids.shape[1]-self.attention_mask.shape[1]), value=1)
            out_target = student(input_ids=input_ids, attention_mask=attention_mask)[:, -1]
            out_Dtarget = dstudent(input_ids=input_ids, attention_mask=attention_mask)[:, -1]
            ad_term = (self.lam / (2*self.eps)) * (out_target.float() - out_Dtarget.float())
            scores = scores.float() + ad_term
            return scores

    def generate_batch(batch_dict):
        """
        Generate traces for a single batch with proper cache isolation.
        Resets student/dstudent KV caches before generation to prevent
        cross-batch cache contamination.
        """
        batch = {key: value.to(accelerator.device) for key, value in batch_dict.items()}

        if cfg.antidistillation:
            student.reset_cache()
            dstudent.reset_cache()

        with torch.inference_mode():
            outputs = teacher.generate(
                **batch,
                max_new_tokens=None, max_length=cfg.max_length,
                temperature=cfg.tau if cfg.tau > 0 else None,
                do_sample=True if cfg.tau > 0 else False,
                top_p=0.95 if cfg.tau > 0 else None,
                logits_processor=(
                    LogitsProcessorList([LogprobsModifier(cfg.lam, cfg.eps, batch["attention_mask"], cfg.repetition_penalty)])
                    if cfg.antidistillation else None
                ),
                renormalize_logits=True if cfg.antidistillation else False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0 if cfg.antidistillation else cfg.repetition_penalty,
            )
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            results = []
            for i, text in enumerate(generated_texts):
                text = text.replace(tokenizer.pad_token, "")
                full_ids = outputs[i:i+1]
                prompt_len = batch["input_ids"][i:i+1].shape[1]
                ppl = log_perplexity_from_ids(teacher, full_ids, prompt_len)
                results.append((text, ppl))

        return results

    if not rejection_sampling:
        dataloader = DataLoader(
            ptds_shard, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        perplexities = []
        traces = []
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"tau={cfg.tau:.2e}, lam={cfg.lam:.2e}, eps={cfg.eps:.2e}"):
            results = generate_batch(batch)
            for text, ppl in results:
                traces.append(text)
                perplexities.append(ppl)

        if accelerator.is_main_process:
            log_color(traces[0], title="First trace")

        final_dataset_shard = dataset_shard

    else:
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        n_dataset = len(ptds_shard)
        target_per_shard = target_samples

        accepted_traces = []
        accepted_ppls = []
        accepted_meta_indices = []

        total_generated = 0
        total_rejected = 0
        cycle_num = 0

        if accelerator.is_main_process:
            log_color(
                f"Dataset shard size: {n_dataset}\n"
                f"Target per shard:   {target_per_shard}\n"
                f"Batch size:         {cfg.batch_size}",
                title="Rejection Sampling Starting"
            )

        while len(accepted_traces) < target_per_shard:
            cycle_num += 1
            indices = list(range(n_dataset))
            random.shuffle(indices)

            for batch_start in range(0, n_dataset, cfg.batch_size):
                if len(accepted_traces) >= target_per_shard:
                    break

                batch_end = min(batch_start + cfg.batch_size, n_dataset)
                batch_indices = indices[batch_start:batch_end]

                batch_rows = [ptds_shard[int(idx)] for idx in batch_indices]
                batch = collator(batch_rows)

                results = generate_batch(batch)
                total_generated += len(results)

                for j, (text, ppl) in enumerate(results):
                    if len(accepted_traces) >= target_per_shard:
                        break
                    if ppl <= rejection_threshold:
                        accepted_traces.append(text)
                        accepted_ppls.append(ppl)
                        accepted_meta_indices.append(batch_indices[j])
                    else:
                        total_rejected += 1

                if accelerator.is_main_process and total_generated % (cfg.batch_size * 4) == 0:
                    accept_rate = len(accepted_traces) / max(total_generated, 1) * 100
                    log.info(
                        f"[Cycle {cycle_num}] "
                        f"Accepted: {len(accepted_traces)}/{target_per_shard} | "
                        f"Generated: {total_generated} | "
                        f"Rejected: {total_rejected} | "
                        f"Accept rate: {accept_rate:.1f}%"
                    )

            if accelerator.is_main_process:
                accept_rate = len(accepted_traces) / max(total_generated, 1) * 100
                log_color(
                    f"Cycle {cycle_num} complete.\n"
                    f"  Accepted so far: {len(accepted_traces)}/{target_per_shard}\n"
                    f"  Total generated: {total_generated}\n"
                    f"  Total rejected:  {total_rejected}\n"
                    f"  Accept rate:     {accept_rate:.1f}%",
                    title="Cycle Summary"
                )

        accepted_traces = accepted_traces[:target_per_shard]
        accepted_ppls = accepted_ppls[:target_per_shard]
        accepted_meta_indices = accepted_meta_indices[:target_per_shard]

        traces = accepted_traces
        perplexities = accepted_ppls

        if accelerator.is_main_process:
            log_color(traces[0], title="First accepted trace")
            final_accept_rate = len(traces) / max(total_generated, 1) * 100
            log_color(
                f"Rejection sampling COMPLETE.\n"
                f"  Final accepted:  {len(traces)}\n"
                f"  Total generated: {total_generated}\n"
                f"  Total rejected:  {total_rejected}\n"
                f"  Final accept rate: {final_accept_rate:.1f}%\n"
                f"  Mean PPL (accepted): {sum(perplexities)/len(perplexities):.4f}\n"
                f"  Max  PPL (accepted): {max(perplexities):.4f}\n"
                f"  All PPL <= {rejection_threshold}: {all(p <= rejection_threshold for p in perplexities)}",
                title="Rejection Sampling Results"
            )

        final_dataset_shard = dataset_shard.select(accepted_meta_indices)

    dataset_shard = final_dataset_shard.add_column(cfg.trace_colname, traces)
    dataset_shard = dataset_shard.add_column(cfg.trace_colname + "_logppl", perplexities)

    if cfg.antidistillation:
        del student, dstudent
        torch.cuda.empty_cache()

    dataset_shard = dataset_shard.map(
        is_correct, fn_kwargs={"trace_colname": cfg.trace_colname},
        desc="Checking raw correctness"
    )
    dataset_shard = dataset_shard.rename_columns({"is_correct": "is_raw_correct"})

    response_strings = {
        "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "qwen": "<|im_start|>assistant\n",
        "r1": "<\uFF5CAssistant\uFF5C>"
    }
    response_string = None
    for _, value in response_strings.items():
        if value in traces[0]:
            response_string = value
            break
    if response_string is None:
        raise ValueError("Response string not found in tokenizer chat template")

    if cfg.answer_force:
        traces_ = []
        for text in traces:
            if "</think>" in text.split(response_string)[-1]:
                traces_.append(text + ANSWER_FORCE_STRING)
            else:
                traces_.append(text + "\n</think>" + ANSWER_FORCE_STRING)

        af_batch_size = cfg.batch_size // 2
        traces_batched = [traces_[i:i+af_batch_size] for i in range(0, len(traces_), af_batch_size)]
        traces_af = []
        for batch in tqdm(traces_batched, total=len(traces_batched)):
            batch = [text.replace(bos_token, '', 1).replace(eos_token, '') for text in batch]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(accelerator.device)
            with torch.inference_mode():
                outputs = teacher.generate(
                    **inputs, do_sample=False, temperature=None, top_p=None,
                    logits_processor=None, renormalize_logits=False,
                    max_new_tokens=32, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                for text in generated_texts:
                    text = text.replace(tokenizer.pad_token, "")
                    if not text.endswith(tokenizer.eos_token):
                        text = text + tokenizer.eos_token
                    traces_af.append(text)

        if accelerator.is_main_process:
            log_color(traces_af[0], title="First af trace")
        del teacher
        torch.cuda.empty_cache()

        dataset_shard = dataset_shard.add_column(cfg.trace_colname+"_af", traces_af)
        dataset_shard = dataset_shard.map(
            is_correct, fn_kwargs={"trace_colname": cfg.trace_colname+"_af"},
            desc="Checking af correctness"
        )
        dataset_shard = dataset_shard.rename_columns({"is_correct": "is_af_correct"})

    tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_ds_"))
    shard_path = tmp_dir / f"shard_rank_{accelerator.process_index:05d}"
    dataset_shard.save_to_disk(shard_path)

    accelerator.wait_for_everyone()

    all_paths = gather_object([shard_path])
    if accelerator.is_main_process:
        trace_dataset = datasets.concatenate_datasets([datasets.load_from_disk(path) for path in all_paths])
        trace_dataset.save_to_disk(cfg.trace_path)
        trace_dataset.to_parquet(cfg.trace_path + ".parquet")

        for path in all_paths:
            shutil.rmtree(path, ignore_errors=True)

        example_row = trace_dataset[random.randint(0, len(trace_dataset)-1)]
        log_color(example_row["problem"], title="Example Problem")
        log_color(example_row["solution"], title="Example Solution")
        log_color(example_row[cfg.trace_colname], title=f"Example Trace [tau={cfg.tau:.2e}, lam={cfg.lam:.2e}, eps={cfg.eps:.2e}]")
        if cfg.answer_force:
            log_color(example_row[cfg.trace_colname + "_af"], title=f"Example AF Trace")

        trace_df = trace_dataset.to_pandas()
        trace_len_stats = {k:float(v) for k,v in trace_df[cfg.trace_colname].map(lambda x: len(tokenizer.encode(x))).describe().items()}
        ppl_stats = {k:float(v) for k,v in trace_df[cfg.trace_colname + "_logppl"].describe().items()}
        log_color(str(pd.Series(ppl_stats).round(4)), title="Perplexity Statistics")

        raw_accuracy = float(trace_df["is_raw_correct"].mean())
        af_accuracy = float(trace_df["is_af_correct"].mean()) if cfg.answer_force else None

        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        hydra_cfg = HydraConfig.get()
        full_cfg["hydra"] = {
            "run_dir": hydra_cfg.run.dir,
            "job_name": hydra_cfg.job.name,
            "cwd": hydra_cfg.runtime.cwd,
        }
        full_cfg["stats"] = {
            "raw_accuracy": raw_accuracy,
            "trace_len_stats": trace_len_stats,
            "ppl_stats": ppl_stats,
        }
        if af_accuracy is not None:
            full_cfg["stats"]["af_accuracy"] = af_accuracy

        if rejection_sampling:
            full_cfg["stats"]["rejection_sampling"] = {
                "total_generated": total_generated,
                "total_rejected": total_rejected,
                "total_accepted": len(traces),
                "acceptance_rate": len(traces) / max(total_generated, 1),
                "threshold": rejection_threshold,
                "all_below_threshold": all(p <= rejection_threshold for p in perplexities),
                "max_ppl_accepted": max(perplexities),
                "mean_ppl_accepted": sum(perplexities) / len(perplexities),
            }

        yaml_path = cfg.trace_path + ".yaml"
        with open(yaml_path, "w") as f:
            OmegaConf.save(full_cfg, f)
        log.info(f"Configuration saved to {yaml_path}")

        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        with open(cfg.trace_registry, "a") as f:
            f.write(json.dumps(flatten_dict(full_cfg)) + "\n")

        content = Syntax(OmegaConf.to_yaml(full_cfg, resolve=True), 'yaml', theme="monokai")
        log_color(content, title="Final Config")

        if cfg.use_wandb and cfg.teacher_cfg:
            with open(cfg.teacher_cfg, "r") as f:
                teacher_cfg = yaml.safe_load(f)
            wandb_run_id = teacher_cfg.get("wandb_run_id")
            if wandb_run_id is None:
                raise ValueError("wandb is true but wandb_run_id not found in teacher config")
            wandb.init(project="antidistillation", id=wandb_run_id, resume="allow")
            log_dict = {"teacher_raw_accuracy" if cfg.is_teacher else "student_raw_accuracy": raw_accuracy}
            if af_accuracy is not None:
                log_dict["teacher_af_accuracy" if cfg.is_teacher else "student_af_accuracy"] = af_accuracy
            if rejection_sampling:
                prefix = "teacher" if cfg.is_teacher else "student"
                log_dict[f"{prefix}_rejection_acceptance_rate"] = len(traces) / max(total_generated, 1)
            wandb.log(log_dict)

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
