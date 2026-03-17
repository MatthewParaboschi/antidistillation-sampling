# Antidistillation Sampling

Defending language models against distillation attacks by modifying the teacher's sampling distribution at inference time. Built as part of ongoing research at the [Toyota Technological Institute at Chicago](https://ttic.edu/) under Professor Mahdi Haghifam.

## What this does

When a capable LLM (the "teacher") generates reasoning traces, a smaller model (the "student") can be fine-tuned on those traces to replicate the teacher's capabilities — this is distillation. Antidistillation sampling (ADS) modifies *how* the teacher generates text so that the traces remain useful to end users but degrade in quality when used as training data for a student model.

The mechanism: at each token, we perturb the teacher's output distribution using gradients from a proxy student model. The perturbation pushes sampling toward tokens that would hurt student learning if the student trains on the resulting traces. This is computed via a finite difference approximation using two copies of the proxy student perturbed in opposite directions along the gradient.

## Pipeline

The system has three stages, each corresponding to a script:

**1. Gradient computation** (`src/save_grad.py`) — Train a proxy student on a small holdout set of clean teacher traces. Save the averaged gradients across the dataset. These gradients tell us which direction in parameter space the student moves when learning from teacher outputs.

**2. Trace generation** (`src/gentraces.py`) — Generate reasoning traces from the teacher model. When `lam=0`, this produces clean traces (baseline). When `lam>0`, the antidistillation logits processor modifies the teacher's token probabilities using the saved gradients, producing "defended" traces. Supports rejection sampling by perplexity threshold to filter low-quality generations.

**3. Student distillation** (`src/distill.py`) — Fine-tune a student model (Qwen2.5-3B-Instruct via LoRA) on the generated traces using completion-only SFT. This is the evaluation step: if antidistillation works, students trained on defended traces should perform worse than students trained on clean traces, while the teacher's own accuracy on the defended traces should remain high.

## Key implementation details

- **KV-cache isolation**: The proxy student models use cached attention during generation for efficiency, but caches must be reset between batches to prevent cross-contamination. This was a non-obvious bug that caused mode collapse before the fix — see `CachedModelWrapper.reset_cache()` in `gentraces.py`.

- **Rejection sampling**: At high lambda values, antidistillation can produce degenerate traces. The rejection sampling mode cycles through the dataset, accepting only traces below a perplexity threshold, until a target number of clean samples is collected. This lets us separate "the defense works" from "the defense destroyed output quality."

- **13 compatibility fixes for Qwen + TRL 0.12.0**: The original codebase targeted Llama. Migrating to Qwen2.5-3B-Instruct required fixes across tokenizer setup, embedding resizing, chat template handling, SFTConfig parameters, and response template encoding. Each fix is tagged `[FIX-1]` through `[FIX-13]` in `distill.py`.

## Models and data

- **Teacher**: Qwen2.5-14B-Instruct (or configurable)
- **Proxy student / Student**: Qwen2.5-3B-Instruct
- **Benchmarks**: GSM8K, MATH (Hendrycks), MMLU
- **Training**: LoRA fine-tuning on RTX A6000 GPUs via SLURM

## Requirements

```
torch >= 2.0
transformers >= 4.47
trl == 0.12.0
accelerate
peft
hydra-core
omegaconf
math-verify
datasets
wandb
rich
pandas
```

## Usage

```bash
# 1. Generate clean holdout traces (lam=0)
accelerate launch src/gentraces.py lam=0 tau=0.6 \
    teacher=Qwen/Qwen2.5-14B-Instruct \
    tokenizer=Qwen/Qwen2.5-3B-Instruct \
    data_split=math_test trace_name=holdout

# 2. Compute proxy student gradients on holdout traces
accelerate launch src/save_grad.py path/to/holdout_config.yaml \
    --proxy_student Qwen/Qwen2.5-3B-Instruct

# 3. Generate antidistillation traces (lam>0)
accelerate launch src/gentraces.py lam=1e-2 tau=0.6 eps=1e-4 \
    teacher=Qwen/Qwen2.5-14B-Instruct \
    proxy_student=Qwen/Qwen2.5-3B-Instruct \
    grad_path=path/to/student_grads.pt \
    data_split=math_train trace_name=defended

# 4. Distill student on defended traces
accelerate launch src/distill.py \
    student=Qwen/Qwen2.5-3B-Instruct \
    train_traces=path/to/defended_traces \
    holdout_traces=path/to/holdout_traces
```

## Repository structure

```
├── README.md
├── requirements.txt
└── src/
    ├── gen_config.yaml      # Hydra config for trace generation
    ├── train_config.yaml    # Hydra config for student distillation
    ├── utils.py             # Shared utilities, prompts, dataset loaders
    ├── gentraces.py         # Trace generation with optional antidistillation
    ├── save_grad.py         # Proxy student gradient computation
    └── distill.py           # Student fine-tuning (LoRA SFT)
```

## Status

Active research — experiments ongoing. This is not a finished product; it is working infrastructure for running antidistillation experiments at scale.
