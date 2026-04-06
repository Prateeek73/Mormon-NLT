"""
training_utils.py — Shared helpers for building LoRA configs and SFTConfig objects.
"""
from peft import LoraConfig, TaskType
from trl import SFTConfig


QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Return a LoraConfig for Qwen2.5 causal LM fine-tuning."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules or QWEN_TARGET_MODULES,
    )


def build_lora_training_args(output_dir: str, **overrides) -> SFTConfig:
    """Return SFTConfig tuned for LoRA training on Qwen2.5-3B."""
    defaults = dict(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,          # effective batch = 16
        dataloader_num_workers=0,               # Windows: must be 0 (no fork)
        max_length=512,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
        report_to="tensorboard",
        seed=42,
        remove_unused_columns=False,
    )
    defaults.update(overrides)
    return SFTConfig(**defaults)


def build_fft_training_args(output_dir: str, **overrides) -> SFTConfig:
    """Return SFTConfig tuned for Full Fine-Tuning on Qwen2.5-3B."""
    defaults = dict(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,         # effective batch = 16
        dataloader_num_workers=0,               # Windows: must be 0 (no fork)
        max_length=512,
        num_train_epochs=3,
        learning_rate=2e-5,                     # 10x lower than LoRA
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch_fused",              # fall back to paged_adamw_32bit if OOM
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,                     # FFT checkpoints are large (~6 GB each)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
        report_to="tensorboard",
        seed=42,
        remove_unused_columns=False,
    )
    defaults.update(overrides)
    return SFTConfig(**defaults)


# ── Experiment 2 variants ────────────────────────────────────────────────────────────────────

def build_lora_training_args_exp2(output_dir: str, **overrides) -> SFTConfig:
    """Exp 2: LoRA with 2 epochs max + EarlyStoppingCallback (added in notebook).
    Only change from exp1: num_train_epochs=2, because val loss rose from epoch 2 onward."""
    defaults = dict(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        dataloader_num_workers=0,
        max_length=512,
        num_train_epochs=2,                     # exp1 had 3; val loss rose from epoch 2 onward
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
        report_to="tensorboard",
        seed=42,
        remove_unused_columns=False,
    )
    defaults.update(overrides)
    return SFTConfig(**defaults)


def build_fft_training_args_exp2(output_dir: str, **overrides) -> SFTConfig:
    """Exp 2: FFT with LR=5e-5 (exp1 had 2e-5; both train/val barely moved — underfitting)."""
    defaults = dict(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        dataloader_num_workers=0,
        max_length=512,
        num_train_epochs=3,
        learning_rate=5e-5,                     # exp1 had 2e-5; raised to break plateau
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
        report_to="tensorboard",
        seed=42,
        remove_unused_columns=False,
    )
    defaults.update(overrides)
    return SFTConfig(**defaults)
