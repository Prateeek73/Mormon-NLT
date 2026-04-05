"""
model_utils.py — Model and tokenizer loading with quantization support.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_bnb_config(mode: str = "4bit") -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig for the given quantization mode."""
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unknown quantization mode: {mode!r}. Use '4bit' or '8bit'.")


def load_model_and_tokenizer(
    model_id: str,
    quantization: str | None = None,
    attn_implementation: str = "eager",
):
    """
    Load a causal LM and its tokenizer.

    Args:
        model_id: HuggingFace model ID (e.g. 'Qwen/Qwen2.5-3B-Instruct')
        quantization: None (BF16), '4bit', or '8bit'
        attn_implementation: 'eager', 'sdpa', or 'flash_attention_2'

    Returns:
        (model, tokenizer) tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        trust_remote_code=True,
        device_map={"": 0},
        attn_implementation=attn_implementation,
    )

    if quantization is None:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["quantization_config"] = get_bnb_config(quantization)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    return model, tokenizer


def print_model_info(model) -> None:
    """Print total and trainable parameter counts."""
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}  ({100 * trainable / total:.2f}%)")
