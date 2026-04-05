"""
inference.py — Single-example translation inference helper.
"""
import torch
from .data_utils import SYSTEM_PROMPT_MOD2SHAK, SYSTEM_PROMPT_SHAK2MOD


def translate(
    model,
    tokenizer,
    text: str,
    direction: str = "mod2shak",
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Translate a single text using the given model and tokenizer.

    Args:
        model: loaded HuggingFace causal LM (LoRA or FFT)
        tokenizer: corresponding tokenizer
        text: source text to translate
        direction: 'mod2shak' (modern→Shakespeare) or 'shak2mod' (Shakespeare→modern)
        max_new_tokens: maximum tokens to generate
        do_sample: False for greedy (BLEU eval), True for temperature sampling (demo)
        temperature: sampling temperature (only used when do_sample=True)
        repetition_penalty: penalizes repeated n-grams

    Returns:
        Translated string (stripped of special tokens and leading/trailing whitespace)
    """
    system_prompt = SYSTEM_PROMPT_MOD2SHAK if direction == "mod2shak" else SYSTEM_PROMPT_SHAK2MOD
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def batch_translate(
    model,
    tokenizer,
    texts: list[str],
    direction: str = "mod2shak",
    max_new_tokens: int = 256,
) -> list[str]:
    """Translate a list of texts one by one (greedy, for evaluation)."""
    return [
        translate(model, tokenizer, t, direction=direction,
                  max_new_tokens=max_new_tokens, do_sample=False)
        for t in texts
    ]
