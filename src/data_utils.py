"""
data_utils.py — Dataset loading, formatting, and splitting utilities.
"""
import json
from pathlib import Path
from datasets import load_dataset, Dataset


SYSTEM_PROMPT_MOD2SHAK = (
    "You are an expert translator of Modern English into Shakespearean English. "
    "Translate the following modern English text into authentic Shakespearean style, "
    "preserving the meaning while using appropriate Early Modern English vocabulary, "
    "grammar, and poetic diction."
)

SYSTEM_PROMPT_SHAK2MOD = (
    "You are an expert translator of Shakespearean English into Modern English. "
    "Translate the following Shakespearean text into clear, contemporary Modern English "
    "that accurately conveys the original meaning."
)


def make_record(modern: str, shakespeare: str, direction: str = "mod2shak") -> dict:
    """Build a single chat-format record for SFT training."""
    if direction == "mod2shak":
        user_content  = modern
        asst_content  = shakespeare
        system_prompt = SYSTEM_PROMPT_MOD2SHAK
    else:
        user_content  = shakespeare
        asst_content  = modern
        system_prompt = SYSTEM_PROMPT_SHAK2MOD
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ]
    }


def load_jsonl_as_hf_dataset(path: str | Path) -> Dataset:
    """Load a .jsonl file as a HuggingFace Dataset (single train split)."""
    return load_dataset("json", data_files=str(path), split="train")


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Write a list of dicts as a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_training_records(df, bidirectional: bool = True) -> list[dict]:
    """
    Convert a DataFrame with 'modern' and 'shakespeare' columns into
    chat-format records suitable for SFTTrainer.

    Args:
        df: pandas DataFrame with columns ['modern', 'shakespeare']
        bidirectional: if True, create both mod→shak and shak→mod records

    Returns:
        List of record dicts with 'messages' key.
    """
    records = []
    for _, row in df.iterrows():
        records.append(make_record(row["modern"], row["shakespeare"], "mod2shak"))
        if bidirectional:
            records.append(make_record(row["modern"], row["shakespeare"], "shak2mod"))
    return records
