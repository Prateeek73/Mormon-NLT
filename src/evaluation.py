"""
evaluation.py — BLEU and ChrF evaluation utilities using sacrebleu.
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF

from .inference import translate


_BLEU = BLEU(effective_order=True)
_CHRF = CHRF()


def compute_bleu_scores(hypotheses: list[str], references: list[str]) -> tuple[dict, list[float]]:
    """
    Compute corpus BLEU, ChrF, and sentence-level BLEU statistics.

    Args:
        hypotheses: model-generated translations
        references: gold reference translations

    Returns:
        (scores_dict, sentence_bleu_list)
        scores_dict contains: corpus_bleu, corpus_chrf, sent_bleu_mean/std/p25/p50/p75
    """
    corpus_bleu = _BLEU.corpus_score(hypotheses, [references])
    corpus_chrf = _CHRF.corpus_score(hypotheses, [references])

    sentence_bleus = [
        _BLEU.sentence_score(h, [r]).score
        for h, r in zip(hypotheses, references)
    ]

    scores = {
        "corpus_bleu":    round(corpus_bleu.score, 2),
        "corpus_chrf":    round(corpus_chrf.score, 2),
        "sent_bleu_mean": round(float(np.mean(sentence_bleus)), 2),
        "sent_bleu_std":  round(float(np.std(sentence_bleus)), 2),
        "sent_bleu_p25":  round(float(np.percentile(sentence_bleus, 25)), 2),
        "sent_bleu_p50":  round(float(np.percentile(sentence_bleus, 50)), 2),
        "sent_bleu_p75":  round(float(np.percentile(sentence_bleus, 75)), 2),
    }
    return scores, sentence_bleus


def run_batch_inference(
    model,
    tokenizer,
    test_jsonl_path: str | Path,
    direction: str = "mod2shak",
    n_samples: int | None = None,
    max_new_tokens: int = 256,
) -> tuple[list[str], list[str]]:
    """
    Run model inference over a JSONL test file and collect hypotheses and references.

    Args:
        model: loaded HuggingFace model
        tokenizer: corresponding tokenizer
        test_jsonl_path: path to .jsonl file with 'messages' records
        direction: 'mod2shak' or 'shak2mod'
        n_samples: limit to first N samples (None = all)
        max_new_tokens: generation budget

    Returns:
        (hypotheses, references)
    """
    with open(test_jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if n_samples is not None:
        records = records[:n_samples]

    hypotheses, references = [], []
    for rec in tqdm(records, desc=f"Translating ({direction})"):
        user_msg = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        ref_msg  = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        hyp = translate(model, tokenizer, user_msg, direction=direction,
                        max_new_tokens=max_new_tokens)
        hypotheses.append(hyp)
        references.append(ref_msg)

    return hypotheses, references


def sentence_bleu(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU for a single hypothesis/reference pair."""
    return round(_BLEU.sentence_score(hypothesis, [reference]).score, 2)
