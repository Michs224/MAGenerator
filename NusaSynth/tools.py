"""Programmatic tools: NusaBERT (sentiment classification) + GlotLID (language ID).

These are NOT LLM tool calls — they run inference locally and return
structured results that are fed into the LLM validator prompts.
"""

from __future__ import annotations

import fasttext
import fasttext.FastText
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import pipeline as hf_pipeline

from NusaSynth.config import GLOTLID_PATH, NUSABERT_DIR

# Suppress fasttext stderr warnings
fasttext.FastText.eprint = lambda x: None

# Fix NumPy 2.x compat: fasttext uses np.array(obj, copy=False) which errors in NumPy 2.x
def _patched_ft_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    if isinstance(text, list):
        all_labels, all_probs = [], []
        for t in text:
            lbl, prob = _patched_ft_predict(self, t, k, threshold, on_unicode_error)
            all_labels.append(lbl)
            all_probs.append(prob)
        return all_labels, all_probs
    pairs = self.f.predict(text, k, threshold, on_unicode_error)
    if pairs:
        probs, labels = zip(*pairs)
    else:
        probs, labels = (), ()
    return labels, np.asarray(probs)


fasttext.FastText._FastText.predict = _patched_ft_predict


# ── NusaBERT sentiment classifier ─────────────────────────────────────────

_nusabert_cache: dict[str, object] = {}


def load_nusabert(lang: str):
    """Load NusaBERT-large fine-tuned model for a language.

    Model path: outputs/nusabert-sentiment_seed_42/nusabert-large-{lang}/best/
    Returns: transformers TextClassificationPipeline
    """
    if lang not in _nusabert_cache:
        model_path = NUSABERT_DIR / f"nusabert-large-{lang}" / "best"
        if not model_path.exists():
            raise FileNotFoundError(
                f"NusaBERT model not found at {model_path}. "
                f"Run finetune_nusabert.py first for language '{lang}'."
            )

        device = 0 if torch.cuda.is_available() else -1
        _nusabert_cache[lang] = hf_pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device,
        )
        print(f"NusaBERT '{lang}' loaded di device={device}")

    return _nusabert_cache[lang]


def classify_sentiment(text: str, lang: str) -> dict:
    """Classify sentiment using NusaBERT-large fine-tuned per language.

    Returns: {"label": "negative"|"neutral"|"positive", "confidence": float}
    """
    clf = load_nusabert(lang)
    result = clf(text, truncation=True, max_length=128)[0]
    return {
        "label": result["label"].lower(),
        "confidence": round(result["score"], 4),
    }


def classify_sentiment_batch(texts: list[str], lang: str) -> list[dict]:
    """Batch sentiment classification (more efficient for multiple texts)."""
    clf = load_nusabert(lang)
    results = clf(texts, truncation=True, max_length=128, batch_size=16)
    return [
        {"label": r["label"].lower(), "confidence": round(r["score"], 4)}
        for r in results
    ]


# ── GlotLID language identification ───────────────────────────────────────

_glotlid: fasttext.FastText._FastText | None = None


def load_glotlid():
    """Load GlotLID fasttext model, downloading from HuggingFace if not cached."""
    global _glotlid
    if _glotlid is None:
        if not GLOTLID_PATH.exists():
            print("Model GlotLID belum ada lokal — download dari HuggingFace...")
            GLOTLID_PATH.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
            import shutil
            shutil.copy(downloaded, GLOTLID_PATH)
            print(f"GlotLID disimpan ke {GLOTLID_PATH}")
        _glotlid = fasttext.load_model(str(GLOTLID_PATH))
        print(f"GlotLID loaded dari {GLOTLID_PATH}")
    return _glotlid


def identify_language(text: str) -> dict:
    """Identify language using GlotLID.

    Returns: {"detected_lang": str (ISO code), "confidence": float}
    """
    model = load_glotlid()
    text_clean = text.replace("\n", " ").strip()
    labels, scores = model.predict(text_clean, k=1)
    # GlotLID returns "__label__jav_Latn" format
    lang_raw = labels[0].replace("__label__", "")
    # Extract ISO code (before _Script suffix)
    lang_code = lang_raw.split("_")[0] if "_" in lang_raw else lang_raw
    return {
        "detected_lang": lang_code,
        "confidence": round(float(scores[0]), 4),
    }


def identify_language_batch(texts: list[str]) -> list[dict]:
    """Batch language identification."""
    return [identify_language(t) for t in texts]


# ── Jaccard bigram deduplication ─────────────────────────────────────────


def jaccard_bigram(s1: str, s2: str) -> float:
    """Compute Jaccard similarity on word bigram sets.

    J(A, B) = |A ∩ B| / |A ∪ B| where A, B are sets of word bigrams.
    Standard near-duplicate detection metric (Broder, 1997).
    """
    words1 = s1.lower().split()
    words2 = s2.lower().split()
    bg1 = set(zip(words1, words1[1:]))
    bg2 = set(zip(words2, words2[1:]))
    if not bg1 or not bg2:
        return 0.0
    return len(bg1 & bg2) / len(bg1 | bg2)
