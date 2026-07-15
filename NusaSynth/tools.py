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
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, BertModel, BertTokenizerFast

from NusaSynth.config import (
    GLOTLID_PATH,
    NUSABERT_BASE_CKPT,
    NUSABERT_DIR,
    NUSABERT_HIDDEN_DROPOUT,
    NUSABERT_LORA_ALPHA,
    NUSABERT_LORA_DROPOUT,
    NUSABERT_LORA_R,
    NUSABERT_MAX_LEN,
    NUSABERT_SEED,
    SENTIMENT_LABELS,
)

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

_nusabert_cache: dict[str, tuple] = {}
_nusabert_tokenizer = None


class _BertMeanPoolClassifier(torch.nn.Module):
    """Arsitektur SV-grounding (champion_p1_base.py): BertModel tanpa pooler + mean-pooling
    bermask + linear classifier. BUKAN BertForSequenceClassification ([CLS] pooling)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


def load_nusabert(lang: str):
    """Load NusaBERT-large SV-grounding champion (LP-FT + LoRA PiSSA drop06, seed 42) per bahasa.

    Bobot = state_dict polos (BertLinearProbe = nn.Module, bukan PreTrainedModel): TANPA config.json,
    LoRA BELUM di-merge (535 tensor). Arsitektur direkonstruksi di sini dari NusaSynth.config, lalu bobot
    dituang; LoRA dibiarkan TERPISAH (JANGAN merge — lihat komentar di bawah).

    JANGAN load sebagai BertForSequenceClassification: pooling beda (CLS-pooler vs mean-pool) + bobot LoRA
    di-drop diam-diam -> jalan tanpa error tapi hasilnya SALAH.

    Resep terverifikasi bit-exact (agent_doc/design/sv_model_loading.md): strict=True (535 tensor, nol
    missing/unexpected) -> eval -> inference WAJIB bf16 autocast, TANPA merge_and_unload. Hasil: logits
    bit-exact vs test_logits.npy, F1 7/7 persis deploy_selection.json.

    Return: (model, device). Tokenizer global (_nusabert_tokenizer) — sama untuk semua bahasa.
    """
    global _nusabert_tokenizer
    if lang not in _nusabert_cache:
        weights = NUSABERT_DIR / f"nusabert-large-{lang}" / "best" / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"Bobot NusaBERT tak ada di {weights}")

        cfg = AutoConfig.from_pretrained(NUSABERT_BASE_CKPT)
        cfg.num_labels = len(SENTIMENT_LABELS)
        cfg.label2id = {v: i for i, v in enumerate(SENTIMENT_LABELS)}
        cfg.id2label = dict(enumerate(SENTIMENT_LABELS))
        cfg.hidden_dropout_prob = NUSABERT_HIDDEN_DROPOUT
        cfg.attention_probs_dropout_prob = NUSABERT_HIDDEN_DROPOUT

        model = _BertMeanPoolClassifier(cfg)
        model.bert = get_peft_model(
            model.bert,
            LoraConfig(
                r=NUSABERT_LORA_R,
                lora_alpha=NUSABERT_LORA_ALPHA,
                target_modules=["query", "key", "value"],
                lora_dropout=NUSABERT_LORA_DROPOUT,
                bias="none",
                layers_to_transform=list(range(cfg.num_hidden_layers)),
                layers_pattern="layer",
                use_dora=False,
                init_lora_weights=True,  # nilai init tak penting: langsung ditimpa state_dict
            ),
        )
        # strict=True: kalau arsitektur meleset sedikit pun, GAGAL KERAS di sini (bukan senyap).
        model.load_state_dict(load_file(str(weights)), strict=True)
        # JANGAN merge_and_unload(): (W+BA)x di bf16 != Wx+B(Ax) bit-per-bit -> logits meleset dari
        # test_logits.npy (dibuat pra-merge) -> 4/7 bahasa geser F1. Biarkan LoRA terpisah (ongkos +0.57ms/kal
        # = ~0.03% latensi vs LLM). Detail: agent_doc/design/sv_model_loading.md §4-5.

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval().to(device)
        if _nusabert_tokenizer is None:
            _nusabert_tokenizer = BertTokenizerFast.from_pretrained(NUSABERT_BASE_CKPT)
        _nusabert_cache[lang] = (model, device)
        print(f"NusaBERT SV-grounding '{lang}' loaded (PiSSA drop06 seed {NUSABERT_SEED}, LoRA unmerged) di {device}")

    return _nusabert_cache[lang]


def classify_sentiment_batch(texts: list[str], lang: str, batch_size: int = 64) -> list[dict]:
    """Batch sentiment classification (champion mean-pool + LoRA-unmerged).

    Inference di bf16 autocast — WAJIB, karena training pakai bf16. Dengan fp32 ada sampel borderline
    yang flip (ace: 399/400 vs 400/400; F1 80.27 vs 80.54). dtype = bagian dari resep bit-exact.

    ⚠️ Confidence = softmax MENTAH (TANPA temperature-scaling ÷T) — DISENGAJA, JANGAN diubah:
    sinyal SV = bukti-lunak (LLM boleh override), softmax-mentah = norma sinyal lunak; ECE 4.30%
    terlalu kecil utk menggeser verdict LLM. Memasang ÷T akan meng-invalidate konsistensi ablation
    (semua run pakai confidence mentah). Detail: agent_doc/thesis/sv_confidence_calibration.md.

    Returns: [{"label": "negative"|"neutral"|"positive", "confidence": float}, ...]
    """
    if not texts:
        return []
    model, device = load_nusabert(lang)
    out: list[dict] = []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        for i in range(0, len(texts), batch_size):
            enc = _nusabert_tokenizer(
                texts[i : i + batch_size],
                max_length=NUSABERT_MAX_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            ).float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                idx = int(p.argmax())
                out.append({"label": SENTIMENT_LABELS[idx], "confidence": round(float(p[idx]), 4)})
    return out


def classify_sentiment(text: str, lang: str) -> dict:
    """Klasifikasi satu kalimat (delegasi ke versi batch)."""
    return classify_sentiment_batch([text], lang)[0]


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
