"""NusaSynth pipeline configuration: paths, constants, language mappings."""

from pathlib import Path
import os
import re

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "nusax_senti"
GLOTLID_PATH = ROOT / "models" / "glotlid" / "model.bin"
OUTPUT_DIR = ROOT / "outputs" / "synthetic"

# ── Sinyal SV: NusaBERT-large SV-grounding (LP-FT + LoRA PiSSA, dropout 0.06) ──
# Dihasilkan scripts/sv_grounding/p1_sweep/pissa_drop06_full.py (champion_p1_base.py).
# CATATAN: bobot disimpan TANPA config.json dan LoRA BELUM di-merge -> tools.py
# merekonstruksi arsitekturnya (BertModel tanpa pooler + MEAN-POOLING + LoRA PEFT).
# Jangan load pakai BertForSequenceClassification: pooling-nya beda -> prediksi ngawur.
NUSABERT_SEED = os.getenv("NUSABERT_SEED", "42")
NUSABERT_DIR = ROOT / "outputs" / "pissa-drop06-full" / f"seed_{NUSABERT_SEED}"
NUSABERT_BASE_CKPT = "LazarusNLP/NusaBERT-large"
NUSABERT_MAX_LEN = 128
NUSABERT_HIDDEN_DROPOUT = 0.25
NUSABERT_LORA_R = 16
NUSABERT_LORA_ALPHA = 32
NUSABERT_LORA_DROPOUT = 0.06
SENTIMENT_LABELS = ["negative", "neutral", "positive"]  # sorted() -> id 0,1,2


_RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")  # run_id = start timestamp


def latest_run_id() -> str | None:
    """Newest run folder name under OUTPUT_DIR (run_ids are sortable timestamps).

    Only folders matching the run_id timestamp pattern count, so legacy flat
    ``synthetic/<lang>/`` folders are ignored (and synthetic_dir falls back to flat).
    """
    if not OUTPUT_DIR.exists():
        return None
    runs = sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_dir() and _RUN_ID_RE.match(p.name))
    return runs[-1] if runs else None


def synthetic_dir(lang: str, run_id: str | None = None) -> Path:
    """Path to outputs/synthetic/<run_id>/<lang>/ (folder-per-run output).

    run_id resolution: explicit arg > env NUSASYNTH_RUN_ID > latest run folder.
    Pin a run for reproducible analysis (thesis) via the arg or env var.
    """
    rid = run_id or os.getenv("NUSASYNTH_RUN_ID") or latest_run_id()
    return (OUTPUT_DIR / rid / lang) if rid else (OUTPUT_DIR / lang)


# ── LLM (Vertex AI: project/location dioper eksplisit; kredensial via ADC) ───
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")  # flash-lite: judge cukup + murah (~$15/run); ablation 4-bahasa
GEMINI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GEMINI_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GEMINI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in ("true", "1", "yes")
# Temperature per peran (rezim NON-reasoning karena thinking=minimal, 2026-07-15):
# - Generator/Ctx = 0.8: ELTEX 2503.15055 "0.7-0.8 balance diversity with domain consistency, 0.9-1.0
#   deviate from format -> selected 0.8"; penguat Zhu 2506.19262 (data-untuk-fine-tuning, 0.7).
# - Validator SV/LV = 0.0: konvensi LLM-as-judge (G-Eval/ChatEval temp 0 utk determinisme) + exp_temp
#   kami (temp 0 = variance NOL, mutu identik). Detail: research/thinking_cost_ablation/experiments.md.
GEMINI_TEMP_GEN = float(os.getenv("GEMINI_TEMP_GEN", "0.8"))  # generator + contextualizer (diversity, ELTEX)
GEMINI_TEMP_VALIDATOR = float(os.getenv("GEMINI_TEMP_VALIDATOR", "0.0"))  # SV/LV judge (determinisme, G-Eval)

# thinking_level Gemini 3: "minimal"|"low"|"medium"|"high". Default = "minimal" (2026-07-15):
# ablation 4-bahasa -> thinking internal tak menaikkan mutu/override validator, biaya naik monoton
# ($15->$54); Le/Goh/Tang 2603.25176 (LLM-as-judge, non-thinking >= thinking utk tugas klasifikasi eksplisit).
# Field CoT OUTPUT tetap ON (audit/feedback/grounding); yg OFF = thinking INTERNAL. Knob biaya: ~66% output token.
GEMINI_THINK_GEN = os.getenv("GEMINI_THINK_GEN") or "minimal"        # generator + contextualizer
GEMINI_THINK_VALIDATOR = os.getenv("GEMINI_THINK_VALIDATOR") or "minimal"  # SV/LV

# ── Pipeline constants ─────────────────────────────────────────────────────
MAX_RETRY = 2  # validator feedback retry per sentence
PARSE_MAX_RETRY = 2  # structured output parse retry per LLM invoke (handles malformed schema)
BATCH_SIZE = 5
SENTENCES_PER_SEED = 5
DEDUP_THRESHOLD = 0.5  # Jaccard bigram similarity ≥ this → near-duplicate

# ── Target languages ──────────────────────────────────────────────────────
# Selected: GlotLID accuracy > 80% AND NusaBERT F1 > 80%
# TARGET_LANGS = ["sun", "ace", "bjn"]
TARGET_LANGS = ["jav", "sun", "ace", "bjn", "mad", "min", "ban"]
# TARGET_LANGS = "jav"


LANG_NAMES: dict[str, str] = {
    "jav": "Javanese",
    "sun": "Sundanese",
    "ace": "Acehnese",
    "bjn": "Banjarese",
    "mad": "Madurese",
    "min": "Minangkabau",
    "ban": "Balinese",
}

LABEL_ORDER = ["negative", "neutral", "positive"]
