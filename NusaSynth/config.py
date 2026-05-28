"""NusaSynth pipeline configuration: paths, constants, language mappings."""

from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "nusax_senti"
NUSABERT_DIR = ROOT / "outputs" / "nusabert-sentiment_seed_42"
GLOTLID_PATH = ROOT / "models" / "glotlid" / "model.bin"
OUTPUT_DIR = ROOT / "outputs" / "synthetic"

# ── LLM ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# ── Pipeline constants ─────────────────────────────────────────────────────
MAX_RETRY = 2  # validator feedback retry per sentence
PARSE_MAX_RETRY = 2  # structured output parse retry per LLM invoke (handles malformed schema)
BATCH_SIZE = 5
SENTENCES_PER_SEED = 5
DEDUP_THRESHOLD = 0.5  # Jaccard bigram similarity ≥ this → near-duplicate

# ── Target languages ──────────────────────────────────────────────────────
# Selected: GlotLID accuracy > 80% AND NusaBERT F1 > 80%
# TARGET_LANGS = ["sun", "ace", "bjn"]
TARGET_LANGS = ["mad", "min", "ban"]
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
