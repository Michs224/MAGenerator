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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Pipeline constants ─────────────────────────────────────────────────────
MAX_RETRY = 2
BATCH_SIZE = 5
SENTENCES_PER_SEED = 5
DEDUP_THRESHOLD = 0.5  # Jaccard bigram similarity ≥ this → near-duplicate

# ── Target languages ──────────────────────────────────────────────────────
# Selected: GlotLID accuracy > 80% AND NusaBERT F1 > 80%
# TARGET_LANGS = ["jav", "sun", "ace", "bjn", "mad", "min", "ban"]
TARGET_LANG = "jav"

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
