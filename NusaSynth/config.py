"""NusaSynth pipeline configuration: paths, constants, language mappings."""

from pathlib import Path
import os
import re

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "nusax_senti"
NUSABERT_DIR = ROOT / "outputs" / "nusabert-sentiment_seed_42"
GLOTLID_PATH = ROOT / "models" / "glotlid" / "model.bin"
OUTPUT_DIR = ROOT / "outputs" / "synthetic"


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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview") # gemini-3.1-flash-lite
GEMINI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GEMINI_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GEMINI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in ("true", "1", "yes")

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
