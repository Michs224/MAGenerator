"""Audit pasca-run Single-LLM: duplication rate + GlotLID purity + NusaBERT label-drift.

Metrik OUTPUT (sifat dataset), dihitung apa adanya. Dedup DIUKUR (bukan diterapkan),
meniru greedy cross-batch save_results: kalimat baru dibanding hanya ke yang sudah 'disimpan'.

  PYTHONUTF8=1 uv run python scripts/single_llm_audit.py
  SINGLE_LLM_RUN_ID=<ts> PYTHONUTF8=1 uv run python scripts/single_llm_audit.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from NusaSynth.config import (
    DEDUP_THRESHOLD,
    SINGLE_LLM_OUTPUT_DIR,
    TARGET_LANGS,
    latest_single_llm_run_id,
)
from NusaSynth.run_pipeline import free_nusabert_cache
from NusaSynth.tools import classify_sentiment_batch, identify_language_batch, jaccard_bigram


def dedup_rate(texts: list[str]) -> tuple[int, int]:
    """Greedy cross-batch dedup (meniru save_results): return (n_dup, n_total)."""
    kept: list[str] = []
    dup = 0
    for t in texts:
        if any(jaccard_bigram(t, ref) >= DEDUP_THRESHOLD for ref in kept):
            dup += 1
        else:
            kept.append(t)
    return dup, len(texts)


def audit_language(lang: str, run_dir: Path) -> dict | None:
    csv_path = run_dir / lang / "synthetic.csv"
    if not csv_path.exists():
        print(f"  {lang}: (tak ada synthetic.csv, skip)")
        return None
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    n = len(texts)
    if not n:
        return None

    dup, _ = dedup_rate(texts)
    on_target = sum(1 for g in identify_language_batch(texts) if g["detected_lang"] == lang)
    nb = classify_sentiment_batch(texts, lang)
    drift = sum(1 for pred, tgt in zip(nb, df["label"].tolist()) if pred["label"] != tgt)

    row = {
        "lang": lang,
        "n": n,
        "dup_rate": round(100.0 * dup / n, 2),
        "lang_confusion": round(100.0 * (n - on_target) / n, 2),
        "label_drift": round(100.0 * drift / n, 2),
    }
    print(f"  {lang:<4} n={n:<5} dup={row['dup_rate']:>5}%  lang_confusion={row['lang_confusion']:>5}%  label_drift={row['label_drift']:>5}%")
    return row


def main() -> None:
    run_id = os.getenv("SINGLE_LLM_RUN_ID") or latest_single_llm_run_id()
    run_dir = (SINGLE_LLM_OUTPUT_DIR / run_id) if run_id else None
    if run_dir is None or not run_dir.exists():
        print(f"Tak ada run Single-LLM di {SINGLE_LLM_OUTPUT_DIR}")
        return

    print(f"Audit Single-LLM run: {run_dir.name}\n{'='*70}")
    rows = []
    for lang in TARGET_LANGS:
        if (r := audit_language(lang, run_dir)):
            rows.append(r)
        free_nusabert_cache()  # lepas NusaBERT-large bahasa ini (~1.35 GB) sebelum bahasa berikutnya

    out = run_dir / "audit_summary.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'='*70}\nDisimpan -> {out}")


if __name__ == "__main__":
    main()
