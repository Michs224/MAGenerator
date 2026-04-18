"""NusaSynth outer loop: run the LangGraph pipeline per language, per label, per batch.

Usage:
  uv run nusasynth/run_pipeline.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from NusaSynth.config import BATCH_SIZE, DATA_DIR, DEDUP_THRESHOLD, LABEL_ORDER, LANG_NAMES, OUTPUT_DIR, SENTENCES_PER_SEED, TARGET_LANG
from NusaSynth.graph import build_pipeline
from NusaSynth.state import SentenceRecord
from NusaSynth.tools import jaccard_bigram

# ── Sample config
SAMPLE_RATIO: float | None = None  # 10% per label — flip to None for full run


def compute_seed_profile(train_df: pd.DataFrame, lang: str) -> dict:
    """Always computed on FULL train_df (not sampled), so stats are accurate."""
    df = train_df.copy()
    df["word_count"] = df["text"].str.split().str.len()
    stats = (
        df.groupby("label")["word_count"]
        .agg(["mean", "std"])
        .round(1)
        .to_dict("index")
    )
    return {
        "lang": lang,
        "label_distribution": train_df["label"].value_counts().to_dict(),
        "avg_length_per_label": stats,
        "unit": "words",
    }


def sample_seeds(seeds: list[dict], ratio: float | None, seed: int = 42) -> list[dict]:
    """Random sample a proportional subset from seeds. Returns all if ratio is None."""
    if ratio is None or ratio >= 1.0:
        return seeds
    import random
    rng = random.Random(seed)
    n = max(1, int(len(seeds) * ratio))
    return rng.sample(seeds, n)


def chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def save_results(
    lang: str,
    label: str,
    accepted: list[SentenceRecord],
    discarded: list[SentenceRecord],
    retried: list[SentenceRecord],
    output_dir: Path,
    batch_idx: int | None = None,
    batch_seed_ids: list[int] | None = None,
):
    """Save accepted → CSV, full state log → JSONL (accepted + discarded + retried).

    Cross-batch Jaccard dedup: new accepted sentences are checked against
    existing CSV rows. Near-duplicates are logged as 'dedup_filtered'.

    Batch failure detection: if accepted+discarded+retried all empty,
    log a 'batch_failed' entry (Contextualizer/Generator validation failed
    after retry → batch dibuang seluruhnya).
    """
    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    log_path = lang_dir / "pipeline_log.jsonl"

    # Detect entire batch failure (no output at all)
    if not accepted and not discarded and not retried:
        if batch_idx is not None and batch_seed_ids is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "outcome": "batch_failed",
                    "target_lang": lang,
                    "target_label": label,
                    "batch_idx": batch_idx,
                    "seed_ids": batch_seed_ids,
                    "expected_sentences": len(batch_seed_ids) * SENTENCES_PER_SEED,
                }, ensure_ascii=False) + "\n")
            print(f"  BATCH_FAILED: lang={lang} label={label} batch={batch_idx} seeds={batch_seed_ids}")
        return

    csv_path = lang_dir / "synthetic.csv"

    # Load existing texts for cross-batch dedup
    existing_texts: list[str] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing_df = pd.read_csv(csv_path)
        existing_texts = existing_df["text"].tolist()

    # Cross-batch Jaccard filter
    kept: list[SentenceRecord] = []
    filtered: list[SentenceRecord] = []
    for sent in accepted:
        is_dup = any(
            jaccard_bigram(sent["text"], ref) >= DEDUP_THRESHOLD
            for ref in existing_texts
        )
        if is_dup:
            filtered.append(sent)
        else:
            kept.append(sent)
            existing_texts.append(sent["text"])

    if filtered:
        print(f"  Dedup antar-batch: {len(filtered)} difilter, {len(kept)} disimpan")

    # Write kept sentences to CSV
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        if write_header:
            writer.writeheader()
        for sent in kept:
            writer.writerow({
                "id": sent["sid"],
                "text": sent["text"],
                "label": label,
            })

    # Write log (all outcomes)
    with open(log_path, "a", encoding="utf-8") as f:
        for sent in kept:
            f.write(json.dumps({**dict(sent), "outcome": "accepted"}, ensure_ascii=False) + "\n")
        for sent in filtered:
            f.write(json.dumps({**dict(sent), "outcome": "dedup_filtered"}, ensure_ascii=False) + "\n")
        for sent in discarded:
            f.write(json.dumps({**dict(sent), "outcome": "discarded"}, ensure_ascii=False) + "\n")
        for sent in retried:
            f.write(json.dumps({**dict(sent), "outcome": "retried"}, ensure_ascii=False) + "\n")


def run_language(lang: str, labels: list[str] | None = None):
    lang_name = LANG_NAMES[lang]
    print(f"\n{'='*60}")
    print(f"NusaSynth: {lang_name} ({lang})")
    if SAMPLE_RATIO is not None:
        print(f"  MODE SAMPLE: {SAMPLE_RATIO:.0%} per label")
    print(f"{'='*60}")

    train_path = DATA_DIR / lang / "train.csv"
    if not train_path.exists():
        print(f"ERROR: Data train gak ketemu: {train_path}")
        return

    train_df = pd.read_csv(train_path)
    profile = compute_seed_profile(train_df, lang)
    print(f"Seed profile (full): {profile['label_distribution']}")

    pipeline = build_pipeline()
    target_labels = labels or LABEL_ORDER
    total_accepted = 0
    total_discarded = 0

    for label in target_labels:
        all_label_seeds = train_df[train_df["label"] == label].to_dict("records")
        label_seeds = sample_seeds(all_label_seeds, SAMPLE_RATIO)
        batches = chunk(label_seeds, BATCH_SIZE)

        if SAMPLE_RATIO is not None:
            print(f"\n--- {lang_name} [{label}]: {len(label_seeds)}/{len(all_label_seeds)} seeds (sampled) -> {len(batches)} batch ---")
        else:
            print(f"\n--- {lang_name} [{label}]: {len(label_seeds)} seeds -> {len(batches)} batch ---")

        for batch_idx, batch in enumerate(batches):
            print(f"\nBatch {batch_idx + 1}/{len(batches)} [{lang} {label}]: seed_ids={[s['id'] for s in batch]}")

            result = pipeline.invoke(
                {
                    "seeds": batch,
                    "seed_profile": profile,
                    "target_label": label,
                    "target_lang": lang,
                    "lang_name": lang_name,
                },
                {"recursion_limit": 50},
            )

            accepted = result.get("all_accepted", [])
            discarded = result.get("all_discarded", [])
            retried = result.get("all_retried", [])
            print(f"  -> {len(accepted)} diterima, {len(discarded)} dibuang, {len(retried)} diretry")

            save_results(
                lang, label, accepted, discarded, retried, OUTPUT_DIR,
                batch_idx=batch_idx,
                batch_seed_ids=[s["id"] for s in batch],
            )
            total_accepted += len(accepted)
            total_discarded += len(discarded)

    print(f"\n{lang_name} selesai: {total_accepted} diterima, {total_discarded} dibuang")
    print(f"Disimpan ke: {OUTPUT_DIR / lang / 'synthetic.csv'}")


if __name__ == "__main__":
    run_language(TARGET_LANG)
