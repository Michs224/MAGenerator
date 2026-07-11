"""Buat varian train_syn dengan rasio sintetis berbeda (C1 ratio-tuning).

Seed (data asli) SELALU utuh; hanya porsi SINTETIS yang di-subsample.
Subsample STRATIFIED per label (proporsi label sintetis dipertahankan),
dibulatkan per label (round). random_state tetap → reproducible.

Output: data/nusax_senti/<lang>/syn/train_syn_<tag>.csv
Pakai di finetune dengan mengganti path train ke train_syn_<tag>.csv +
OUTPUT_BASE_DIR yang sesuai (mis. ...-syn-r40).

Jalankan dari root project:  uv run python scripts/make_ratio_variants.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from NusaSynth.config import synthetic_dir  # folder-per-run resolver (latest run / NUSASYNTH_RUN_ID)

LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
RATIOS = {"40": 0.40, "50": 0.50, "60": 0.60}   # fraksi SINTETIS yang dipertahankan
SAMPLE_SEED = 42                     # reproducible subsample
DATA = Path("data/nusax_senti")


def make_variant(seed_df: pd.DataFrame, syn_df: pd.DataFrame, frac: float) -> pd.DataFrame:
    # stratified per label, dibulatkan
    parts = []
    for label, g in syn_df.groupby("label"):
        n = int(round(len(g) * frac))          # <-- pembulatan per label
        n = min(n, len(g))
        parts.append(g.sample(n=n, random_state=SAMPLE_SEED))
    syn_sub = pd.concat(parts).reset_index(drop=True)
    # reassign id sintetis melanjutkan dari seed (konvensi train_syn)
    last = int(seed_df["id"].max())
    syn_sub = syn_sub.copy()
    syn_sub["id"] = range(last + 1, last + 1 + len(syn_sub))
    return pd.concat([seed_df, syn_sub], ignore_index=True)


def main():
    for lang in LANGS:
        seed_path = DATA / lang / "train.csv"
        syn_path = synthetic_dir(lang) / "synthetic.csv"
        if not (seed_path.exists() and syn_path.exists()):
            print(f"[{lang}] SKIP (seed={seed_path.exists()} syn={syn_path.exists()})")
            continue
        seed_df = pd.read_csv(seed_path)
        syn_df = pd.read_csv(syn_path)[["id", "text", "label"]]
        outdir = DATA / lang / "syn"
        outdir.mkdir(parents=True, exist_ok=True)
        for tag, frac in RATIOS.items():
            combined = make_variant(seed_df, syn_df, frac)
            out = outdir / f"train_syn_{tag}.csv"
            combined.to_csv(out, index=False, encoding="utf-8")
            n_syn = len(combined) - len(seed_df)
            lc = combined["label"].value_counts().to_dict()
            print(f"[{lang}] r{tag}%: seed {len(seed_df)} + syn {n_syn} = {len(combined)} | {lc}")


if __name__ == "__main__":
    main()
