"""Bangun data latih augmented (seed + sintetis) dari run pipeline TERBARU.

Dua keluaran:
  1. `full`  -> data/nusax_senti/<lang>/syn/train_syn.csv        <- DIPAKAI skrip finetune
  2. rasio   -> data/nusax_senti/<lang>/syn/train_syn_<tag>.csv  <- varian C1 ratio-tuning

Seed (data asli) SELALU utuh; hanya porsi SINTETIS yang di-subsample (varian rasio).
Subsample STRATIFIED per label (proporsi label sintetis dipertahankan), dibulatkan
per label (round). random_state tetap -> reproducible. `full` tidak di-sample sama sekali.

Sumber sintetis = `synthetic_dir(lang)` (folder-per-run: run TERBARU, atau pin lewat env
NUSASYNTH_RUN_ID). JALANKAN ULANG setiap ada run pipeline baru, kalau tidak skrip finetune
akan melatih data run LAMA tanpa error (kegagalan senyap).

Id sintetis di-reassign melanjutkan id seed (last+1, ...) -> tidak bentrok dengan id asli.

Jalankan dari root project:  uv run python scripts/make_ratio_variants.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from NusaSynth.config import single_llm_synthetic_dir, synthetic_dir  # folder-per-run resolver

LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
# "full" = SELURUH sintetis -> train_syn.csv (skenario Augmentasi utama, dipakai finetune).
# Sisanya = varian rasio C1 (ratio-tuning), belum dipakai di run-1.
RATIOS = {"full": 1.00, "40": 0.40, "50": 0.50, "60": 0.60}   # fraksi SINTETIS yang dipertahankan
SAMPLE_SEED = 42                     # reproducible subsample
DATA = Path("data/nusax_senti")

# Sumber sintetis: "full" = pipeline NusaSynth penuh (outputs/synthetic/); "single_llm" = baseline (outputs/synthetic_single_llm/).
SYNTH_SOURCE = "single_llm"  # "full" | "single_llm"
_resolver = synthetic_dir if SYNTH_SOURCE == "full" else single_llm_synthetic_dir
_out_sub = "syn" if SYNTH_SOURCE == "full" else "syn_single_llm"   # data/nusax_senti/<lang>/<_out_sub>/
_main_file = "train_syn.csv" if SYNTH_SOURCE == "full" else "train_syn_single_llm.csv"  # nama file utama per-source


def make_variant(seed_df: pd.DataFrame, syn_df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if frac >= 1.0:
        syn_sub = syn_df.reset_index(drop=True)   # full: tanpa sampling, urutan asli dipertahankan
    else:
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
        syn_path = _resolver(lang) / "synthetic.csv"
        if not (seed_path.exists() and syn_path.exists()):
            print(f"[{lang}] SKIP (seed={seed_path.exists()} syn={syn_path.exists()})")
            continue
        seed_df = pd.read_csv(seed_path)
        syn_df = pd.read_csv(syn_path)[["id", "text", "label"]]
        outdir = DATA / lang / _out_sub
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[{lang}] sumber sintetis: {syn_path}")
        for tag, frac in RATIOS.items():
            combined = make_variant(seed_df, syn_df, frac)
            # 'full' -> train_syn.csv (nama kanonik yang dibaca skrip finetune)
            out = outdir / (_main_file if tag == "full" else f"train_syn_{tag}.csv")
            combined.to_csv(out, index=False, encoding="utf-8")
            n_syn = len(combined) - len(seed_df)
            assert combined["id"].is_unique, f"{lang}/{tag}: id tidak unik!"
            lc = combined["label"].value_counts().to_dict()
            print(f"[{lang}] {tag:>4}: seed {len(seed_df)} + syn {n_syn} = {len(combined)} -> {out.name} | {lc}")


if __name__ == "__main__":
    main()
