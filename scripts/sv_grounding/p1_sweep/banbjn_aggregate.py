"""Aggregate probe ban+bjn: mean±std test_f1 + gap per config (multi-seed) vs FT/vanilla.
Flag '<<<' kalau ban & bjn dua-duanya >= FT (mean). Putus di sini buat pilih config yg di-full-run.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/banbjn_aggregate.py
"""
import os
import glob
import json
import numpy as np

SEEDS = [42, 0, 1, 2, 3]
FT_RUNS = ["outputs/nusabert-sentiment-large"] + [f"outputs/p0-ft-multiseed/seed_{s}" for s in [0, 1, 2, 3]]
CH_RUNS = ["outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25"] + [f"outputs/p0-champion-multiseed/seed_{s}" for s in [0, 1, 2, 3]]


def load(paths):
    out = []
    for p in paths:
        sp = os.path.join(p, "results_summary.json")
        if os.path.exists(sp):
            out.append(json.load(open(sp))["results"])
    return out


def load_probe(cfgdir):
    out = []
    for s in SEEDS:
        sp = os.path.join(cfgdir, f"seed_{s}", "results_summary.json")
        if os.path.exists(sp):
            out.append(json.load(open(sp))["results"])
    return out


def stat(runs, lang, key):
    vals = []
    for r in runs:
        if lang in r:
            d = r[lang]
            vals.append((d["train_f1"] - d["test_f1"]) * 100 if key == "gap" else d[key] * 100)
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")), len(vals)


def main():
    ft, ch = load(FT_RUNS), load(CH_RUNS)
    ftb = stat(ft, "ban", "test_f1")[0]
    ftj = stat(ft, "bjn", "test_f1")[0]
    print(f"\nTarget FT (reproduce): ban >= {ftb:.2f}  |  bjn >= {ftj:.2f}\n")
    print(f"{'config':24} | {'ban test (Δ)':18} {'gap':6} | {'bjn test (Δ)':18} {'gap':6} | verdict")
    print("-" * 92)

    def row(runs, name):
        if not runs:
            return
        bt = stat(runs, "ban", "test_f1"); bg = stat(runs, "ban", "gap")[0]
        jt = stat(runs, "bjn", "test_f1"); jg = stat(runs, "bjn", "gap")[0]
        flag = "<<< ban&bjn >= FT" if (bt[0] >= ftb and jt[0] >= ftj) else ""
        print(f"{name:24} | {bt[0]:5.2f}±{bt[1]:4.2f}({bt[0]-ftb:+5.2f}) {bg:6.2f} | "
              f"{jt[0]:5.2f}±{jt[1]:4.2f}({jt[0]-ftj:+5.2f}) {jg:6.2f} | {flag}")

    row(ft, "FT (baseline)")
    row(ch, "vanilla-CH")
    print("-" * 92)
    for d in sorted(glob.glob("outputs/p1-banbjn-probe*/*")):
        # nama = relpath biar niter_16 (p1-banbjn-probe/...) vs full-SVD (p1-banbjn-probe-pissa/...) kebedaan
        tag = os.path.relpath(d, "outputs").replace("p1-banbjn-probe-pissa/", "pissa·").replace("p1-banbjn-probe/", "")
        row(load_probe(d), tag)
    print("\nΔ vs FT. gap=train-test (makin kecil=overfit makin rendah). '<<<' = kandidat full-run.")


if __name__ == "__main__":
    main()
