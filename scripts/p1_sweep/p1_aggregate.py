"""P1 aggregate — banding semua config sweep P1 vs vanilla-CH vs FT (scope, mean per seed).

Baca outputs/p1-*-sweep/<tag>/seed_*/results_summary.json + baseline P0 (FT & vanilla-CH).
Cetak per-config: ban, bjn (+selisih vs FT), winners-avg, scope-avg, gap-avg. Tandai '<<<' kalau
ban & bjn dua-duanya >= FT (kandidat champion baru).

Jalankan dari root:  uv run python scripts/p1_sweep/p1_aggregate.py
"""
import os
import glob
import json
import numpy as np

SCOPE = ["ace", "jav", "mad", "ban", "bjn", "min", "sun"]
WINNERS = ["ace", "jav", "mad", "min", "sun"]

FT_RUNS = {42: "outputs/nusabert-sentiment-large",
           0: "outputs/p0-ft-multiseed/seed_0", 1: "outputs/p0-ft-multiseed/seed_1",
           2: "outputs/p0-ft-multiseed/seed_2", 3: "outputs/p0-ft-multiseed/seed_3"}
CH_RUNS = {42: "outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25",
           0: "outputs/p0-champion-multiseed/seed_0", 1: "outputs/p0-champion-multiseed/seed_1",
           2: "outputs/p0-champion-multiseed/seed_2", 3: "outputs/p0-champion-multiseed/seed_3"}


def load_seeds(run_map):
    out = {}
    for s, f in run_map.items():
        p = os.path.join(f, "results_summary.json")
        if os.path.exists(p):
            out[s] = json.load(open(p)).get("results", {})
    return out


def lang_mean(runs, lang, key="test_f1"):
    vals = [runs[s][lang][key] * 100 for s in runs if lang in runs[s] and key in runs[s][lang]]
    return np.mean(vals) if vals else float("nan")


def gap_mean(runs, lang):
    vals = [(runs[s][lang]["train_f1"] - runs[s][lang]["test_f1"]) * 100
            for s in runs if lang in runs[s] and "train_f1" in runs[s][lang]]
    return np.mean(vals) if vals else float("nan")


def main():
    ft = load_seeds(FT_RUNS)
    ch = load_seeds(CH_RUNS)
    ft_ban, ft_bjn = lang_mean(ft, "ban"), lang_mean(ft, "bjn")

    print(f"FT baseline: ban={ft_ban:.2f} bjn={ft_bjn:.2f} (target: config >= ini)\n")
    print(f"{'config':36} {'ban':>14} {'bjn':>14} {'win':>6} {'scope':>6} {'gap':>5}")
    print("-" * 90)

    def row(runs, name):
        if not runs:
            return
        ban, bjn = lang_mean(runs, "ban"), lang_mean(runs, "bjn")
        win = np.nanmean([lang_mean(runs, l) for l in WINNERS])
        sc = np.nanmean([lang_mean(runs, l) for l in SCOPE])
        gp = np.nanmean([gap_mean(runs, l) for l in SCOPE])
        flag = " <<< ban&bjn>=FT" if (ban >= ft_ban and bjn >= ft_bjn) else ""
        print(f"{name:36} {ban:6.2f}({ban-ft_ban:+5.2f}) {bjn:6.2f}({bjn-ft_bjn:+5.2f}) {win:6.2f} {sc:6.2f} {gp:5.2f}{flag}")

    row(ft, "FT (baseline)")
    row(ch, "vanilla-CH (baseline)")
    print("-" * 90)
    for d in sorted(glob.glob("outputs/p1-*-sweep/*")):
        runs = load_seeds({s: f"{d}/seed_{s}" for s in [42, 0, 1, 2, 3]})
        row(runs, os.path.relpath(d, "outputs"))
    print("\n'<<<' = kandidat (ban & bjn dua-duanya >= FT). Putus di VAL dulu; ini ringkas test buat lihat pola.")


if __name__ == "__main__":
    main()
