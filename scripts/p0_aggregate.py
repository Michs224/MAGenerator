"""P0 — Agregasi multi-seed: mean+-std per bahasa + vonis champion-vs-FT (adil, mean-vs-mean).

Baca results_summary.json semua seed (seed 42 di-REUSE dari folder kanonik), hitung
mean+-std test/val/gap per bahasa untuk FT (paper) dan champion (LP-FT+LoRA), lalu cetak
tabel vonis: apakah ban/bjn (dan lainnya) regresi NYATA atau cuma DALAM NOISE.

Keputusan = mean lintas seed (anti-leakage; ini di test untuk laporan akhir, val untuk
seleksi sebenarnya — di sini dua-duanya ditampilkan). Single-seed FT bjn=86.76 itu draw
upper-tail; pembanding adil = FT mean.

Jalankan setelah p0_ft_multiseed.py + p0_champion_multiseed.py selesai:
  uv run python scripts/p0_aggregate.py
"""
import os
import json
import math

import numpy as np

SCOPE = ["ace", "jav", "mad", "ban", "bjn", "min", "sun"]   # 7 bahasa proyek
ALL_LANGS = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]
SEEDS = [42, 0, 1, 2, 3]
DEPLOY_SEED = 42   # Cara A: seed deploy ditetapkan di muka (pre-commit)

# Peta seed -> folder yang punya results_summary.json (42 = run kanonik yang di-reuse)
FT_RUNS = {
    42: "outputs/nusabert-sentiment-large",
    0:  "outputs/p0-ft-multiseed/seed_0",
    1:  "outputs/p0-ft-multiseed/seed_1",
    2:  "outputs/p0-ft-multiseed/seed_2",
    3:  "outputs/p0-ft-multiseed/seed_3",
}
CH_RUNS = {
    42: "outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25",
    0:  "outputs/p0-champion-multiseed/seed_0",
    1:  "outputs/p0-champion-multiseed/seed_1",
    2:  "outputs/p0-champion-multiseed/seed_2",
    3:  "outputs/p0-champion-multiseed/seed_3",
}
OUT_DIR = "outputs/p0-aggregate"


def load_runs(run_map):
    """seed -> results dict (per lang). Lewati seed yang summary-nya belum ada."""
    out = {}
    for seed, folder in run_map.items():
        path = os.path.join(folder, "results_summary.json")
        if os.path.exists(path):
            out[seed] = json.load(open(path)).get("results", {})
        else:
            print(f"  [skip] seed {seed}: belum ada {path}")
    return out


def collect(runs, lang, key):
    """List nilai metrik (×100) lintas seed untuk satu bahasa. gap dihitung train-test."""
    vals = []
    for seed, res in runs.items():
        if lang not in res:
            continue
        r = res[lang]
        if key == "gap":
            if "train_f1" in r and "test_f1" in r:
                vals.append((r["train_f1"] - r["test_f1"]) * 100)
        elif key in r:
            vals.append(r[key] * 100)
    return vals


def msn(vals):
    """mean, std(sample), n."""
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], float("nan"), 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)


def fmt(mean, std, n):
    if n == 0:
        return "   n/a  "
    if n == 1:
        return f"{mean:5.2f}(1) "
    return f"{mean:5.2f}±{std:4.2f}"


def verdict(ch_test, ft_test):
    cm, cs, cn = ch_test
    fm, fs, fn = ft_test
    if cn == 0 or fn == 0:
        return "?", float("nan")
    delta = cm - fm
    # standard error of difference of means (pp); fallback std=1.3pp kalau n<2
    cs_ = cs if (cn >= 2 and not math.isnan(cs)) else 1.3
    fs_ = fs if (fn >= 2 and not math.isnan(fs)) else 1.3
    se = math.sqrt(cs_**2 / max(cn, 1) + fs_**2 / max(fn, 1))
    if delta >= 0:
        label = "OK (>=FT)"
    elif abs(delta) <= max(1.0, 2 * se):
        label = "within-noise"
    else:
        label = "REGRESI nyata"
    return label, delta


def main():
    ft_runs = load_runs(FT_RUNS)
    ch_runs = load_runs(CH_RUNS)
    print(f"\nFT seeds tersedia: {sorted(ft_runs)} | Champion seeds tersedia: {sorted(ch_runs)}\n")

    rows = {}
    print("\nTabel P0 (mean±std test/gap, champion vs FT):")
    print(f"{'lang':5}| {'FT test':12} {'CH test':12} {'dTest':7} {'verdict':14}| {'FT gap':12} {'CH gap':12}| {'CH val':12}")
    print("-" * 90)
    for lang in ALL_LANGS:
        ch_test = msn(collect(ch_runs, lang, "test_f1"))
        ft_test = msn(collect(ft_runs, lang, "test_f1"))
        ch_gap = msn(collect(ch_runs, lang, "gap"))
        ft_gap = msn(collect(ft_runs, lang, "gap"))
        ch_val = msn(collect(ch_runs, lang, "val_f1"))
        label, delta = verdict(ch_test, ft_test)
        mark = " *" if lang in ("ban", "bjn") else ("  " if lang in SCOPE else " .")
        print(f"{lang:5}|{mark} {fmt(*ft_test):12} {fmt(*ch_test):12} {delta:+6.2f} {label:14}| "
              f"{fmt(*ft_gap):12} {fmt(*ch_gap):12}| {fmt(*ch_val):12}")
        rows[lang] = {
            "ft_test_mean": ft_test[0], "ft_test_std": ft_test[1], "ft_test_n": ft_test[2],
            "ch_test_mean": ch_test[0], "ch_test_std": ch_test[1], "ch_test_n": ch_test[2],
            "delta_test": delta, "verdict": label,
            "ft_gap_mean": ft_gap[0], "ch_gap_mean": ch_gap[0],
            "ch_val_mean": ch_val[0], "ch_val_std": ch_val[1],
            "in_scope": lang in SCOPE,
        }
    print("* = ban/bjn (target) | (blank)=scope | .=luar scope | dTest = CH_mean - FT_mean (test, pp)")
    print("verdict: OK(>=FT) | within-noise (|dTest|<=2*SE) | REGRESI nyata\n")

    # Ringkas scope
    print("RINGKAS (7 scope):")
    for lang in SCOPE:
        r = rows[lang]
        print(f"  {lang}: champion test {r['ch_test_mean']:.2f} vs FT {r['ft_test_mean']:.2f} "
              f"(d{r['delta_test']:+.2f}) -> {r['verdict']} | gap {r['ch_gap_mean']:.2f}")

    # ---- Pemilihan seed deploy (champion): Cara A (pre-commit 42) vs Cara B (median) ----
    # Skor per-seed = val rata-rata across SCOPE (7). Median = draw representatif.
    per_seed_val = {}
    for seed, res in ch_runs.items():
        vals = [res[l]["val_f1"] * 100 for l in SCOPE if l in res and "val_f1" in res[l]]
        if vals:
            per_seed_val[seed] = float(np.mean(vals))

    deploy = {"cara_A_precommit_seed": DEPLOY_SEED, "cara_B_median_seed": None, "per_seed_val_scope": per_seed_val}
    print("\nPILIHAN SEED DEPLOY (champion) — val rata-rata 7 scope per seed:")
    if per_seed_val:
        for seed in sorted(per_seed_val, key=lambda s: per_seed_val[s]):
            tag = "  <- Cara A (pre-commit)" if seed == DEPLOY_SEED else ""
            print(f"  seed {seed:>3}: {per_seed_val[seed]:.2f}{tag}")
        med_val = float(np.median(list(per_seed_val.values())))
        median_seed = min(per_seed_val, key=lambda s: abs(per_seed_val[s] - med_val))
        deploy["cara_B_median_seed"] = median_seed
        deploy["median_val_scope"] = med_val
        a_val = per_seed_val.get(DEPLOY_SEED)
        print(f"  -> Cara A: deploy seed {DEPLOY_SEED}" + (f" (val {a_val:.2f})" if a_val is not None else " (BELUM ada — jalankan seed 42)"))
        print(f"  -> Cara B: deploy seed {median_seed} (val {per_seed_val[median_seed]:.2f}, median={med_val:.2f})")
        print("  (bobot SEMUA seed disimpan -> tinggal pilih A atau B; deploy = model seed terpilih untuk ke-7 bahasa)")
    else:
        print("  (belum ada run champion)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "p0_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "seeds_requested": SEEDS,
            "ft_seeds_found": sorted(ft_runs), "champion_seeds_found": sorted(ch_runs),
            "scope": SCOPE, "rows": rows, "deploy_seed_selection": deploy,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
