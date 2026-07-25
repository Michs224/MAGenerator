"""Uji statistik: apakah single vs full test_f1 beda signifikan? Paired per (lang,seed)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path("y:/Michh/Python/Projects/MAGenerator")
LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
SEEDS = [42, 0, 123]
MODELS = [("nusabert-sentiment-base", "NusaBERT-base"),
          ("nusabert-sentiment-large", "NusaBERT-large"),
          ("xlmr-sentiment-base", "XLM-R-base")]

print("Paired test single-vs-full test_f1 (pasangan = lang×seed, n=21 per model)")
print("H0: tidak ada beda. diff = single - full (poin F1)\n")
print(f"{'model':16} {'n':>3} {'mean diff':>9} {'95% CI':>16} {'t p':>8} {'Wilcoxon p':>11} {'signifikan?':>11}")
for folder, name in MODELS:
    full, single = [], []
    for seed in SEEDS:
        for src, box in [("full", full), ("single_llm", single)]:
            p = ROOT / f"outputs/downstream_{src}/{folder}-syn-seed{seed}/results_summary.json"
            d = json.loads(p.read_text())["results"]
            box.append([d[l]["test_f1"] * 100 for l in LANGS])
    full = np.array(full).ravel(); single = np.array(single).ravel()
    diff = single - full
    n = len(diff)
    t, tp = stats.ttest_rel(single, full)
    try:
        w, wp = stats.wilcoxon(diff)
    except Exception:
        wp = float("nan")
    ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff))
    sig = "YA" if tp < 0.05 else "tidak"
    print(f"{name:16} {n:>3} {diff.mean():+9.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  {tp:8.3f} {wp:11.3f} {sig:>11}")

print("\nCatatan: CI mencakup 0 & p>0.05 ⇒ beda single-vs-full tak signifikan ⇒ SERI.")
