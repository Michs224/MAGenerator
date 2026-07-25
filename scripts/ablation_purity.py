"""Purity head-to-head: FULL vs SINGLE-LLM synthetic.
lang_confusion (GlotLID = nilai LV) + label_drift (NusaBERT champion = nilai SV).
free_nusabert_cache antar-bahasa (hindari paging). Semua via code.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("y:/Michh/Python/Projects/MAGenerator")
sys.path.insert(0, str(ROOT))
from NusaSynth.run_pipeline import free_nusabert_cache
from NusaSynth.tools import classify_sentiment_batch, identify_language_batch

LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
SETS = {
    "full":   ROOT / "outputs/synthetic/2026-07-17_02-40-52",
    "single": ROOT / "outputs/synthetic_single_llm/2026-07-23_22-12-08",
}


def audit(texts, labels, lang):
    on_target = sum(1 for g in identify_language_batch(texts) if g["detected_lang"] == lang)
    nb = classify_sentiment_batch(texts, lang)
    drift = sum(1 for pred, tgt in zip(nb, labels) if pred["label"] != tgt)
    n = len(texts)
    return 100.0 * (n - on_target) / n, 100.0 * drift / n


res = {s: {} for s in SETS}
for lang in LANGS:
    for s, base in SETS.items():
        df = pd.read_csv(base / lang / "synthetic.csv")
        texts = df["text"].astype(str).tolist()
        lc, ld = audit(texts, df["label"].tolist(), lang)
        res[s][lang] = (lc, ld)
        print(f"  [{s:6}] {lang}: lang_confusion={lc:5.2f}%  label_drift={ld:5.2f}%")
        free_nusabert_cache()

print("\n" + "=" * 60)
print(f"{'lang':5} | {'full lang_conf':>14} {'sngl lang_conf':>14} | {'full drift':>11} {'sngl drift':>11}")
for lang in LANGS:
    f, s = res["full"][lang], res["single"][lang]
    print(f"{lang:5} | {f[0]:14.2f} {s[0]:14.2f} | {f[1]:11.2f} {s[1]:11.2f}")
print("-" * 60)
fc = np.mean([res["full"][l][0] for l in LANGS]); sc = np.mean([res["single"][l][0] for l in LANGS])
fd = np.mean([res["full"][l][1] for l in LANGS]); sd = np.mean([res["single"][l][1] for l in LANGS])
print(f"{'MEAN':5} | {fc:14.2f} {sc:14.2f} | {fd:11.2f} {sd:11.2f}")
print(f"\nlang_confusion (GlotLID, nilai LV): full {fc:.2f}% vs single {sc:.2f}%  (Δ {sc-fc:+.2f})")
print(f"label_drift    (NusaBERT, nilai SV): full {fd:.2f}% vs single {sd:.2f}%  (Δ {sd-fd:+.2f})")
