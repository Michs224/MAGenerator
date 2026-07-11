"""Bandingin JOINT multilingual vs champion per-lang (CH) vs FT — scope 7 bahasa, test-F1 + gap.
Pertanyaan: joint bantu ban/bjn (vs CH) TANPA rontokin pemenang (jav/min/sun/ace)? Putusan by per-lang + agregat.

CH/FT = 5-seed mean. JOINT = seed yg ada (screen 1-seed dulu -> caveat noisy, baru multi-seed kalau promising).

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/joint_aggregate.py
"""
import os
import glob
import json
import numpy as np

SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
FT_RUNS = ["outputs/nusabert-sentiment-large"] + [f"outputs/p0-ft-multiseed/seed_{s}" for s in [0, 1, 2, 3]]
CH_RUNS = ["outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25"] + [f"outputs/p0-champion-multiseed/seed_{s}" for s in [0, 1, 2, 3]]
JOINT_DIR = "outputs/joint-multilingual-champion"


def load(paths):
    out = []
    for p in paths:
        sp = os.path.join(p, "results_summary.json")
        if os.path.exists(sp):
            out.append(json.load(open(sp))["results"])
    return out


def f1(runs, lang):
    v = [r[lang]["test_f1"] * 100 for r in runs if lang in r]
    return np.mean(v) if v else float("nan")


def gap(runs, lang):
    v = [(r[lang]["train_f1"] - r[lang]["test_f1"]) * 100 for r in runs if lang in r]
    return np.mean(v) if v else float("nan")


def main():
    ft, ch = load(FT_RUNS), load(CH_RUNS)
    jt = load(sorted(glob.glob(f"{JOINT_DIR}/seed_*")))
    if not jt:
        print(f"Belum ada hasil joint di {JOINT_DIR}. Jalanin joint_multilingual.py dulu.")
        return
    print(f"\nJOINT n-seed={len(jt)} (CH/FT=5). Scope test-F1 + gap:\n")
    print(f"{'lang':6} | {'FT':12} | {'CH':12} | {'JOINT':12} | J-CH    J-FT")
    print("-" * 78)
    for lang in SCOPE:
        print(f"{lang:6} | {f1(ft,lang):5.2f} g{gap(ft,lang):5.2f} | {f1(ch,lang):5.2f} g{gap(ch,lang):5.2f} | "
              f"{f1(jt,lang):5.2f} g{gap(jt,lang):5.2f} | {f1(jt,lang)-f1(ch,lang):+5.2f} {f1(jt,lang)-f1(ft,lang):+5.2f}")
    print("-" * 78)
    fm = np.mean([f1(ft, l) for l in SCOPE]); cm = np.mean([f1(ch, l) for l in SCOPE]); jm = np.mean([f1(jt, l) for l in SCOPE])
    fg = np.mean([gap(ft, l) for l in SCOPE]); cg = np.mean([gap(ch, l) for l in SCOPE]); jg = np.mean([gap(jt, l) for l in SCOPE])
    print(f"{'MEAN':6} | {fm:5.2f} g{fg:5.2f} | {cm:5.2f} g{cg:5.2f} | {jm:5.2f} g{jg:5.2f} | {jm-cm:+5.2f} {jm-fm:+5.2f}")
    print(f"\nban J-CH {f1(jt,'ban')-f1(ch,'ban'):+.2f} | bjn J-CH {f1(jt,'bjn')-f1(ch,'bjn'):+.2f} "
          f"| pemenang(jav/min/sun/ace) J-CH: " + ", ".join(f"{l}{f1(jt,l)-f1(ch,l):+.2f}" for l in ["jav", "min", "sun", "ace"]))
    print("Promising kalau ban/bjn naik (J-CH +) & pemenang nggak rontok. Caveat: joint 1-seed = noisy.")


if __name__ == "__main__":
    main()
