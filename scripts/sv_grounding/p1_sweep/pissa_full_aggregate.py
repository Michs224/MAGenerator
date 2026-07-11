"""Bandingin AGREGAT scope (7 bahasa): pissa-drop05-full vs vanilla-CH vs FT (5-seed, mean+-std).
Per bahasa: test-F1 (mean+-std) + gap. Plus scope-MEAN. Keputusan: PiSSA scope-mean >= CH -> champion baru.

Jalankan dari root (kelar full run):  uv run python scripts/sv_grounding/p1_sweep/pissa_full_aggregate.py
"""
import os
import json
import numpy as np

SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
FT_RUNS = ["outputs/nusabert-sentiment-large"] + [f"outputs/p0-ft-multiseed/seed_{s}" for s in [0, 1, 2, 3]]
CH_RUNS = ["outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25"] + [f"outputs/p0-champion-multiseed/seed_{s}" for s in [0, 1, 2, 3]]
PI_RUNS = [f"outputs/pissa-drop06-full/seed_{s}" for s in [42, 0, 1, 2, 3]]


def load(paths):
    out = []
    for p in paths:
        sp = os.path.join(p, "results_summary.json")
        if os.path.exists(sp):
            out.append(json.load(open(sp))["results"])
    return out


def f1(runs, lang):
    v = [r[lang]["test_f1"] * 100 for r in runs if lang in r]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else float("nan")) if v else (float("nan"), float("nan"))


def gap(runs, lang):
    v = [(r[lang]["train_f1"] - r[lang]["test_f1"]) * 100 for r in runs if lang in r]
    return np.mean(v) if v else float("nan")


def main():
    ft, ch, pi = load(FT_RUNS), load(CH_RUNS), load(PI_RUNS)
    if not pi:
        print("Belum ada hasil pissa-drop05-full. Jalanin pissa_drop05_full.py dulu.")
        return
    print(f"\n{'lang':6} | {'FT f1':12} {'gap':6} | {'CH f1':12} {'gap':6} | {'PiSSA f1':12} {'gap':6} | Pi-CH  Pi-FT")
    print("-" * 92)
    for lang in SCOPE:
        ff, fs = f1(ft, lang); cf, cs = f1(ch, lang); pf, ps = f1(pi, lang)
        print(f"{lang:6} | {ff:5.2f}±{fs:4.2f}  {gap(ft,lang):5.2f} | {cf:5.2f}±{cs:4.2f}  {gap(ch,lang):5.2f} | "
              f"{pf:5.2f}±{ps:4.2f}  {gap(pi,lang):5.2f} | {pf-cf:+5.2f} {pf-ff:+5.2f}")
    print("-" * 92)
    fm = np.mean([f1(ft, l)[0] for l in SCOPE]); cm = np.mean([f1(ch, l)[0] for l in SCOPE]); pm = np.mean([f1(pi, l)[0] for l in SCOPE])
    fg = np.mean([gap(ft, l) for l in SCOPE]); cg = np.mean([gap(ch, l) for l in SCOPE]); pg = np.mean([gap(pi, l) for l in SCOPE])
    print(f"{'MEAN':6} | {fm:5.2f}       {fg:5.2f} | {cm:5.2f}       {cg:5.2f} | {pm:5.2f}       {pg:5.2f} | {pm-cm:+5.2f} {pm-fm:+5.2f}")
    print(f"\nKEPUTUSAN: PiSSA scope-mean {pm:.2f} vs vanilla-CH {cm:.2f} -> "
          + ("PiSSA >= CH: champion baru (cek pemenang gak rontok + gap)." if pm >= cm else "PiSSA < CH (washout): balik vanilla-CH, hapus run ini."))


if __name__ == "__main__":
    main()
