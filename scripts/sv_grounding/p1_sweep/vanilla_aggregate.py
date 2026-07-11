"""Aggregate probe vanilla (ace/ban/bjn/mad, 5-seed) vs FT / vanilla-CH / PiSSA-drop05-full.
Cek: config vanilla-baru bisa recover ban/bjn/mad TANPA korbanin ace (yg jebol di PiSSA)?

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/vanilla_aggregate.py
"""
import os
import glob
import json
import numpy as np

SEEDS = [42, 0, 1, 2, 3]
LANGS = ["ace", "ban", "bjn", "mad"]
FT_RUNS = ["outputs/nusabert-sentiment-large"] + [f"outputs/p0-ft-multiseed/seed_{s}" for s in [0, 1, 2, 3]]
CH_RUNS = ["outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25"] + [f"outputs/p0-champion-multiseed/seed_{s}" for s in [0, 1, 2, 3]]
PISSA_RUNS = [f"outputs/pissa-drop06-full/seed_{s}" for s in SEEDS]   # drop06 = incumbent (lolos semua 7)


def load(paths):
    out = []
    for p in paths:
        sp = os.path.join(p, "results_summary.json")
        if os.path.exists(sp):
            out.append(json.load(open(sp))["results"])
    return out


def load_probe(cfgdir):
    return load([os.path.join(cfgdir, f"seed_{s}") for s in SEEDS])


def stat(runs, lang, key):
    vals = []
    for r in runs:
        if lang in r:
            d = r[lang]
            vals.append((d["train_f1"] - d["test_f1"]) * 100 if key == "gap" else d[key] * 100)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"))


def main():
    ft, ch, pi = load(FT_RUNS), load(CH_RUNS), load(PISSA_RUNS)
    print("\nTarget: F1>=80 DAN gap<10, utamanya ace(jaga)/ban/bjn/mad(recover)\n")
    header = f"{'config':26} | " + " | ".join(f"{l:>17}" for l in LANGS)
    print(header)
    print("-" * len(header))

    def row(runs, name):
        if not runs:
            return
        cells = []
        for l in LANGS:
            f1, fs = stat(runs, l, "test_f1")
            g = stat(runs, l, "gap")[0]
            flag = "*" if (f1 >= 80 and g < 10) else " "
            cells.append(f"{f1:5.2f}±{fs:4.2f}g{g:4.1f}{flag}")
        print(f"{name:26} | " + " | ".join(cells))

    row(ft, "FT (baseline)")
    row(ch, "vanilla-CH")
    row(pi, "PiSSA-drop06-full <=inc")
    print("-" * len(header))
    for d in sorted(glob.glob("outputs/p1-pissa-reg/*")):
        row(load_probe(d), "reg:" + os.path.basename(d))
    for d in sorted(glob.glob("outputs/p1-pissa-capacity/*")):
        row(load_probe(d), "capacity:" + os.path.basename(d))
    print("-" * len(header))
    for d in sorted(glob.glob("outputs/p1-pissa-ace-rescue/*")):
        row(load_probe(d), "rescue:" + os.path.basename(d))
    for d in sorted(glob.glob("outputs/p1-vanilla-probe/*")):
        row(load_probe(d), "vanilla:" + os.path.basename(d))
    print("\n'*' = F1>=80 DAN gap<10. INCUMBENT drop06: ace80.56/g7.7 ban81.10/g9.9 bjn84.85/g8.1 mad81.82/g7.8.")
    print("Target capacity-probe: ace/bjn F1 NAIK ke arah FT (ace>81, bjn>85.5) TANPA ada yg <80 / gap>10.")


if __name__ == "__main__":
    main()
