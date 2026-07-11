"""Cek PER-SEED: seed mana yg lolos SEMUA 7 scope (F1>=80 & gap<10) + seed deploy by-median-val.

Kenapa: rule "semua >=80 & gap<10" itu properti MEAN; deploy = 1 seed. Perlu tau berapa seed yg lolos
single-seed + seed by-median-val (deploy) lolos bersih nggak. by-median-val = defensible (bukan by-test).

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/perseed_check.py <folder>
  contoh:  uv run python scripts/sv_grounding/p1_sweep/perseed_check.py pissa-drop07wd10-full
           uv run python scripts/sv_grounding/p1_sweep/perseed_check.py pissa-drop06-full
"""
import os
import sys
import json
import numpy as np

SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
SEEDS = [42, 0, 1, 2, 3]


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "pissa-drop06-full"
    d = os.path.join("outputs", folder)
    rows = []
    for s in SEEDS:
        sp = os.path.join(d, f"seed_{s}", "results_summary.json")
        if not os.path.exists(sp):
            continue
        r = json.load(open(sp))["results"]
        if not all(l in r for l in SCOPE):
            continue
        f1 = {l: r[l]["test_f1"] * 100 for l in SCOPE}
        gap = {l: (r[l]["train_f1"] - r[l]["test_f1"]) * 100 for l in SCOPE}
        val = np.mean([r[l]["val_f1"] * 100 for l in SCOPE if "val_f1" in r[l]])
        n80 = sum(1 for l in SCOPE if f1[l] >= 80)
        ng = sum(1 for l in SCOPE if gap[l] < 10)
        ok = (n80 == 7 and ng == 7)
        fail = [f"{l}(F1{f1[l]:.1f})" for l in SCOPE if f1[l] < 80] + [f"{l}(g{gap[l]:.1f})" for l in SCOPE if gap[l] >= 10]
        rows.append((s, n80, ng, ok, val, fail))

    if not rows:
        print(f"Belum ada hasil lengkap di outputs/{folder}")
        return
    print(f"\n=== outputs/{folder} — per-seed (7 scope) ===")
    print(f"{'seed':5} {'#>=80':6} {'#g<10':6} {'LOLOS7':7} {'val':7} gagal-di")
    for s, n80, ng, ok, val, fail in rows:
        print(f"{s:5} {n80}/7   {ng}/7   {'YA' if ok else 'nggak':7} {val:6.2f}  {', '.join(fail) if fail else '-'}")

    npass = sum(1 for r in rows if r[3])
    # seed by-median-val
    sv = sorted(rows, key=lambda x: x[4])
    med = sv[len(sv) // 2]
    print(f"\nLolos semua-7: {npass}/{len(rows)} seed.")
    print(f"Deploy by-median-val = seed {med[0]} (val {med[4]:.2f}) -> {'LOLOS bersih' if med[3] else 'GAGAL: '+', '.join(med[5])}")
    print("(drop06 dulu: 1/5 lolos, median-val=seed42 GAGAL ban gap 10.77. Cari yg >1/5 + median-val lolos.)")


if __name__ == "__main__":
    main()
