"""Full-12 run kandidat champion = plain pissa_drop05 (champion + PiSSA init + lora_dropout 0.05).
Prioritas F1 (bukan gap<10): ban F1 80.84 (>=80, terbaik per-lang, probe 5-seed) + gap 11.07 (turun dari FT 17.79).
ban/bjn dilatih per-lang = angka udah final; full-12 ini buat cek PEMENANG (jav/min/sun/ace) + mad naik nggak + scope.

Tujuan: dapet AGREGAT 12 bahasa (5 seed) buat bandingin scope-mean vs vanilla-CH (83.83).
- Menang/seri agregat -> champion baru + bobot deploy langsung kesimpen.
- Washout di pemenang (kayak DoRA) -> balik vanilla-CH (bobot udah ada, nol retrain) -> hapus run ini.

KEEP_WEIGHTS="all": simpan bobot SEMUA seed (5 x 12) biar bisa pilih deploy by-VAL (median/ensemble)
tanpa retrain (training non-deterministik -> retrain = beda). ~81GB; hapus kalau PiSSA washout.

Subprocess-per-SEED (12 bahasa in-process/subprocess = aman dari fragmentasi). Resumable.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_drop05_full.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
# plain pissa_drop05: ban F1 80.84 (>=80, terbaik per-lang) + gap 11.07 (turun dari FT 17.79). Prioritas F1 (bukan gap<10).
OVERRIDES = {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.05}
OUT = "outputs/pissa-drop05-full"
N_LANGS = 12

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_p1_base.py")


def run():
    print(f"Full-12 plain pissa_drop05 | {OUT} | seeds {SEEDS} | KEEP=all")
    for s in SEEDS:
        sp = os.path.join(ROOT, OUT, f"seed_{s}", "results_summary.json")
        if os.path.exists(sp) and len(json.load(open(sp))["results"]) >= N_LANGS:
            print(f"--- seed {s} SKIP (12 bahasa lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps(OVERRIDES)
        env["P1_OUTPUT_ROOT"] = OUT
        env["P1_SEEDS"] = json.dumps([s])
        env["P1_KEEP_WEIGHTS"] = "all"
        print(f"\n--- seed {s} (12 bahasa, subprocess, VRAM fresh)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! seed {s} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\nFull-12 SELESAI. Agg agregat: uv run python scripts/sv_grounding/p1_sweep/p1_aggregate.py")


if __name__ == "__main__":
    run()
