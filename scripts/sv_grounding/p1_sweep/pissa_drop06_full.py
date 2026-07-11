"""Full-12 run kandidat champion = PiSSA-init + dropout 0.06 (sweet-spot micro-search).

drop06 = SATU-SATUNYA config yg lolos rule (F1>=80 & gap<10) di SEMUA 4 bahasa kontes (5-seed probe:
ace 81.90/9.8, ban 80.50/9.8, bjn 84.69/7.8, mad 80.18/7.8). dropout 0.06 = titik feasible sempit antara
drop05 (ace jebol) & drop08 (mad jebol). Full-12 ini buat: (1) konfirmasi PEMENANG (jav/min/sun) tetap
>=80 & gap<10, (2) dapet bobot deploy 5-seed. Margin tipis (ace/ban gap 9.8 mepet 10) -> full-run bisa geser.

KEEP_WEIGHTS="all": simpan 5 seed x 12 (~80GB) buat pilih deploy by-VAL tanpa retrain. Hapus kalau washout.
Subprocess-per-seed (VRAM fresh, hindari crash akumulasi 0xC0000005). Resumable.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_drop06_full.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
OVERRIDES = {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.06}
OUT = "outputs/pissa-drop06-full"
N_LANGS = 12

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_p1_base.py")


def run():
    print(f"Full-12 PiSSA drop06 | {OUT} | seeds {SEEDS} | KEEP=all")
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
    print("\nFull-12 drop06 SELESAI. Agg: uv run python scripts/sv_grounding/p1_sweep/pissa_full_aggregate.py")


if __name__ == "__main__":
    run()
