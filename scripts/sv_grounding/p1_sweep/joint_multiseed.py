"""Joint multilingual MULTI-SEED (subprocess per seed = VRAM/RAM fresh, hindari MemoryError akumulasi).

Konfirmasi hasil joint (seed 42 udah jalan, 1-seed) dgn 5 seed -> mean+-std (best-practice, anti-hoki).
Joint = 1 model/seed -> KEEP=all MURAH (~6.75GB total, bukan 81GB kayak per-lang) -> simpan semua seed,
pilih deploy by-val/ensemble tanpa retrain. Resumable (skip seed yg udah lengkap).

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/joint_multiseed.py
Lalu:  PYTHONIOENCODING=utf-8 uv run python scripts/sv_grounding/p1_sweep/joint_aggregate.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
OUT = "outputs/joint-multilingual-champion"
N_LANGS = 12

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_multilingual.py")


def done(seed):
    sp = os.path.join(ROOT, OUT, f"seed_{seed}", "results_summary.json")
    return os.path.exists(sp) and len(json.load(open(sp)).get("results", {})) >= N_LANGS


def run():
    for s in SEEDS:
        if done(s):
            print(f"--- joint seed {s} SKIP (12 bahasa lengkap)")
            continue
        env = dict(os.environ)
        env["P1_SEEDS"] = json.dumps([s])
        env["P1_OUTPUT_ROOT"] = OUT
        env["P1_KEEP_WEIGHTS"] = "all"   # joint = 1 model/seed, murah; simpan semua buat pilih by-val
        print(f"\n--- joint seed {s} (subprocess, fresh)")
        rc = subprocess.run([sys.executable, SCRIPT], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! joint seed {s} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\nJoint multi-seed SELESAI. Agg: uv run python scripts/sv_grounding/p1_sweep/joint_aggregate.py")


if __name__ == "__main__":
    run()
