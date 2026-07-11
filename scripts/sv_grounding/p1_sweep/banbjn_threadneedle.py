"""Thread-the-needle: cari config per-lang yang ban F1 >= 80 DAN gap < 10 (sudut Pareto sempit).

Data sekarang: F1>80 ATAU gap<10, belum dua-duanya (per-lang). Coba titik tengah:
  - LR 4.5e-5 (antara 4e-5 gap-kecil & 5e-5 F1-tinggi)
  - + r24 (kapasitas ekstra buat dorong F1 tanpa naikin LR)
Semua PiSSA(plain) + dropout 0.05. 5-seed ban+bjn (murah, reliable). Bandingin: banbjn_aggregate.py.

CATATAN JUJUR: ini meluncur di frontier Pareto -> mungkin dapet, mungkin meleset tipis di satu sumbu.
JOINT udah nge-hit ban F1 80.93/gap 4.66 (frontier digeser pakai data). Ini cuma cek apakah per-lang bisa juga.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/banbjn_threadneedle.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
LANGS = ["ban", "bjn"]
N_LANGS = 2
FOLDER = "outputs/p1-banbjn-probe-threadneedle"   # match glob banbjn_aggregate (p1-banbjn-probe*)
CONFIGS = [
    {"tag": "lr45_drop05",     "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.05, "FT_LEARNING_RATE": 4.5e-5}},
    {"tag": "lr4_drop05_r24",  "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.05, "FT_LEARNING_RATE": 4e-5, "LORA_R": 24, "LORA_ALPHA": 48}},
    {"tag": "lr45_drop05_r24", "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.05, "FT_LEARNING_RATE": 4.5e-5, "LORA_R": 24, "LORA_ALPHA": 48}},
]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_p1_base.py")


def _config_done(out):
    for s in SEEDS:
        sp = os.path.join(ROOT, out, f"seed_{s}", "results_summary.json")
        if not os.path.exists(sp) or len(json.load(open(sp))["results"]) < N_LANGS:
            return False
    return True


def run():
    for cfg in CONFIGS:
        out = f"{FOLDER}/{cfg['tag']}"
        if _config_done(out):
            print(f"--- threadneedle:{cfg['tag']} SKIP (lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps(cfg["ov"])
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SEEDS)
        env["P1_TARGET_LANGS"] = json.dumps(LANGS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- threadneedle:{cfg['tag']} (ban+bjn x {len(SEEDS)} seed)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! threadneedle:{cfg['tag']} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\nthreadneedle SELESAI. Lihat: uv run python scripts/sv_grounding/p1_sweep/banbjn_aggregate.py")


if __name__ == "__main__":
    run()
