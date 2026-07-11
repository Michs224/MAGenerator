"""Full-run kandidat = drop07_wd10 (PiSSA + dropout 0.07 + weight_decay 0.10) — cek per-seed pass-rate.

drop07_wd10 = 1 dari 3 config reg-sweep yg ban-gap-nya < drop06 (ban gap 9.0 vs 9.9, F-P1r), semua 4 kontes >=80.
Full-run 12 BAHASA x 5-seed + bobot (konsisten dgn drop06 full-12; perseed_check tetap nilai 7 scope karena rule di situ).
Tujuan: cek apakah PER-SEED lebih banyak yg lolos semua-7 dibanding drop06 (yg cuma 1/5) -> deploy-seed by-val bisa bersih.

KEEP=all (bobot 5 seed buat deploy by-val). Subprocess-per-seed (VRAM fresh). Resumable.
SEBELUM jalanin: mesin fresh (reboot/kill zombie/tutup Wallpaper Engine) biar nggak crash 0xC0000005.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_drop07wd10_full.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
OVERRIDES = {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.07, "WEIGHT_DECAY": 0.10}
OUT = "outputs/pissa-drop07wd10-full"
N_LANGS = 12   # base default TARGET_LANGS = 12 bahasa (nggak set P1_TARGET_LANGS)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_p1_base.py")


def run():
    print(f"Full-12 drop07_wd10 | {OUT} | seeds {SEEDS} | {N_LANGS} bahasa | KEEP=all")
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
            print(f"! seed {s} gagal (exit {rc}). Resumable, jalanin lagi (reboot dulu kalau 0xC0000005).")
            sys.exit(rc)
    print("\nFull drop07_wd10 SELESAI. Cek per-seed: uv run python scripts/sv_grounding/p1_sweep/perseed_check.py pissa-drop07wd10-full")


if __name__ == "__main__":
    run()
