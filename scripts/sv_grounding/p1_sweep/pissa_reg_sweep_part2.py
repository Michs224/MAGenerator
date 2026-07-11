"""Reg-sweep PART 2 — 6 permutasi SISA (yang di-cut dari sweep awal biar total 12 permutasi).

Part 1 (pissa_reg_sweep.py, UDAH dijalanin) = wd10/wd15/hd30/wd10_hd30/r12/last4layers.
Part 2 (ini) = 6 titik sisa buat coverage lebih luas: weight_decay ekstrem (wd08/wd20), hidden_dropout kuat
(hd35 + kombo wd15_hd35), kapasitas-turun agresif (r8), dropout+wd (drop07_wd10).

Dibatch 6+6 biar bisa REBOOT antar-batch (mesin numpuk python-zombie -> resource exhaustion -> crash
0xC0000005 kalau sweep kelamaan). Tulis ke folder SAMA (`outputs/p1-pissa-reg`) -> vanilla_aggregate baca semua.
Resumable (skip yg lengkap).

SEBELUM jalanin: mesin fresh dulu (reboot / kill zombie python / tutup Wallpaper Engine) biar nggak crash.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_reg_sweep_part2.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
LANGS = ["ace", "ban", "bjn", "mad"]
N_LANGS = 4
FOLDER = "outputs/p1-pissa-reg"
BASE_OV = {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.06}   # basis = drop06
CONFIGS = [
    {"tag": "wd08",        "ov": {"WEIGHT_DECAY": 0.08}},                              # wd lebih lembut
    {"tag": "wd20",        "ov": {"WEIGHT_DECAY": 0.20}},                              # wd ekstrem
    {"tag": "hd35",        "ov": {"HIDDEN_DROPOUT": 0.35}},                            # backbone reg kuat
    {"tag": "wd15_hd35",   "ov": {"WEIGHT_DECAY": 0.15, "HIDDEN_DROPOUT": 0.35}},       # kombo reg kuat
    {"tag": "r8",          "ov": {"LORA_R": 8, "LORA_ALPHA": 16}},                      # kapasitas-turun agresif
    {"tag": "drop07_wd10", "ov": {"LORA_DROPOUT": 0.07, "WEIGHT_DECAY": 0.10}},          # dropout + wd
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
            print(f"--- reg2:{cfg['tag']} SKIP (lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps({**BASE_OV, **cfg["ov"]})
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SEEDS)
        env["P1_TARGET_LANGS"] = json.dumps(LANGS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- reg2:{cfg['tag']} (ace+ban+bjn+mad x {len(SEEDS)} seed) ov={cfg['ov']}")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! reg2:{cfg['tag']} gagal (exit {rc}). Resumable, jalanin lagi (reboot dulu kalau 0xC0000005).")
            sys.exit(rc)
    print("\nreg PART 2 SELESAI. Lihat: uv run python scripts/sv_grounding/p1_sweep/vanilla_aggregate.py")


if __name__ == "__main__":
    run()
