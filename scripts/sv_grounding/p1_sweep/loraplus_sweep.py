"""P1 — LoRA+: sweep HP LoRA+-nya sendiri (lambda, LR, rank, dropout) di sekitar center.

LoRA+ = LR matriks-B lebih tinggi (lr_B = lambda*lr_A). Mekanisme BEDA (optimizer 2-grup).
Signature = lambda. Center = LoRA+ lambda8 @ champion. Perturbasi lambda + sumbu HP lain.
NOTE: integrasi optimizer custom -> SMOKE-TEST 1 bahasa dulu sebelum full (belum keuji-run penuh).

Tiap config = SUBPROCESS terpisah (VRAM fresh antar-config). Resumable.
Screening 1 seed (42).  Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/loraplus_sweep.py
"""
import os
import sys
import json
import subprocess

FAMILY = "loraplus"
SCREEN_SEEDS = [42]
N_LANGS = 12
METHOD = {}
CONFIGS = [
    {"tag": "lambda4",            "ov": {"LORAPLUS_LR_RATIO": 4}},                                  # lambda axis (signature)
    {"tag": "lambda8",            "ov": {"LORAPLUS_LR_RATIO": 8}},                                  # center
    {"tag": "lambda16",           "ov": {"LORAPLUS_LR_RATIO": 16}},
    {"tag": "lambda8_lr3e-5",     "ov": {"LORAPLUS_LR_RATIO": 8, "FT_LEARNING_RATE": 3e-5}},        # LR axis
    {"tag": "lambda8_r32",        "ov": {"LORAPLUS_LR_RATIO": 8, "LORA_R": 32, "LORA_ALPHA": 64}},  # rank axis
    {"tag": "lambda8_loradrop20", "ov": {"LORAPLUS_LR_RATIO": 8, "LORA_DROPOUT": 0.20}},            # dropout axis
]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_p1_base.py")


def _config_done(output_root):
    sp = os.path.join(ROOT, output_root, f"seed_{SCREEN_SEEDS[0]}", "results_summary.json")
    if not os.path.exists(sp):
        return False
    try:
        return len(json.load(open(sp))["results"]) >= N_LANGS
    except Exception:
        return False


def run():
    for cfg in CONFIGS:
        out = f"outputs/p1-{FAMILY}-sweep/{cfg['tag']}"
        if _config_done(out):
            print(f"--- {FAMILY}:{cfg['tag']} SKIP (config sudah lengkap {N_LANGS} bahasa)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps({**METHOD, **cfg["ov"]})
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SCREEN_SEEDS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- {FAMILY}:{cfg['tag']} (subprocess, VRAM fresh)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! {FAMILY}:{cfg['tag']} gagal (exit {rc}). Berhenti — resumable, jalanin lagi.")
            sys.exit(rc)
    print(f"\n{FAMILY} sweep SELESAI.")


if __name__ == "__main__":
    run()
