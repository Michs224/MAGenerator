"""P1 — PiSSA: sweep HP PiSSA-nya sendiri (LR, rank, scaling, dropout) di sekitar center.

PiSSA = init adapter dari SVD top-r bobot -> plateau lebih tinggi. Mekanisme BEDA DoRA (init).
Center = PiSSA @ resep champion (lr5e-5/r16/a32). Tiap config perturbasi SATU sumbu HP.
NOTE: PiSSA ubah residual weight -> saat confirm/deploy WAJIB reload-and-eval.

Tiap config = SUBPROCESS terpisah (VRAM fresh antar-config). Resumable.
Screening 1 seed (42).  Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_sweep.py
"""
import os
import sys
import json
import subprocess

FAMILY = "pissa"
SCREEN_SEEDS = [42]
N_LANGS = 12
METHOD = {"INIT_LORA_WEIGHTS": "pissa_niter_16"}
CONFIGS = [
    {"tag": "center_lr5e-5_r16", "ov": {}},
    {"tag": "lr3e-5",            "ov": {"FT_LEARNING_RATE": 3e-5}},
    {"tag": "r32",               "ov": {"LORA_R": 32, "LORA_ALPHA": 64}},  # PiSSA + kapasitas lebih
    {"tag": "scaling1_a16",      "ov": {"LORA_ALPHA": 16}},
    {"tag": "scaling4_a64",      "ov": {"LORA_ALPHA": 64}},
    {"tag": "loradrop05",        "ov": {"LORA_DROPOUT": 0.05}},
    {"tag": "loradrop20",        "ov": {"LORA_DROPOUT": 0.20}},
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
