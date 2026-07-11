"""P1 — DoRA: sweep HP DoRA-nya sendiri (LR, rank, scaling, dropout) di sekitar center.

DoRA = dekomposisi magnitude+arah. Center = DoRA @ resep champion (lr5e-5/r16/a32/lora_drop0.1)
= persis run DoRA-vanilla yang DITOLAK di P0 (redistribusi). Tiap config perturbasi SATU sumbu HP.

Tiap config dijalanin sebagai SUBPROCESS terpisah (champion_p1_base.py) -> VRAM dibersihin total
antar-config (cegah fragmentasi numpuk di GPU kecil; fix Windows-compatible, eval tetap 64, fair).
Resumable: config yang results_summary-nya udah 12 bahasa di-skip; partial lanjut di dalam subprocess.

Screening 1 seed (42).  Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/dora_sweep.py
"""
import os
import sys
import json
import subprocess

FAMILY = "dora"
SCREEN_SEEDS = [42]
N_LANGS = 12
METHOD = {"USE_DORA": True}                 # di-apply ke SEMUA config
CONFIGS = [
    {"tag": "center_lr5e-5_r16", "ov": {}},                              # DoRA @ champion (ref)
    {"tag": "lr4e-5",            "ov": {"FT_LEARNING_RATE": 4e-5}},       # LR axis
    {"tag": "lr3e-5",            "ov": {"FT_LEARNING_RATE": 3e-5}},
    {"tag": "r8",                "ov": {"LORA_R": 8,  "LORA_ALPHA": 16}}, # rank axis (held scaling)
    {"tag": "r32",               "ov": {"LORA_R": 32, "LORA_ALPHA": 64}},
    {"tag": "scaling1_a16",      "ov": {"LORA_ALPHA": 16}},               # scaling axis (alpha/r): 1 & 4
    {"tag": "scaling4_a64",      "ov": {"LORA_ALPHA": 64}},
    {"tag": "loradrop05",        "ov": {"LORA_DROPOUT": 0.05}},           # dropout axis
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
        env["P1_KEEP_WEIGHTS"] = "none"            # screening: nggak simpan bobot
        print(f"\n--- {FAMILY}:{cfg['tag']} (subprocess, VRAM fresh)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! {FAMILY}:{cfg['tag']} gagal (exit {rc}). Berhenti — resumable, jalanin lagi.")
            sys.exit(rc)
    print(f"\n{FAMILY} sweep SELESAI.")


if __name__ == "__main__":
    run()
