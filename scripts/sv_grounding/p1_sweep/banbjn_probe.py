"""Probe ban+bjn DOANG (2 bahasa), MULTI-SEED — cari config yang ban/bjn >= FT (nggak minus).

Murah & reliable: cuma 2 bahasa x 5 seed per config (mean jelas, bukan 1-seed noisy). Semua
PiSSA-based (metode terbaik di screening P1: pissa-loradrop05 = ban/bjn dua-duanya dalam ~1σ FT),
mengeksplor kombinasi knob (dropout-rendah, LR-rendah, rank) buat dorong ban/bjn ke >= FT.

Pemenang di sini (ban/bjn >= FT mean) lalu di-FULL-run (12 bahasa) buat cek pemenang lain + gap.
Tiap config = subprocess. Resumable.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/banbjn_probe.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
LANGS = ["ban", "bjn"]
N_LANGS = 2
PISSA_INIT = "pissa"   # ganti ke "pissa" buat full-SVD PiSSA (eksak, bukan randomized)
# folder auto-pisah: niter_16 -> p1-banbjn-probe (lama), pissa -> p1-banbjn-probe-pissa (baru)
FOLDER = "outputs/p1-banbjn-probe" + ("" if PISSA_INIT == "pissa_niter_16" else "-pissa")
METHOD = {"INIT_LORA_WEIGHTS": PISSA_INIT}
CONFIGS = [
    {"tag": "pissa_drop05",        "ov": {"LORA_DROPOUT": 0.05}},                                 # lead screening
    {"tag": "pissa_drop05_lr3e-5", "ov": {"LORA_DROPOUT": 0.05, "FT_LEARNING_RATE": 3e-5}},        # + LR-rendah (bjn-recovery)
    {"tag": "pissa_drop05_lr4e-5", "ov": {"LORA_DROPOUT": 0.05, "FT_LEARNING_RATE": 4e-5}},
    {"tag": "pissa_lr3e-5",        "ov": {"FT_LEARNING_RATE": 3e-5}},                              # LR-rendah, dropout default
    {"tag": "pissa_drop05_r32",    "ov": {"LORA_DROPOUT": 0.05, "LORA_R": 32, "LORA_ALPHA": 64}},  # + kapasitas
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
    print(f"PiSSA init = {PISSA_INIT} | folder = {FOLDER}")
    for cfg in CONFIGS:
        out = f"{FOLDER}/{cfg['tag']}"
        if _config_done(out):
            print(f"--- banbjn:{cfg['tag']} SKIP (lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps({**METHOD, **cfg["ov"]})
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SEEDS)
        env["P1_TARGET_LANGS"] = json.dumps(LANGS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- banbjn:{cfg['tag']} (ban+bjn x {len(SEEDS)} seed)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! banbjn:{cfg['tag']} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\nbanbjn probe SELESAI. Lihat: uv run python scripts/sv_grounding/p1_sweep/banbjn_aggregate.py")


if __name__ == "__main__":
    run()
