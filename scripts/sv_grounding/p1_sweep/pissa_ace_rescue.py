"""PiSSA varian yang JAGA ace (belum pernah ditest ace) sambil PERTAHANKAN recover ban/bjn/mad.

Temuan vanilla_probe: dropout/hidden_dropout MURNI TIDAK BISA naikin ban (semua varian <= CH 79.86) --
ban CUMA naik lewat PiSSA-init (SVD align ke arah utama backbone). Tapi PiSSA-init+drop05 ngerusak ace
(79.40, -2.81 vs CH). Hipotesis: drop05 (kapasitas lebih dipakai) yg bikin ace jebol, bukan PiSSA-init
sendiri (PiSSA+dropout0.1 -- screening 1-seed lama -- ban 81.68 bagus tapi bjn 81.24 hancur, ace blm dites).

Configs (ace/ban/bjn/mad, 5-seed):
  r32       -> PiSSA+drop05+rank32 (lebih banyak arah singular dipertahankan, truncation kurang agresif
              -> hipotesis: kurangi bias-arah yg ngerusak ace, sambil tetap bantu ban/bjn/mad)
  drop08    -> PiSSA+dropout0.08 (versi lebih lembut dari drop05, antara champion 0.1 dan drop05 0.05)
  default   -> PiSSA-init SAJA, dropout tetap champion 0.1 (konfirmasi 5-seed + ace utk temuan 1-seed lama)

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_ace_rescue.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
LANGS = ["ace", "ban", "bjn", "mad"]
N_LANGS = 4
FOLDER = "outputs/p1-pissa-ace-rescue"
CONFIGS = [
    {"tag": "r32",     "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.05, "LORA_R": 32, "LORA_ALPHA": 64}},
    {"tag": "drop08",  "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.08}},
    {"tag": "default", "ov": {"INIT_LORA_WEIGHTS": "pissa"}},   # dropout tetap champion 0.1
    # micro-search sweet-spot (ace puncak@0.08, ban/mad puncak@0.05, non-monotonik -> coba antara)
    {"tag": "drop06",  "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.06}},
    {"tag": "drop07",  "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.07}},
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
            print(f"--- pissa-rescue:{cfg['tag']} SKIP (lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps(cfg["ov"])
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SEEDS)
        env["P1_TARGET_LANGS"] = json.dumps(LANGS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- pissa-rescue:{cfg['tag']} (ace+ban+bjn+mad x {len(SEEDS)} seed)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! pissa-rescue:{cfg['tag']} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\npissa_ace_rescue SELESAI. Lihat: uv run python scripts/sv_grounding/p1_sweep/vanilla_aggregate.py")


if __name__ == "__main__":
    run()
