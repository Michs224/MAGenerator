"""Dorong ace/bjn lebih DEKET FT tanpa ngerusak pencapaian drop06 (semua >=80 & gap<10).

Konteks: drop06 (PiSSA r16 + dropout 0.06) LOLOS semua 7 scope (>=80 & gap<10, F-P1n), TAPI ace -1.30 &
bjn -1.93 masih lewat "~1pt dari FT". Hipotesis: ace/bjn di bawah FT krn r16 KAPASITAS-TERBATAS (FT kapasitas
penuh) -> tambah kapasitas (rank 20/24) mungkin narik F1 ace/bjn ke arah FT, sambil dropout jaga gap<10.
Ruang 2D (rank x dropout) di sekitar drop06. Risiko: rank naik -> gap naik (r32 dulu gap 10-14); drop06/08
buat kontrol gap.

Configs (ace/ban/bjn/mad, 5-seed) -- semua PiSSA-init:
  r20_drop06 -> kapasitas milder (r20/a40) + dropout 0.06
  r24_drop06 -> kapasitas lebih (r24/a48) + dropout 0.06
  r24_drop08 -> r24 + dropout 0.08 (extra reg kalau r24_drop06 gap>10)

Target: ace/bjn F1 NAIK ke arah FT (ace ke >81, bjn ke >85.5) SAMBIL semua tetap >=80 & gap<10. Kalau ada yg
lolos + ace/bjn lebih deket FT dari drop06 -> full-run 12. Kalau gap meledak / F1 ga naik -> drop06 tetap terbaik.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_capacity_probe.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
LANGS = ["ace", "ban", "bjn", "mad"]
N_LANGS = 4
FOLDER = "outputs/p1-pissa-capacity"
CONFIGS = [
    {"tag": "r20_drop06", "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.06, "LORA_R": 20, "LORA_ALPHA": 40}},
    {"tag": "r24_drop06", "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.06, "LORA_R": 24, "LORA_ALPHA": 48}},
    {"tag": "r24_drop08", "ov": {"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.08, "LORA_R": 24, "LORA_ALPHA": 48}},
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
            print(f"--- capacity:{cfg['tag']} SKIP (lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps(cfg["ov"])
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SEEDS)
        env["P1_TARGET_LANGS"] = json.dumps(LANGS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- capacity:{cfg['tag']} (ace+ban+bjn+mad x {len(SEEDS)} seed)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! capacity:{cfg['tag']} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\ncapacity probe SELESAI. Lihat: uv run python scripts/sv_grounding/p1_sweep/vanilla_aggregate.py")


if __name__ == "__main__":
    run()
