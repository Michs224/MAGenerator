"""Reg-sweep di atas drop06 (PiSSA r16 + dropout 0.06) — dorong ban gap FIRMLY <10 tanpa F1<80.

Konteks (F-P1n/p): drop06 lolos semua 7 di MEAN, TAPI ban gap mean 9.9 (mepet 10, noisy 6.6-10.8) -> seed
deploy by-val (seed 42) ban gap 10.77 (lewat). Butuh config yg turunin ban gap mean ke ~8 (firmly <10) biar
seed by-val lolos bersih. Sumbu BELUM disentuh: weight_decay (default 0.05) + hidden_dropout (default 0.25).
weight_decay = regularisasi lebih lembut dari dropout (biasanya gap turun tanpa banyak korban F1).

6 config TRIMMED (hemat: 120 training vs 240), semua PiSSA + dropout 0.06 basis, ace/ban/bjn/mad x 5-seed:
  wd10 / wd15       -> weight_decay (reg lembut, default 0.05) = taruhan utama
  hd30              -> hidden_dropout 0.25->0.30 (backbone reg)
  wd10_hd30         -> kombo reg
  r12               -> rank 16->12 (kapasitas-turun mild = gap turun; risiko F1)
  last4layers       -> cuma 4 layer di-LoRA (kapasitas-turun beda mekanisme)
Catatan: reg (wd/hidden) = gap turun lembut, F1 kejaga; kapasitas-turun (r12/last4) = gap turun agresif, bisa F1<80.
Yg lolos (ban gap firmly<10 & semua>=80) = kandidat kuat -> full-run 7-scope 5-seed + bobot. Incumbent = drop06.

Target: ban gap <10 FIRMLY (idealnya ~8) SAMBIL ace/ban/bjn/mad semua >=80. Config yg lolos + ban gap lebih
rendah dari drop06 -> full-run 12 + multi-seed konfirmasi (biar seed by-val lolos). Incumbent = drop06.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/pissa_reg_sweep.py
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
# TRIMMED 12->6 (hemat waktu): 6 titik terbaik lintas 4 sumbu. Yg dibuang: wd08(~drop06), wd20(ekstrem),
# hd35/wd15_hd35(marginal), r8(F1 pasti<80), drop07_wd10(drop07 udah kalah). 6 x 4 bahasa x 5 seed = 120 training.
CONFIGS = [
    {"tag": "wd10",        "ov": {"WEIGHT_DECAY": 0.10}},                              # weight_decay (utama)
    {"tag": "wd15",        "ov": {"WEIGHT_DECAY": 0.15}},
    {"tag": "hd30",        "ov": {"HIDDEN_DROPOUT": 0.30}},                            # backbone reg
    {"tag": "wd10_hd30",   "ov": {"WEIGHT_DECAY": 0.10, "HIDDEN_DROPOUT": 0.30}},       # kombo reg
    {"tag": "r12",         "ov": {"LORA_R": 12, "LORA_ALPHA": 24}},                     # kapasitas-turun mild
    {"tag": "last4layers", "ov": {"LORA_ALL_LAYERS": False}},                           # kapasitas-turun (layer)
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
            print(f"--- reg:{cfg['tag']} SKIP (lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps({**BASE_OV, **cfg["ov"]})
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SEEDS)
        env["P1_TARGET_LANGS"] = json.dumps(LANGS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- reg:{cfg['tag']} (ace+ban+bjn+mad x {len(SEEDS)} seed) ov={cfg['ov']}")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! reg:{cfg['tag']} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\nreg sweep SELESAI. Lihat: uv run python scripts/sv_grounding/p1_sweep/vanilla_aggregate.py")


if __name__ == "__main__":
    run()
