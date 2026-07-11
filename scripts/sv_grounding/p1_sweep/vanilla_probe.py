"""Probe VANILLA LoRA (NO PiSSA/DoRA init) di 4 bahasa kontes (ace/ban/bjn/mad), 5-seed.

Tujuan: isolasi confound. pissa_drop05 = PiSSA-init + dropout0.05 SEKALIGUS (2 variabel), belum pernah
diisolasi. Data screening (1-seed) nunjukin PiSSA-init SENDIRIAN bikin bjn ANCUR (81.24, -5.54) --
dropout0.05 yang kerjanya nyelametin bjn (balik ke 85.46 pas digabung). Jadi:
  drop05            -> vanilla + dropout0.05 SAJA (isolasi: dropout doang, tanpa PiSSA) = sel yang hilang
  hiddendrop20      -> axis BELUM PERNAH disweep: backbone-wide dropout (turun dari champion 0.25)
  hiddendrop15      -> versi lebih agresif dari axis di atas
  drop05_hiddendrop20 -> kombo (cuma relevan diinterpretasi kalau 2 di atas individually promising)

Scope 4 bahasa (bukan cuma ban/bjn): ace ikut diperiksa krn ace = bahasa yg jebol di PiSSA -- mau tau
apakah vanilla+regularisasi-baru bisa recover ban/bjn/mad TANPA korbanin ace (yg PiSSA gagal jaga).

Kalau ada yg ban&bjn&mad>=80/gap<10 DAN ace tetap aman -> full-run 12 bahasa. Kalau nggak -> konfirmasi
lagi bahwa tembok bias-variance nggak bisa dihindari knob apapun (vanilla ataupun PiSSA/DoRA/LoRA+).

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/vanilla_probe.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1, 2, 3]
LANGS = ["ace", "ban", "bjn", "mad"]
N_LANGS = 4
FOLDER = "outputs/p1-vanilla-probe"
CONFIGS = [
    {"tag": "drop05",              "ov": {"LORA_DROPOUT": 0.05}},                            # isolasi dropout (no PiSSA)
    {"tag": "hiddendrop20",        "ov": {"HIDDEN_DROPOUT": 0.20}},                           # axis baru: backbone dropout
    {"tag": "hiddendrop15",        "ov": {"HIDDEN_DROPOUT": 0.15}},                           # axis baru, lebih agresif
    {"tag": "drop05_hiddendrop20", "ov": {"LORA_DROPOUT": 0.05, "HIDDEN_DROPOUT": 0.20}},      # kombo (opsional interpretasi)
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
            print(f"--- vanilla:{cfg['tag']} SKIP (lengkap)")
            continue
        env = dict(os.environ)
        env["P1_OVERRIDES"] = json.dumps(cfg["ov"])   # NO INIT_LORA_WEIGHTS -> tetap vanilla
        env["P1_OUTPUT_ROOT"] = out
        env["P1_SEEDS"] = json.dumps(SEEDS)
        env["P1_TARGET_LANGS"] = json.dumps(LANGS)
        env["P1_KEEP_WEIGHTS"] = "none"
        print(f"\n--- vanilla:{cfg['tag']} (ace+ban+bjn+mad x {len(SEEDS)} seed)")
        rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
        if rc != 0:
            print(f"! vanilla:{cfg['tag']} gagal (exit {rc}). Resumable, jalanin lagi.")
            sys.exit(rc)
    print("\nvanilla probe SELESAI. Lihat: uv run python scripts/sv_grounding/p1_sweep/vanilla_aggregate.py")


if __name__ == "__main__":
    run()
