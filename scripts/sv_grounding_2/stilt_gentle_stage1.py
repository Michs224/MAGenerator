"""Stage-1 ace-only, MULTI-SOURCE matched-seed — poles ace di atas beberapa stage-0 sekaligus.

Konteks: bandingin apakah stage-0 mana pun (gentle-LR variasi ATAU LoRA) bisa angkat ace ke >=80 di stage-1.
Matched-seed (stage-1 seed s <- stage-0 seed s) = anti cherry-pick (pelajaran R3). Resep stage-1 = drop06 FIXED
(init pissa, drop 0.06) — jangan permutasi stage-1 HP (confound). Subprocess per (source,seed), CMD-friendly.

Baseline: gentle-5e-6 stage-1 ace = 78.99 (done, di outputs/sv2-stilt-gentle). drop06 ace = 80.56.
Tambah source LoRA (Job A) HABIS gate-nya ketahuan config terbaik — tinggal append ke SOURCES.

Jalankan dari root:  uv run python scripts/sv_grounding_2/stilt_gentle_stage1.py
"""
import os
import sys
import json
import subprocess

SEEDS = [42, 0, 1]                        # matched: stage-1 seed s <- stage-0 seed s
# TEMUAN 2026-07-07: ace TEMBUS 80 konsisten (2.5e-6: 81.16±0.71 semua seed >=80, NGALAHIN drop06 80.56;
# 1e-6ep15: 80.92±0.30). PELAJARAN: gate zero-shot SALAH (5e-6 init terbaik tapi stage-1 terburuk; 1e-6ep15
# seed1 init 56.6 tetap stage-1 80.5) -> zero-shot != fine-tunability. Maka LoRA (Job A, zs 69.2) TAK BOLEH
# dicoret dari zero-shot; harus diuji stage-1 juga.
# CHAMPION TEST: 7 bahasa (bukan ace-only lagi) di stage-0 terbaik + LoRA terbaik.
LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]   # 7 scope (ace-only fase sebelumnya = outputs/sv2-stilt-*)
SOURCES = [
    ("g2p5e6-7l",  "outputs/sv2-smsa-gentle/lr2.5e-06_seed{s}/best"),          # ace 81.16 (TERBAIK) -> uji 7 bahasa
    ("g1e6ep15-7l", "outputs/sv2-smsa-gentle/lr1e-06ep15_seed{s}/best"),        # ace 80.92 -> uji 7 bahasa
    ("lora1e4r16-7l", "outputs/sv2-smsa-lora/lora_lr0.0001_r16ep10_seed{s}/best"),  # LoRA terbaik (zs 69.2, tapi gate zs tak reliabel)
]
OUT_ROOT = "outputs/sv2-stilt"            # -> {OUT_ROOT}-{tag}/seed_{s}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "scripts", "sv_grounding", "p1_sweep", "champion_p1_base.py")


def run():
    if not os.path.exists(BASE):
        sys.exit(f"champion_p1_base tak ketemu: {BASE}")
    for tag, tmpl in SOURCES:
        out = f"{OUT_ROOT}-{tag}"
        for s in SEEDS:
            ckpt = tmpl.format(s=s)
            if not os.path.exists(os.path.join(ROOT, ckpt, "model.safetensors")):
                print(f"! [{tag}] stage-0 seed {s} tak ada ({ckpt}) — SKIP.")
                continue
            sp = os.path.join(ROOT, out, f"seed_{s}", "results_summary.json")
            if os.path.exists(sp) and all(l in json.load(open(sp)).get("results", {}) for l in LANGS):
                print(f"--- [{tag}] seed {s} SKIP ({len(LANGS)} bahasa sudah ada)")
                continue
            env = dict(os.environ)
            env["P1_OVERRIDES"] = json.dumps({"INIT_LORA_WEIGHTS": "pissa", "LORA_DROPOUT": 0.06,
                                              "MODEL_CHECKPOINT": ckpt})
            env["P1_OUTPUT_ROOT"] = out
            env["P1_SEEDS"] = json.dumps([s])
            env["P1_TARGET_LANGS"] = json.dumps(LANGS)
            env["P1_KEEP_WEIGHTS"] = "none"
            print(f"\n--- stage-1 ace [{tag}] seed {s} <- {ckpt}")
            rc = subprocess.run([sys.executable, BASE], env=env, cwd=ROOT).returncode
            if rc != 0:
                sys.exit(f"! [{tag}] seed {s} gagal (exit {rc}). Resumable, jalanin lagi.")
    print(f"\nSELESAI. Hasil per source di {OUT_ROOT}-<tag>/seed_*/results_summary.json — lapor ke asisten buat agregat.")


if __name__ == "__main__":
    run()
