"""P1 — PiSSA: sweep HP PiSSA-nya sendiri (LR, rank, scaling, dropout) di sekitar center.

PiSSA = init adapter dari SVD top-r bobot -> plateau lebih tinggi, rank sama. Mekanisme BEDA DoRA (init).
Center = PiSSA @ resep champion (lr5e-5/r16/a32). Tiap config perturbasi SATU sumbu HP.
NOTE: PiSSA ubah residual weight -> saat confirm/deploy WAJIB reload-and-eval.

Screening 1 seed (42).  Jalankan dari root:  uv run python scripts/p1_sweep/pissa_sweep.py
"""
import champion_p1_base as base

FAMILY = "pissa"
SCREEN_SEEDS = [42]
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


def run():
    for cfg in CONFIGS:
        base.reset_champion_defaults()
        base.SEEDS = SCREEN_SEEDS
        base.KEEP_WEIGHTS = "none"        # screening: nggak usah simpan bobot (winner retrain di confirm)
        for k, v in {**METHOD, **cfg["ov"]}.items():
            setattr(base, k, v)
        base.OUTPUT_ROOT = f"outputs/p1-{FAMILY}-sweep/{cfg['tag']}"
        print(f"\n--- {FAMILY}:{cfg['tag']}")
        base.main()


if __name__ == "__main__":
    run()
