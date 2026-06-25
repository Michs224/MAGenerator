"""P1 — LoRA+: sweep HP LoRA+-nya sendiri (lambda, LR, rank, dropout) di sekitar center.

LoRA+ = LR matriks-B lebih tinggi (lr_B = lambda*lr_A). Mekanisme BEDA (optimizer 2-grup).
Signature = lambda. Center = LoRA+ lambda8 @ champion. Perturbasi lambda + sumbu HP lain.
NOTE: integrasi optimizer custom -> SMOKE-TEST 1 bahasa dulu sebelum full (belum keuji-run penuh).

Screening 1 seed (42).  Jalankan dari root:  uv run python scripts/p1_sweep/loraplus_sweep.py
"""
import champion_p1_base as base

FAMILY = "loraplus"
SCREEN_SEEDS = [42]
CONFIGS = [
    {"tag": "lambda4",          "ov": {"LORAPLUS_LR_RATIO": 4}},                       # lambda axis (signature)
    {"tag": "lambda8",          "ov": {"LORAPLUS_LR_RATIO": 8}},                       # center
    {"tag": "lambda16",         "ov": {"LORAPLUS_LR_RATIO": 16}},
    {"tag": "lambda8_lr3e-5",   "ov": {"LORAPLUS_LR_RATIO": 8, "FT_LEARNING_RATE": 3e-5}},   # LR axis
    {"tag": "lambda8_r32",      "ov": {"LORAPLUS_LR_RATIO": 8, "LORA_R": 32, "LORA_ALPHA": 64}},  # rank axis
    {"tag": "lambda8_loradrop20", "ov": {"LORAPLUS_LR_RATIO": 8, "LORA_DROPOUT": 0.20}},     # dropout axis
]


def run():
    for cfg in CONFIGS:
        base.reset_champion_defaults()
        base.SEEDS = SCREEN_SEEDS
        base.KEEP_WEIGHTS = "none"        # screening: nggak usah simpan bobot (winner retrain di confirm)
        for k, v in cfg["ov"].items():
            setattr(base, k, v)
        base.OUTPUT_ROOT = f"outputs/p1-{FAMILY}-sweep/{cfg['tag']}"
        print(f"\n--- {FAMILY}:{cfg['tag']}")
        base.main()


if __name__ == "__main__":
    run()
