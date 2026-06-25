"""P1 — DoRA: sweep HP DoRA-nya sendiri (LR, rank, scaling, dropout) di sekitar center.

DoRA = dekomposisi magnitude+arah. Center = DoRA @ resep champion (lr5e-5/r16/a32/lora_drop0.1)
= persis run DoRA-vanilla yang DITOLAK di P0 (redistribusi). Tiap config perturbasi SATU sumbu HP
dari center -> ngecek apakah HP DoRA yang beda mecahin redistribusi. (Vanilla LoRA TIDAK di-sweep
lagi; champion-nya udah hasil tuning vanilla sebelumnya.)

Screening 1 seed (42).  Jalankan dari root:  uv run python scripts/p1_sweep/dora_sweep.py
"""
import champion_p1_base as base

FAMILY = "dora"
SCREEN_SEEDS = [42]
METHOD = {"USE_DORA": True}                 # di-apply ke SEMUA config
CONFIGS = [
    {"tag": "center_lr5e-5_r16", "ov": {}},                              # DoRA @ champion (ref)
    {"tag": "lr4e-5",            "ov": {"FT_LEARNING_RATE": 4e-5}},       # LR axis
    {"tag": "lr3e-5",            "ov": {"FT_LEARNING_RATE": 3e-5}},
    {"tag": "r8",                "ov": {"LORA_R": 8,  "LORA_ALPHA": 16}}, # rank axis (DoRA kuat di rank rendah; held scaling)
    {"tag": "r32",               "ov": {"LORA_R": 32, "LORA_ALPHA": 64}},
    {"tag": "scaling1_a16",      "ov": {"LORA_ALPHA": 16}},               # scaling axis (alpha/r): 1 & 4
    {"tag": "scaling4_a64",      "ov": {"LORA_ALPHA": 64}},
    {"tag": "loradrop05",        "ov": {"LORA_DROPOUT": 0.05}},           # dropout axis
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
