"""E6b — GENTLE stage-0 sweep: uji apakah LR-kecil + 5 epoch (× multiseed) menyelamatkan ACE.

DIAGNOSIS (dari E6): ace jebol (test 75.78, gap 10.76) karena stage-0 default (LR 2e-5) OVER-spesialisasi ke
Indonesia → init ace rusak (zero-shot cuma 64.04, terburuk 7 bahasa) → stage-1 harus ngapalin. Hipotesis: stage-0
LEBIH LEMBUT (LR kecil) preserve fitur multibahasa → ace zero-shot naik → stage-1 nggak ngapalin. (LR kecil + epoch
naik itu KONSISTEN, bukan kontradiksi: Mosbach ICLR21 — small-LR + more-iter = generalisasi lebih stabil.)

GATE MURAH (script ini): latih gentle stage-0 di grid (LR × seed), ukur **ace zero-shot** (+ 7 bahasa konteks) —
TANPA stage-1. Kalau ace zero-shot NAIK jauh dari 64 → lanjut stage-1 ace (script terpisah). Kalau tetap ~64 →
pendekatan SmSA mati utk ace, STOP (hemat semua compute stage-1). Reuse fetch+leakage-filter dari stilt_smsa.

Jalankan dari root:  uv run python scripts/sv_grounding_2/stilt_gentle.py
"""
import os
import json
import shutil

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          TrainingArguments, Trainer, EarlyStoppingCallback, BertTokenizerFast)

import stilt_smsa as base   # reuse: fetch_smsa, leakage_filter, make_ds, compute_metrics, gate_zeroshot, konstanta

# --- GRID gentle stage-0 (semua LEBIH lembut dari default 2e-5/3ep) ---
GRID_LR = [1e-5, 5e-6]          # setengah & seperempat LR default
GRID_SEED = [42, 0, 1]         # multiseed (nutup caveat "stage-0 1-run" ala R3)
EPOCHS = 5                      # user: naikin ke 5 (dengan LR kecil = aman, bukan over-spesialisasi)
PATIENCE = 3
OUT = "outputs/sv2-smsa-gentle"
BASELINE_ACE_ZS = 64.04        # ace zero-shot stage-0 default (LR2e-5/3ep) — pembanding
GATE_ACE_ZS = 70.0             # ambang "worth lanjut stage-1": ace zero-shot >= 70 (naik >~6 dari baseline)


def train_gentle(smsa, tok, lr, seed, out_dir):
    cfg = AutoConfig.from_pretrained(base.MODEL_CHECKPOINT)
    cfg.num_labels = 3; cfg.label2id = base.LABEL2ID
    cfg.id2label = {i: v for i, v in enumerate(base.LABEL_LIST)}
    ds_tr = base.make_ds(smsa["train_filtered"], tok)
    ds_va = base.make_ds(smsa["valid"], tok)
    collator = DataCollatorWithPadding(tokenizer=tok)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(base.MODEL_CHECKPOINT, config=cfg,
                                                                  ignore_mismatched_sizes=True)
    args = TrainingArguments(
        output_dir=f"{out_dir}/run", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=16, per_device_eval_batch_size=64, optim="adamw_torch_fused",
        learning_rate=lr, weight_decay=0.01, num_train_epochs=EPOCHS, warmup_ratio=0.1,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1", save_only_model=True,
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed)
    trainer = Trainer(model_init=model_init, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                      processing_class=tok, data_collator=collator, compute_metrics=base.compute_metrics,
                      callbacks=[EarlyStoppingCallback(PATIENCE)])
    trainer.train()
    val = float(trainer.evaluate(ds_va)["eval_f1"])
    os.makedirs(f"{out_dir}/best", exist_ok=True)
    trainer.save_model(f"{out_dir}/best"); tok.save_pretrained(f"{out_dir}/best")
    d = f"{out_dir}/run"
    if os.path.isdir(d):
        for e in os.listdir(d):
            if e.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(d, e), ignore_errors=True)
    return trainer, val


def main():
    print(f"CUDA={torch.cuda.is_available()} | GENTLE stage-0 sweep (LR {GRID_LR} × seed {GRID_SEED}, {EPOCHS}ep)")
    tok = BertTokenizerFast.from_pretrained(base.MODEL_CHECKPOINT)
    smsa = base.fetch_smsa()
    filtered, _ = base.leakage_filter(smsa["train"])
    smsa["train_filtered"] = filtered
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for lr in GRID_LR:
        for seed in GRID_SEED:
            tag = f"lr{lr:.0e}_seed{seed}"
            out_dir = f"{OUT}/{tag}"
            sp = f"{out_dir}/zeroshot.json"
            if os.path.exists(sp):
                d = json.load(open(sp)); rows.append((tag, d["smsa_val"], d["zeroshot"]));
                print(f"--- {tag} SKIP (ada)"); continue
            print(f"\n--- {tag} (lr={lr}, seed={seed}, {EPOCHS}ep)")
            trainer, val = train_gentle(smsa, tok, lr, seed, out_dir)
            zs = base.gate_zeroshot(trainer, tok)
            json.dump({"lr": lr, "seed": seed, "epochs": EPOCHS, "smsa_val": val, "zeroshot": zs}, open(sp, "w"), indent=2)
            rows.append((tag, val, zs))
            del trainer; torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"{'config':16} {'smsaVal':>7} | {'ACE-zs':>7} {'ban':>6} {'bjn':>6} {'jav':>6} {'mad':>6} {'min':>6} {'sun':>6} {'mean':>6}")
    print(f"{'DEFAULT(2e-5/3ep)':16} {'92.20':>7} | {BASELINE_ACE_ZS:7.2f} {'74.32':>6} {'79.42':>6} {'87.30':>6} {'70.43':>6} {'81.56':>6} {'80.96':>6} {'76.86':>6}  <- pembanding")
    best = None
    for tag, val, zs in rows:
        m = np.mean(list(zs.values()))
        flag = " ***" if zs["ace"] >= GATE_ACE_ZS else ""
        print(f"{tag:16} {val*100:7.2f} | {zs['ace']:7.2f} {zs['ban']:6.1f} {zs['bjn']:6.1f} {zs['jav']:6.1f} {zs['mad']:6.1f} {zs['min']:6.1f} {zs['sun']:6.1f} {m:6.2f}{flag}")
        if best is None or zs["ace"] > best[1]:
            best = (tag, zs["ace"])
    print(f"\nACE zero-shot terbaik: {best[0]} = {best[1]:.2f} (baseline default {BASELINE_ACE_ZS}, gate lanjut >= {GATE_ACE_ZS})")
    if best[1] >= GATE_ACE_ZS:
        print(f"✅ GATE PASS → gentle stage-0 preserve fitur ace. Lanjut stage-1 ace dari: {OUT}/{best[0]}/best")
        print("   (P1_TARGET_LANGS=['ace'], multiseed, resep drop06 — lapor ke asisten buat setup)")
    else:
        print(f"❌ GATE STOP → gentle pun tak angkat ace zero-shot ke >={GATE_ACE_ZS}. Jarak ace↔Indonesia struktural.")
        print("   SmSA = jalur buntu utk ace. Tutup buku, drop06 champion.")
    print(f"\nHasil per-config: {OUT}/*/zeroshot.json")


if __name__ == "__main__":
    main()
