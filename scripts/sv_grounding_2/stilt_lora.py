"""E6c — LoRA stage-0: gentleness lewat PARAMETER BEKU (bukan LR/epoch). Dimensi terakhir eksplorasi SmSA.

DIAGNOSIS: full-FT SmSA over-spesialisasi -> seluruh backbone ke-update Indonesia -> fitur multibahasa ace terkikis.
LEVER BARU (beda dari gentle-LR): stage-0 pakai LoRA -> backbone BEKU, cuma adapter kecil + classifier belajar SmSA.
Fitur ace di FFN/embedding/LayerNorm TAK tersentuh; cuma QKV dapat 'bumbu' Indonesia low-rank saat di-merge.
Hipotesis: preserve ace lebih baik dari SEMUA config gentle-LR (yg puncaknya ace-zs 71 @ 5e-6).

Pipeline: LoRA stage-0 (r16/a32 QKV all-layer, classifier trainable) -> merge adapter ke backbone -> simpan sbg
BertForSequenceClassification (stage-1 load identik) -> gate zero-shot. Reuse fetch+leakage-filter dari stilt_smsa.
Grid: LR {2e-4} (LoRA butuh LR lebih tinggi dari full-FT) × epoch 10 × seed {42,0,1}. (r16 QKV = sebanding champion.)

Jalankan dari root:  uv run python scripts/sv_grounding_2/stilt_lora.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # jaga2 OOM (LoRA lebih ringan tapi aman)
import json
import shutil

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          TrainingArguments, Trainer, EarlyStoppingCallback, BertTokenizerFast)
from peft import LoraConfig, get_peft_model

import stilt_smsa as base

# (lr, epochs, rank) — rank = lever gentleness UTAMA LoRA (kecil = backbone kurang berubah = ace lebih awet).
# alpha selalu = 2*r (scaling 2.0, sama champion). Sweep rank {8,16,32} @ lr2e-4 + cek lr1e-4.
GRID_CONFIGS = [(2e-4, 10, 8), (2e-4, 10, 16), (2e-4, 10, 32), (1e-4, 10, 16)]
GRID_SEED = [42, 0, 1]
PATIENCE = 3
OUT = "outputs/sv2-smsa-lora"
BASELINE_ACE_ZS = 64.04              # full-FT default
SWEETSPOT_GENTLE_ACE = 71.27         # puncak gentle-LR (5e-6) — LoRA harus lewatin ini biar bermakna


def train_lora(smsa, tok, lr, seed, out_dir, epochs, r):
    cfg = AutoConfig.from_pretrained(base.MODEL_CHECKPOINT)
    cfg.num_labels = 3; cfg.label2id = base.LABEL2ID
    cfg.id2label = {i: v for i, v in enumerate(base.LABEL_LIST)}
    ds_tr = base.make_ds(smsa["train_filtered"], tok)
    ds_va = base.make_ds(smsa["valid"], tok)
    collator = DataCollatorWithPadding(tokenizer=tok)
    n = cfg.num_hidden_layers

    def model_init():
        m = AutoModelForSequenceClassification.from_pretrained(base.MODEL_CHECKPOINT, config=cfg,
                                                               ignore_mismatched_sizes=True)
        for p in m.parameters():
            p.requires_grad = False                       # backbone BEKU
        lc = LoraConfig(r=r, lora_alpha=2 * r, target_modules=["query", "key", "value"],
                        lora_dropout=0.1, bias="none", layers_to_transform=list(range(n)),
                        layers_pattern="layer", init_lora_weights=True, modules_to_save=["classifier"])
        return get_peft_model(m, lc)                       # LoRA di QKV + classifier trainable

    args = TrainingArguments(
        output_dir=f"{out_dir}/run", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=16, per_device_eval_batch_size=64, optim="adamw_torch_fused",
        learning_rate=lr, weight_decay=0.01, num_train_epochs=epochs, warmup_ratio=0.1,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1", save_only_model=True,
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed)
    trainer = Trainer(model_init=model_init, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                      processing_class=tok, data_collator=collator, compute_metrics=base.compute_metrics,
                      callbacks=[EarlyStoppingCallback(PATIENCE)])
    trainer.train()
    val = float(trainer.evaluate(ds_va)["eval_f1"])

    # merge adapter ke backbone -> BertForSequenceClassification biasa (stage-1 load identik full-FT)
    merged = trainer.model.merge_and_unload()
    os.makedirs(f"{out_dir}/best", exist_ok=True)
    merged.save_pretrained(f"{out_dir}/best"); tok.save_pretrained(f"{out_dir}/best")
    d = f"{out_dir}/run"
    if os.path.isdir(d):
        for e in os.listdir(d):
            if e.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(d, e), ignore_errors=True)
    del trainer, merged; torch.cuda.empty_cache()
    return val


@torch.no_grad()
def zeroshot_merged(out_dir, tok):
    """Eval checkpoint merged (BertForSeqCls) zero-shot di 7 NusaX test."""
    from sklearn.metrics import f1_score
    import pandas as pd
    m = AutoModelForSequenceClassification.from_pretrained(f"{out_dir}/best").to(
        "cuda" if torch.cuda.is_available() else "cpu").eval()
    res = {}
    for lang in base.SCOPE:
        te = pd.read_csv(f"data/nusax_senti/{lang}/test.csv")
        l2i = {v: i for i, v in enumerate(sorted(te["label"].unique()))}
        y = te["label"].map(l2i).to_numpy()
        preds = []
        txt = te["text"].tolist()
        for i in range(0, len(txt), 64):
            b = tok(txt[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to(m.device)
            preds.append(m(**b).logits.argmax(-1).cpu().numpy())
        res[lang] = round(f1_score(y, np.concatenate(preds), average="macro") * 100, 2)
        print(f"  {lang}: zero-shot = {res[lang]:.2f}")
    del m; torch.cuda.empty_cache()
    return res


def main():
    print(f"CUDA={torch.cuda.is_available()} | LoRA stage-0 (backbone BEKU) configs={GRID_CONFIGS} × seed {GRID_SEED}")
    tok = BertTokenizerFast.from_pretrained(base.MODEL_CHECKPOINT)
    smsa = base.fetch_smsa()
    filtered, _ = base.leakage_filter(smsa["train"])
    smsa["train_filtered"] = filtered
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for lr, epochs, r in GRID_CONFIGS:
        for seed in GRID_SEED:
            tag = f"lora_lr{lr:g}_r{r}ep{epochs}_seed{seed}"
            out_dir = f"{OUT}/{tag}"
            sp = f"{out_dir}/zeroshot.json"
            if os.path.exists(sp):
                d = json.load(open(sp)); rows.append((tag, d["smsa_val"], d["zeroshot"]))
                print(f"--- {tag} SKIP (ada)"); continue
            print(f"\n--- {tag} (LoRA r{r}, lr={lr}, {epochs}ep, seed={seed})")
            val = train_lora(smsa, tok, lr, seed, out_dir, epochs, r)
            print(f"  SmSA val-F1 = {val*100:.2f} | zero-shot:")
            zs = zeroshot_merged(out_dir, tok)
            json.dump({"lr": lr, "epochs": epochs, "seed": seed, "smsa_val": val, "zeroshot": zs}, open(sp, "w"), indent=2)
            rows.append((tag, val, zs))

    print("\n" + "=" * 70)
    print(f"{'config':26} {'smsaV':>6} | {'ACE':>6} {'mean7':>6}")
    print(f"{'full-FT default 2e-5':26} {92.20:6.2f} | {BASELINE_ACE_ZS:6.2f} {76.86:6.2f}")
    print(f"{'full-FT gentle 5e-6 (puncak)':26} {91.40:6.2f} | {SWEETSPOT_GENTLE_ACE:6.2f} {79.72:6.2f}")
    aces = []
    for tag, val, zs in rows:
        m = np.mean([zs[l] for l in base.SCOPE]); aces.append(zs["ace"])
        fl = " *** LEWAT SWEET-SPOT" if zs["ace"] >= SWEETSPOT_GENTLE_ACE else ""
        print(f"{tag:26} {val*100:6.2f} | {zs['ace']:6.2f} {m:6.2f}{fl}")
    if aces:
        print(f"\nLoRA stage-0 ACE zero-shot mean = {np.mean(aces):.2f} (vs gentle-LR puncak 71.27, full-FT 64.04)")
        print("→ kalau > 71: LoRA-stage-0 preserve ace lebih baik, lanjut stage-1. Kalau <=71: SmSA tuntas 2 mekanisme.")


if __name__ == "__main__":
    main()
