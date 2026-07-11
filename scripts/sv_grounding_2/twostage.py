"""E5 — Two-stage: joint multilingual (stage-1, checkpoint ADA) -> refinement per-bahasa singkat (stage-2).

IDE: joint pooled (F-P1e) meruntuhkan gap ke 3.66 TAPI redistribusi F1 (bjn/jav/ace turun). Two-stage =
pakai joint sebagai INIT bersama, lalu poles tiap bahasa singkat (LR kecil, epoch sedikit) biar idiosinkrasi
bahasa balik TANPA sempat buka gap. STRUKTURAL kebal redistribusi: stage-2 tiap bahasa nyentuh model sendiri
-> naikin bjn TAK BISA jatuhin ace (beda dari DoRA/joint yg zero-sum). Start semua gap<5 -> ruang ~5pt.

Stage-1 = checkpoint joint seed-42 (`outputs/joint-multilingual-champion/seed_42/best/`, MeanPoolClf + LoRA
r16/a32 vanilla, unmerged) -> di-merge SEKALI jadi init bersama.
Stage-2 = 4 SEL (pre-registered, JANGAN nambah): {LoRA-baru r8/a16 lr1e-5, full-FT lr5e-6} x {data target-500,
target+ind-500}. max 5 epoch, patience 2, bs8. val & seleksi checkpoint = TARGET-only. gap = train TARGET-only.

SCREENING (default): seed 42 x 4 bahasa kontes (ace/ban/bjn/mad) x 4 sel. Confirm -> 7 x 5 seed sel pemenang.
Env: TS_LANGS, TS_SEEDS, TS_CELLS (json list of "arm:data"), TS_KEEP (none|all), TS_OUT.

Jalankan dari root:  uv run python scripts/sv_grounding_2/twostage.py
"""
import os
import sys
import gc
import json
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset, Features, Value, ClassLabel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoConfig, BertTokenizerFast, BertModel, DataCollatorWithPadding,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import safetensors.torch as st
from peft import LoraConfig, get_peft_model

# champion_p1_base ada di scripts/sv_grounding/p1_sweep/ (lokasi iter-1 setelah reorg). Import resilient (cek kandidat).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in [os.path.join(_HERE, "..", "sv_grounding", "p1_sweep"), os.path.join(_HERE, "..", "p1_sweep")]:
    if os.path.exists(os.path.join(_cand, "champion_p1_base.py")):
        sys.path.insert(0, _cand)
        break
import champion_p1_base as base   # reuse konstanta champion

CKPT = base.MODEL_CHECKPOINT
MAXLEN = base.INPUT_MAX_LENGTH
LABEL_LIST = ["negative", "neutral", "positive"]
LABEL2ID = {v: i for i, v in enumerate(LABEL_LIST)}
NUM_LABELS = 3
# R3 stage-2 FINAL (2026-07-07): gate joint_select PASS -> joint seed 3 (maximin val 84.47, val-ace 83.56
# vs seed42 79.01, Δ+4.5). Riwayat: E5/R2 jalan di seed_42 (outputs/sv2-twostage). JANGAN campur output.
JOINT_CKPT = "outputs/joint-multilingual-champion/seed_3/best/model.safetensors"

# R2 INTENSITY LADDER (2026-07-07, PRE-REGISTERED, run TERAKHIR two-stage — jangan nambah sel setelah lihat test):
# Confirm 5-seed nunjuk stage-2 default (lr1e-5/5ep/r8) TERLALU LEMBUT: ace 79.31 & bjn 83.22 DI BAWAH level
# per-lang mereka sendiri (drop06: ace 80.56, bjn 84.85) padahal gap 3.88 punya headroom ~6pt ke batas 10.
# -> perkuat stage-2 GLOBAL (sama utk 7 bahasa) di 4 titik intensitas; target: ace>=80 & semua>=80 & gap<10,
# sadar trade-off: sun/min (di atas per-lang krn joint) bakal turun dikit ke arah per-lang -> margin aman.
# Keputusan (amandemen pre-run 2026-07-07): (1) gate keras semua>=80 & gap<10; (2) ranking: prioritas sel dgn
# worst-case Delta-vs-FT >= -1.0 per bahasa (BUKAN gate keras -- drop06 sendiri gagal ini: ace -1.29/bjn -1.93);
# (3) tie-break mean-F1. JANGAN iterasi HP lagi setelah ini. FT ref: ace81.85 ban81.96 bjn86.78 jav87.04 mad80.31 min83.29 sun85.99.
TARGET_LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
SEEDS = [42, 0, 1, 2, 3]
# R3 stage-2 final: SATU arm pre-registered (lora_lr2e5ep10 = terbaik R2: mean 83.71/gap 3.84) di joint seed-3.
# Output root BARU biar tak campur hasil R2 (joint seed-42) di outputs/sv2-twostage.
CELLS = ["lora_lr2e5ep10:target"]
KEEP_WEIGHTS = "all"   # calon deploy kalau lolos 7/7 -> simpan bobot. NOTE: full state dict ~1.35GB x 35 ≈ 47GB
                       # (disk free 183GB, muat). Simpan > retrain (rerun same-config bisa geser angka, preseden drop06).
OUTPUT_ROOT = "outputs/sv2-twostage-j3"

# stage-2 HP default (arm "lora" & "fullft" lama — JANGAN diubah, kunci konsistensi resume/riwayat)
S2_LORA_R = 8
S2_LORA_ALPHA = 16
S2_LORA_DROPOUT = 0.1
S2_LORA_LR = 1e-5
S2_FULLFT_LR = 5e-6
S2_MAX_EPOCHS = 5
S2_PATIENCE = 2

# Ladder intensitas: nama arm -> override HP stage-2 (kind lora semua; yang tak disebut = default di atas)
ARMS = {
    "lora":           {},                                              # baseline confirm (sudah jalan)
    "lora_lr2e5":     {"lr": 2e-5},                                    # I1: 2x LR
    "lora_lr3e5":     {"lr": 3e-5},                                    # I2: 3x LR (titik terkuat ladder LR)
    "lora_lr2e5ep10": {"lr": 2e-5, "epochs": 10, "patience": 3},       # I3: 2x LR + ruang epoch
    "lora_r16lr2e5":  {"lr": 2e-5, "r": 16, "alpha": 32},              # I4: 2x LR + kapasitas 2x
}

f1m = evaluate.load("f1")
accm = evaluate.load("accuracy")


def compute_metrics(ep):
    pred, lab = ep
    pred = np.argmax(pred, axis=1)
    return {"f1": round(f1m.compute(predictions=pred, references=lab, average="macro")["f1"], 4),
            "accuracy": round(accm.compute(predictions=pred, references=lab)["accuracy"], 4)}


class MeanPoolClf(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.classifier = torch.nn.Linear(cfg.hidden_size, NUM_LABELS)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kw):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        logits = self.classifier(pooled)
        loss = torch.nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_joint_init(cfg):
    """Load checkpoint joint (LoRA r16/a32 unmerged) -> merge -> state dict MeanPoolClf init bersama (CPU)."""
    m = MeanPoolClf(cfg)
    n = cfg.num_hidden_layers
    lora_layers = list(range(n)) if base.LORA_ALL_LAYERS else list(range(n - 4, n))
    lc = LoraConfig(r=base.LORA_R, lora_alpha=base.LORA_ALPHA, target_modules=base.LORA_TARGET_MODULES,
                    lora_dropout=base.LORA_DROPOUT, bias="none", layers_to_transform=lora_layers,
                    layers_pattern="layer", init_lora_weights=True)
    m.bert = get_peft_model(m.bert, lc)
    sd = st.load_file(JOINT_CKPT)
    m.load_state_dict(sd, strict=True)
    m.bert = m.bert.merge_and_unload()      # -> BertModel plain (backbone joint ter-merge)
    init_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del m, sd
    gc.collect()
    return init_state


def make_ds(df, tok):
    feats = Features({"text": Value("string"), "label": ClassLabel(names=LABEL_LIST)})
    d = Dataset.from_pandas(df[["text", "label"]], features=feats, preserve_index=False)
    return d.map(lambda e: tok(e["text"], max_length=MAXLEN, truncation=True), batched=True)


def load_lang(lang, data_mode, tok):
    dd = f"data/nusax_senti/{lang}"
    tr = pd.read_csv(f"{dd}/train.csv"); va = pd.read_csv(f"{dd}/valid.csv"); te = pd.read_csv(f"{dd}/test.csv")
    for df in (tr, va, te):
        df["label"] = df["label"].map(LABEL2ID)
    train_target = tr.copy()                                  # 500 target (buat gap target-only)
    if data_mode == "target_ind" and lang != "ind":
        ind = pd.read_csv("data/nusax_senti/ind/train.csv"); ind["label"] = ind["label"].map(LABEL2ID)
        tr = pd.concat([tr, ind], ignore_index=True)          # +500 ind
    return (make_ds(tr, tok), make_ds(va, tok), make_ds(te, tok), make_ds(train_target, tok))


def run_cell(lang, seed, arm, data_mode, init_state, cfg, tok, collator, out_dir):
    tag = f"{arm}_{data_mode}"
    output_dir = f"{out_dir}/{tag}/nusabert-large-{lang}"
    print(f"\n>>> TWOSTAGE [seed {seed}] [{lang}] arm={arm} data={data_mode}")
    ds_tr, ds_va, ds_te, ds_tr_target = load_lang(lang, data_mode, tok)

    n = cfg.num_hidden_layers
    lora_layers = list(range(n)) if base.LORA_ALL_LAYERS else list(range(n - 4, n))

    # HP per-arm: default S2_* + override dari ARMS (arm ladder). "fullft" = kind khusus (semua param trainable).
    ov = ARMS.get(arm, {})
    hp = {"r": ov.get("r", S2_LORA_R), "alpha": ov.get("alpha", S2_LORA_ALPHA),
          "dropout": ov.get("dropout", S2_LORA_DROPOUT),
          "lr": ov.get("lr", S2_FULLFT_LR if arm == "fullft" else S2_LORA_LR),
          "epochs": ov.get("epochs", S2_MAX_EPOCHS), "patience": ov.get("patience", S2_PATIENCE)}
    print(f"    hp: {hp}")

    def init_model():
        m = MeanPoolClf(cfg)
        m.load_state_dict(init_state, strict=True)            # warm start dari joint-merged
        if arm != "fullft":
            for p in m.parameters(): p.requires_grad = False
            lc = LoraConfig(r=hp["r"], lora_alpha=hp["alpha"], target_modules=base.LORA_TARGET_MODULES,
                            lora_dropout=hp["dropout"], bias="none", layers_to_transform=lora_layers,
                            layers_pattern="layer", init_lora_weights=True)
            m.bert = get_peft_model(m.bert, lc)
            for p in m.classifier.parameters(): p.requires_grad = True
        # arm fullft: semua param trainable (default)
        return m

    ft_args = TrainingArguments(
        output_dir=f"{output_dir}/ft", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=base.TRAIN_BATCH_SIZE, per_device_eval_batch_size=base.EVAL_BATCH_SIZE,
        optim="adamw_torch_fused", warmup_ratio=0.1, learning_rate=hp["lr"], weight_decay=base.WEIGHT_DECAY,
        num_train_epochs=hp["epochs"], save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1",
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed)
    set_seed(seed)
    trainer = Trainer(model_init=init_model, args=ft_args, train_dataset=ds_tr, eval_dataset=ds_va,
                      processing_class=tok, data_collator=collator, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(hp["patience"])])
    trainer.train()
    trainer.remove_callback(EarlyStoppingCallback)

    tr_res = trainer.evaluate(ds_tr_target, metric_key_prefix="train")    # gap TARGET-only
    va_res = trainer.evaluate(ds_va, metric_key_prefix="validation")
    te_res = trainer.evaluate(ds_te)
    tp = trainer.predict(ds_te); vp = trainer.predict(ds_va)
    yt, yp = tp.label_ids, np.argmax(tp.predictions, axis=1)
    p, r, f, sup = precision_recall_fscore_support(yt, yp, labels=list(range(NUM_LABELS)), zero_division=0)
    per_class = {LABEL_LIST[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i in range(NUM_LABELS)}

    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/test_logits.npy", tp.predictions); np.save(f"{output_dir}/test_labels.npy", yt)
    np.save(f"{output_dir}/val_logits.npy", vp.predictions); np.save(f"{output_dir}/val_labels.npy", vp.label_ids)
    gap = tr_res["train_f1"] - te_res["eval_f1"]
    print(f"[{lang}][{tag}] train={tr_res['train_f1']*100:.2f} val={va_res['validation_f1']*100:.2f} "
          f"test={te_res['eval_f1']*100:.2f} gap={gap*100:.2f} negR={per_class['negative']['recall']*100:.1f}")

    if KEEP_WEIGHTS == "all":
        trainer.save_model(f"{output_dir}/best"); tok.save_pretrained(f"{output_dir}/best")
    d = os.path.join(output_dir, "ft")
    if os.path.isdir(d):
        for e in os.listdir(d):
            if e.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(d, e), ignore_errors=True)
    del trainer; gc.collect(); torch.cuda.empty_cache()
    return {"train_f1": tr_res["train_f1"], "val_f1": va_res["validation_f1"], "test_f1": te_res["eval_f1"],
            "gap": gap, "per_class_test": per_class, "label_order": LABEL_LIST}


def main():
    tok = BertTokenizerFast.from_pretrained(CKPT)
    cfg = AutoConfig.from_pretrained(CKPT)
    cfg.num_labels = NUM_LABELS; cfg.label2id = LABEL2ID; cfg.id2label = {i: v for i, v in enumerate(LABEL_LIST)}
    cfg.hidden_dropout_prob = base.HIDDEN_DROPOUT; cfg.attention_probs_dropout_prob = base.HIDDEN_DROPOUT
    collator = DataCollatorWithPadding(tokenizer=tok)
    print(f"CUDA={torch.cuda.is_available()} | merge joint init dari {JOINT_CKPT} ...")
    init_state = build_joint_init(cfg)
    print(f"joint init siap ({len(init_state)} tensor) | langs={TARGET_LANGS} seeds={SEEDS} cells={CELLS} keep={KEEP_WEIGHTS}")

    for seed in SEEDS:
        out_dir = f"{OUTPUT_ROOT}/seed_{seed}"
        os.makedirs(out_dir, exist_ok=True)
        sp = f"{out_dir}/results_summary.json"
        results = json.load(open(sp)) if os.path.exists(sp) else {}
        for cell in CELLS:
            arm, data_mode = cell.split(":")
            for lang in TARGET_LANGS:
                key = f"{arm}_{data_mode}/{lang}"
                if key in results:
                    print(f"[{key}] SKIP"); continue
                results[key] = run_cell(lang, seed, arm, data_mode, init_state, cfg, tok, collator, out_dir)
                json.dump(results, open(sp, "w"), indent=2)
        print(f"[seed {seed}] -> {sp}")
    print("TWOSTAGE SELESAI. Agg per sel: baca results_summary.json (test/gap per bahasa per sel).")


def _apply_env():
    for k, ek, fn in [("TARGET_LANGS", "TS_LANGS", json.loads), ("SEEDS", "TS_SEEDS", json.loads),
                      ("CELLS", "TS_CELLS", json.loads), ("KEEP_WEIGHTS", "TS_KEEP", str), ("OUTPUT_ROOT", "TS_OUT", str)]:
        if ek in os.environ:
            globals()[k] = fn(os.environ[ek])


if __name__ == "__main__":
    _apply_env()
    main()
