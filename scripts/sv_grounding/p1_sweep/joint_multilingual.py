"""Joint multilingual fine-tune: SATU model dilatih atas POOLED train 12 bahasa (~6000), eval PER-BAHASA.

Hipotesis (frontier-shifter, BUKAN knob): per-lang model cuma liat 500 contoh -> large OVERFIT (base-flip).
Pooled 12 bahasa = ~6000 + korpus PARALEL (transfer lintas-bahasa cue-negatif) -> low-resource ban/bjn
piggyback ke high-resource. Resep = CHAMPION VANILLA persis (LP-FT + LoRA), biar variabel cuma joint-vs-per-lang.

Eval per-bahasa (test 400/bahasa terpisah) -> bandingin joint-on-X vs champion-per-lang-on-X. Fallback =
champion per-lang (bobot udah ada) kalau joint regresi pemenang. Screen 1 seed dulu; kalau ban/bjn naik -> multi-seed.

Jalankan dari root:  uv run python scripts/sv_grounding/p1_sweep/joint_multilingual.py
"""
import os
import sys
import json
import gc
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset, Features, Value, ClassLabel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, BertTokenizerFast, BertModel,
    DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import champion_p1_base as base   # reuse konstanta champion (resep identik)

LANGS = base.TARGET_LANGS                       # 12 bahasa
SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
LABEL_LIST = ["negative", "neutral", "positive"]
LABEL2ID = {v: i for i, v in enumerate(LABEL_LIST)}
NUM_LABELS = 3
CKPT = base.MODEL_CHECKPOINT
MAXLEN = base.INPUT_MAX_LENGTH

SEEDS = json.loads(os.environ.get("P1_SEEDS", "[42]"))
OUTPUT_ROOT = os.environ.get("P1_OUTPUT_ROOT", "outputs/joint-multilingual-champion")
KEEP_WEIGHTS = os.environ.get("P1_KEEP_WEIGHTS", "deploy_only")
DEPLOY_SEED = 42

f1m = evaluate.load("f1")
accm = evaluate.load("accuracy")


def compute_metrics(ep):
    pred, lab = ep
    pred = np.argmax(pred, axis=1)
    return {"f1": round(f1m.compute(predictions=pred, references=lab, average="macro")["f1"], 4),
            "accuracy": round(accm.compute(predictions=pred, references=lab)["accuracy"], 4)}


def make_ds(df, tok):
    feats = Features({"text": Value("string"), "label": ClassLabel(names=LABEL_LIST)})
    d = Dataset.from_pandas(df[["text", "label"]], features=feats, preserve_index=False)
    return d.map(lambda e: tok(e["text"], max_length=MAXLEN, truncation=True), batched=True)


def load_data(tok):
    tr, va, train_by, test_by = [], [], {}, {}
    for lang in LANGS:
        dd = f"data/nusax_senti/{lang}"
        a = pd.read_csv(f"{dd}/train.csv"); b = pd.read_csv(f"{dd}/valid.csv"); c = pd.read_csv(f"{dd}/test.csv")
        for df in (a, b, c):
            df["label"] = df["label"].map(LABEL2ID)
        tr.append(a); va.append(b)
        train_by[lang] = make_ds(a, tok); test_by[lang] = make_ds(c, tok)
    pool_tr = make_ds(pd.concat(tr, ignore_index=True), tok)
    pool_va = make_ds(pd.concat(va, ignore_index=True), tok)
    return pool_tr, pool_va, train_by, test_by


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


def run_seed(seed):
    out_dir = f"{OUTPUT_ROOT}/seed_{seed}"
    summ = f"{out_dir}/results_summary.json"
    if os.path.exists(summ) and len(json.load(open(summ)).get("results", {})) >= len(LANGS):
        print(f"seed {seed} SKIP (lengkap)"); return
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== JOINT [seed {seed}] {len(LANGS)} bahasa pooled (resep champion vanilla) ===")

    tok = BertTokenizerFast.from_pretrained(CKPT)
    cfg = AutoConfig.from_pretrained(CKPT)
    cfg.num_labels = NUM_LABELS; cfg.label2id = LABEL2ID; cfg.id2label = {i: v for i, v in enumerate(LABEL_LIST)}
    cfg.hidden_dropout_prob = base.HIDDEN_DROPOUT; cfg.attention_probs_dropout_prob = base.HIDDEN_DROPOUT

    pool_tr, pool_va, train_by, test_by = load_data(tok)
    collator = DataCollatorWithPadding(tokenizer=tok)

    raw = AutoModelForSequenceClassification.from_pretrained(CKPT, config=cfg, ignore_mismatched_sizes=True)
    fixed = OrderedDict((k.replace(".gamma", ".weight").replace(".beta", ".bias"), v.clone()) for k, v in raw.state_dict().items())
    del raw

    # Stage 1: linear probe (backbone beku)
    def init_lp():
        m = MeanPoolClf(cfg); m.load_state_dict(fixed, strict=False)
        for p in m.bert.parameters(): p.requires_grad = False
        return m
    # LP = warmup head; save_strategy="no" (pakai epoch terakhir) -> NGGAK serialize tiap epoch (fix MemoryError host-RAM)
    lp_args = TrainingArguments(
        output_dir=f"{out_dir}/lp", eval_strategy="epoch", save_strategy="no", logging_strategy="epoch",
        per_device_train_batch_size=base.TRAIN_BATCH_SIZE, per_device_eval_batch_size=base.EVAL_BATCH_SIZE,
        optim="adamw_torch_fused", learning_rate=base.LP_LEARNING_RATE, weight_decay=base.WEIGHT_DECAY,
        num_train_epochs=base.LP_EPOCHS, load_best_model_at_end=False,
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed)
    lp = Trainer(model_init=init_lp, args=lp_args, train_dataset=pool_tr, eval_dataset=pool_va,
                 processing_class=tok, data_collator=collator, compute_metrics=compute_metrics)
    lp.train()
    lp_state = {k: v.clone() for k, v in lp.model.state_dict().items()}
    del lp, init_lp, fixed; gc.collect(); torch.cuda.empty_cache()   # bebasin ~1.3GB CPU (fixed gak kepake lagi)

    # Stage 2: LoRA FT (champion vanilla)
    n = cfg.num_hidden_layers
    lora_layers = list(range(n)) if base.LORA_ALL_LAYERS else list(range(n - 4, n))
    def init_ft():
        m = MeanPoolClf(cfg); m.load_state_dict(lp_state)
        for p in m.parameters(): p.requires_grad = False
        lc = LoraConfig(r=base.LORA_R, lora_alpha=base.LORA_ALPHA, target_modules=base.LORA_TARGET_MODULES,
                        lora_dropout=base.LORA_DROPOUT, bias="none", layers_to_transform=lora_layers,
                        layers_pattern="layer", init_lora_weights=True)
        m.bert = get_peft_model(m.bert, lc)
        for p in m.classifier.parameters(): p.requires_grad = True
        return m
    ft_args = TrainingArguments(
        output_dir=f"{out_dir}/ft", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=base.TRAIN_BATCH_SIZE, per_device_eval_batch_size=base.EVAL_BATCH_SIZE,
        optim="adamw_torch_fused", warmup_ratio=0.1, learning_rate=base.FT_LEARNING_RATE, weight_decay=base.WEIGHT_DECAY,
        num_train_epochs=base.FT_EPOCHS, save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1",
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed)
    # build model manual (set_seed dulu) -> bisa del lp_state SEBELUM train/save -> bebasin ~1.3GB CPU buat serialize FT
    set_seed(seed)
    m = init_ft()
    del init_ft, lp_state; gc.collect()
    trainer = Trainer(model=m, args=ft_args, train_dataset=pool_tr, eval_dataset=pool_va,
                      processing_class=tok, data_collator=collator, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(base.EARLY_STOPPING_PATIENCE)])
    trainer.train()
    trainer.remove_callback(EarlyStoppingCallback)

    # eval PER-BAHASA (test + train per-lang buat gap)
    results = {}
    for lang in LANGS:
        tr_res = trainer.evaluate(train_by[lang], metric_key_prefix="train")
        tp = trainer.predict(test_by[lang])
        yt, yp = tp.label_ids, np.argmax(tp.predictions, axis=1)
        te_f1 = float(f1m.compute(predictions=yp, references=yt, average="macro")["f1"])
        te_acc = float(accm.compute(predictions=yp, references=yt)["accuracy"])
        p, r, f, sup = precision_recall_fscore_support(yt, yp, labels=list(range(NUM_LABELS)), zero_division=0)
        per_class = {LABEL_LIST[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i in range(NUM_LABELS)}
        np.save(f"{out_dir}/{lang}_test_logits.npy", tp.predictions)
        np.save(f"{out_dir}/{lang}_test_labels.npy", yt)
        gap = tr_res["train_f1"] - te_f1
        results[lang] = {"train_f1": tr_res["train_f1"], "test_f1": round(te_f1, 4), "test_accuracy": round(te_acc, 4),
                         "gap": gap, "per_class_test": per_class, "confusion_matrix_test": confusion_matrix(yt, yp, labels=list(range(NUM_LABELS))).tolist(), "label_order": LABEL_LIST}
        print(f"[{lang}] train={tr_res['train_f1']*100:.2f} test={te_f1*100:.2f} gap={gap*100:.2f}")

    if (KEEP_WEIGHTS == "all") or (KEEP_WEIGHTS == "deploy_only" and seed == DEPLOY_SEED):
        trainer.save_model(f"{out_dir}/best"); tok.save_pretrained(f"{out_dir}/best")
    for sub in ("ft", "lp"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            for e in os.listdir(d):
                if e.startswith("checkpoint-"):
                    shutil.rmtree(os.path.join(d, e), ignore_errors=True)

    scope_mean = float(np.mean([results[l]["test_f1"] * 100 for l in SCOPE]))
    json.dump({"seed": seed, "scope_mean_test_f1": scope_mean, "results": results}, open(summ, "w"), indent=2)
    print(f"[seed {seed}] scope-mean test-F1 = {scope_mean:.2f}")
    del trainer; gc.collect(); torch.cuda.empty_cache()


def main():
    print(f"JOINT multilingual | {OUTPUT_ROOT} | seeds {SEEDS} | KEEP={KEEP_WEIGHTS}")
    for s in SEEDS:
        run_seed(s)
    print("\nJOINT SELESAI. Bandingin: uv run python scripts/sv_grounding/p1_sweep/joint_aggregate.py")


if __name__ == "__main__":
    main()
