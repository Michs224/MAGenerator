"""P0 — FT (paper, full fine-tune) MULTI-SEED baseline untuk perbandingan adil.

Recipe = NusaBERT-large full-FT sesuai paper (LR 2e-5, wd 0.01, bs 16, dropout 0.1,
patience 5, 100 ep early-stop) — IDENTIK dengan run `outputs/nusabert-sentiment-large`
(itu = seed 42, di-REUSE; jangan dilatih ulang). Script ini melatih seed LAINNYA.

Tujuan: dapat FT mean+-std per bahasa biar pembanding ke champion (LP-FT+LoRA) ADIL
(jumlah seed sama). Single-seed FT bjn=86.76 itu draw upper-tail; mean lebih jujur.

Cleanup FT (sesuai permintaan): tiap selesai 1 bahasa -> hapus checkpoint-* DAN bobot
model (model.safetensors), SISAIN config.json + tokenizer + training_args.bin +
train_history.json + results_summary.json (persis struktur folder nusabert-sentiment-large).
FT = referensi, nggak di-deploy, jadi bobot nggak perlu disimpan.

Jalankan dari root:  uv run python scripts/p0_multiseed/p0_ft_multiseed.py
"""
import os
import gc
import json
import shutil

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoConfig,
    BertTokenizerFast,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ---- Config (paper full-FT, IDENTIK nusabert-sentiment-large) ----
MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
TARGET_LANGS = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]  # 12 langs
SEEDS = [0, 1, 2, 3]              # seed 42 = nusabert-sentiment-large (di-reuse, tidak dilatih ulang)
INPUT_MAX_LENGTH = 128
NUM_TRAIN_EPOCHS = 100
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5
OUTPUT_ROOT = "outputs/p0-ft-multiseed"   # per seed -> {OUTPUT_ROOT}/seed_{SEED}


def finetune(lang_code: str, seed: int, output_base_dir: str):
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    output_dir = f"{output_base_dir}/{model_name}-{lang_code}"

    # Skip kalau sudah ada summary per-bahasa (resumable)
    done_marker = f"{output_dir}/train_history.json"
    if os.path.exists(done_marker):
        print(f"[seed {seed}][{lang_code}] SKIP (sudah ada {done_marker})")
        return None

    print(f"\n>>> FT [seed {seed}] {model_name} [{lang_code}]")

    data_dir = f"data/nusax_senti/{lang_code}"
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    valid_df = pd.read_csv(f"{data_dir}/valid.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    label_list = sorted(train_df["label"].unique().tolist())
    label2id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}
    num_labels = len(label_list)

    for df in [train_df, valid_df, test_df]:
        df["label"] = df["label"].map(label2id)

    features = Features({"id": Value("int64"), "text": Value("string"), "label": ClassLabel(names=label_list)})
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, features=features, preserve_index=False),
        "validation": Dataset.from_pandas(valid_df, features=features, preserve_index=False),
        "test": Dataset.from_pandas(test_df, features=features, preserve_index=False),
    })
    print(f"Train: {len(dataset['train'])}, Valid: {len(dataset['validation'])}, Test: {len(dataset['test'])} | labels={label_list}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label

    # Fix LayerNorm gamma/beta -> weight/bias (NusaBERT naming)
    from transformers import AutoModelForSequenceClassification as AutoMSC
    from collections import OrderedDict
    raw_model = AutoMSC.from_pretrained(MODEL_CHECKPOINT, config=config, ignore_mismatched_sizes=True)
    new_state_dict = OrderedDict()
    for key, value in raw_model.state_dict().items():
        new_state_dict[key.replace(".gamma", ".weight").replace(".beta", ".bias")] = value
    del raw_model

    def preprocess_function(examples):
        return tokenizer(examples["text"], max_length=INPUT_MAX_LENGTH, truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        return {"f1": round(f1["f1"], 4), "accuracy": round(acc["accuracy"], 4)}

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    _fixed_state_dict = {k: v.clone() for k, v in new_state_dict.items()}

    def model_init():
        m = AutoMSC.from_config(config)
        m.load_state_dict(_fixed_state_dict, strict=False)
        return m

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # Early-stop diagnostics (sebelum evaluate(test) mengontaminasi log_history)
    _evals = [e for e in trainer.state.log_history if "eval_f1" in e]
    _best = max(_evals, key=lambda e: e["eval_f1"]) if _evals else {}
    stop_epoch = round(_evals[-1]["epoch"]) if _evals else None
    best_epoch = round(_best["epoch"]) if _evals else None

    trainer.remove_callback(EarlyStoppingCallback)
    train_results = trainer.evaluate(tokenized_dataset["train"], metric_key_prefix="train")
    val_results = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="validation")
    train_history = trainer.state.log_history
    test_results = trainer.evaluate(tokenized_dataset["test"])

    # Per-class P/R/F1 + confusion matrix di TEST (diagnostik; macro tetap metrik utama)
    test_pred = trainer.predict(tokenized_dataset["test"])
    y_true = test_pred.label_ids
    y_pred = np.argmax(test_pred.predictions, axis=1)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_labels)), zero_division=0)
    per_class = {label_list[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i in range(num_labels)}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels))).tolist()

    gap = train_results["train_f1"] - test_results["eval_f1"]
    print(f"[seed {seed}][{lang_code}] train={train_results['train_f1']*100:.2f} val={val_results['validation_f1']*100:.2f} "
          f"test={test_results['eval_f1']*100:.2f} gap={gap*100:.2f} | stop@ep{stop_epoch} best@ep{best_epoch}")

    # Simpan best/ (config+tokenizer+training_args) lalu HAPUS bobot model (FT = referensi)
    best_dir = f"{output_dir}/best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    for wf in ("model.safetensors", "pytorch_model.bin"):
        wp = os.path.join(best_dir, wf)
        if os.path.exists(wp):
            os.remove(wp)
            print(f"  [cleanup FT] deleted weight {wf}")

    with open(f"{output_dir}/train_history.json", "w") as fjson:
        json.dump(train_history, fjson, indent=2)

    # Cleanup checkpoint-* dirs
    for entry in os.listdir(output_dir):
        if entry.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, entry)):
            shutil.rmtree(os.path.join(output_dir, entry))

    del trainer, tokenized_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "train_f1": train_results["train_f1"], "train_accuracy": train_results["train_accuracy"],
        "val_f1": val_results["validation_f1"], "val_accuracy": val_results["validation_accuracy"],
        "test_f1": test_results["eval_f1"], "test_accuracy": test_results["eval_accuracy"],
        "gap": gap, "stop_epoch": stop_epoch, "best_epoch": best_epoch,
        "per_class_test": per_class, "confusion_matrix_test": cm, "label_order": label_list,
    }


def run_seed(seed: int):
    output_base_dir = f"{OUTPUT_ROOT}/seed_{seed}"
    os.makedirs(output_base_dir, exist_ok=True)
    summary_path = f"{output_base_dir}/results_summary.json"
    results = json.load(open(summary_path))["results"] if os.path.exists(summary_path) else {}

    for lang in TARGET_LANGS:
        res = finetune(lang, seed, output_base_dir)
        if res is not None:
            results[lang] = res
            # tulis incremental biar aman kalau ke-interrupt
            _write_summary(summary_path, seed, results)

    _write_summary(summary_path, seed, results)
    print(f"\n[seed {seed}] summary -> {summary_path}")


def _write_summary(summary_path, seed, results):
    summary = {
        "hyperparams": {
            "method": "FT-paper-full-finetune",
            "model_checkpoint": MODEL_CHECKPOINT, "target_langs": TARGET_LANGS,
            "input_max_length": INPUT_MAX_LENGTH, "num_train_epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "train_batch_size": TRAIN_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE, "hidden_dropout_prob": 0.1,
            "seed": seed,
        },
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"P0 FT multi-seed | seeds={SEEDS} (42 di-reuse dari nusabert-sentiment-large) | langs={len(TARGET_LANGS)}")
    for seed in SEEDS:
        run_seed(seed)
    print("\nSELESAI. Lanjut: champion multi-seed, lalu p0_aggregate.py")
