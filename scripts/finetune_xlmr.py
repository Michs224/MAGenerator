"""Fine-tune XLM-RoBERTa (large/base) on NusaX-Senti untuk downstream evaluation.

Hyperparameter = recipe NusaX (Winata et al., 2023) Tabel 10 "Hyperparameters of
pre-trained LMs on sentiment analysis". Lihat agent_doc/thesis_writing_notes.md Bab III.B.
  - learning rate 1e-5, batch 32      → grid-search best (bold) NusaX Tabel 10
  - num epochs 100, early stop 3       → fixed (semua pre-trained LM)
  - max norm 10                        → fixed NusaX Tabel 10
  - weight decay 0                     → NusaX Tabel 10 TAK punya baris wd → Adam tanpa wd
  - optimizer AdamW (wd=0 ≡ Adam)      → NusaX pakai "Adam"; AdamW wd=0 identik
  - LR scheduler linear                → NusaX γ=0.9 undefined di paper → pakai default HF
  - max seq length 128                 → config (kalimat NusaX-Senti pendek; plafon XLM-R 512)
  - seed 42

Cara pakai: flip MODEL_SIZE ("large"/"base") dan USE_AUGMENTED (False=baseline / True=augmented).
4 konfigurasi total: {large,base} x {baseline,augmented}.

Baseline (USE_AUGMENTED=False) dipakai untuk sanity-check vs NusaX Tabel 2.
Augmented (USE_AUGMENTED=True) baca data/nusax_senti/<lang>/syn/train_syn.csv.
"""
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ─── Toggles ──────────────────────────────────────────────────────────────────
MODEL_SIZE = "large"        # "large" atau "base" — flip untuk jalankan yang lain
USE_AUGMENTED = False        # False = baseline (train.csv); True = augmented (syn/train_syn.csv)

# ─── Hyperparams (NusaX Tabel 10, identik untuk base & large) ────────────────
MODEL_CHECKPOINT = f"FacebookAI/xlm-roberta-{MODEL_SIZE}"

# Auto-pilih bahasa berdasarkan skenario (pola sama NusaBERT):
#   baseline  → 12 bahasa (reproduksi penuh NusaX Tabel 2 + sanity-check)
#   augmented → 7 bahasa target (hanya ini yang punya data sintetis)
# Display akhir tetap 7 bahasa (intersection) untuk Δ-F1.
LANGS_12 = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]
LANGS_7 = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
TARGET_LANGS = LANGS_7 if USE_AUGMENTED else LANGS_12

INPUT_MAX_LENGTH = 128
NUM_TRAIN_EPOCHS = 100
LEARNING_RATE = 1e-5            # NusaX Tabel 10 best (bold)
WEIGHT_DECAY = 0.0             # NusaX Tabel 10 tak punya wd → 0 (AdamW wd=0 ≡ Adam)
TRAIN_BATCH_SIZE = 32          # NusaX Tabel 10 best (bold) — effective batch
PER_DEVICE_BATCH = 16         # turunkan kalau OOM (mis. 8); grad-accum jaga effective = 32
GRAD_ACCUM = max(1, TRAIN_BATCH_SIZE // PER_DEVICE_BATCH)
EVAL_BATCH_SIZE = 64
MAX_GRAD_NORM = 10.0          # NusaX Tabel 10 (max norm 10)
EARLY_STOPPING_PATIENCE = 3   # NusaX Tabel 10
SEED = 42

_suffix = "-syn" if USE_AUGMENTED else ""
OUTPUT_BASE_DIR = f"outputs/xlmr-sentiment-{MODEL_SIZE}{_suffix}"

# NusaX Tabel 2 (Winata 2023) — referensi baseline untuk sanity-check (macro-F1 %)
NUSAX_TABLE2 = {
    "large": {"ace": 75.9, "ban": 77.1, "bbc": 65.5, "bjn": 86.3, "bug": 70.0,
              "eng": 92.6, "ind": 91.6, "jav": 84.2, "mad": 74.9, "min": 83.1,
              "nij": 73.3, "sun": 86.0},
    "base":  {"ace": 73.9, "ban": 72.8, "bbc": 62.3, "bjn": 76.6, "bug": 66.6,
              "eng": 90.8, "ind": 88.4, "jav": 78.9, "mad": 69.7, "min": 79.1,
              "nij": 75.0, "sun": 80.1},
}


def finetune(lang_code: str):
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()   # xlm-roberta-large / -base
    print(f"\n{'='*60}")
    print(f"Fine-tuning {model_name} on NusaX-Senti [{lang_code}] "
          f"({'AUGMENTED' if USE_AUGMENTED else 'BASELINE'})")
    print(f"{'='*60}")

    # Load data — baseline=train.csv, augmented=syn/train_syn.csv
    data_dir = f"data/nusax_senti/{lang_code}"
    train_path = f"{data_dir}/syn/train_syn.csv" if USE_AUGMENTED else f"{data_dir}/train.csv"
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(f"{data_dir}/valid.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    label_list = sorted(train_df["label"].unique().tolist())
    label2id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}
    num_labels = len(label_list)
    print(f"Train src: {train_path}  | Labels: {label_list}")

    for df in [train_df, valid_df, test_df]:
        df["label"] = df["label"].map(label2id)

    features = Features({
        "id": Value("int64"),
        "text": Value("string"),
        "label": ClassLabel(names=label_list),
    })

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, features=features, preserve_index=False),
        "validation": Dataset.from_pandas(valid_df, features=features, preserve_index=False),
        "test": Dataset.from_pandas(test_df, features=features, preserve_index=False),
    })
    print(f"Train: {len(dataset['train'])}, Valid: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    # Tokenizer + config (XLM-R = SentencePiece, AutoTokenizer; tak perlu fix gamma/beta)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT, num_labels=num_labels, label2id=label2id, id2label=id2label,
    )
    print(f"CUDA: {torch.cuda.is_available()}, "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    def preprocess_function(examples):
        return tokenizer(examples["text"], max_length=INPUT_MAX_LENGTH, truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        return {"f1": round(f1["f1"], 4), "accuracy": round(acc["accuracy"], 4)}

    output_dir = f"{OUTPUT_BASE_DIR}/{model_name}-{lang_code}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,   # effective batch = PER_DEVICE_BATCH * GRAD_ACCUM = 32
        optim="adamw_torch_fused",                # AdamW (wd=0 → ≡ Adam NusaX)
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,                # 0 (NusaX tak punya wd)
        max_grad_norm=MAX_GRAD_NORM,              # 10 (NusaX Tabel 10)
        num_train_epochs=NUM_TRAIN_EPOCHS,
        # lr_scheduler_type default "linear" (NusaX γ undefined → pakai default HF)
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    # model_init agar classification head di-init SETELAH seed di-set (reproducible)
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=config)

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
    trainer.remove_callback(EarlyStoppingCallback)

    train_results = trainer.evaluate(tokenized_dataset["train"], metric_key_prefix="train")
    print(f"\nTrain [{lang_code}]: F1={train_results['train_f1']:.4f} Acc={train_results['train_accuracy']:.4f}")

    val_results = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="validation")
    print(f"Valid [{lang_code}]: F1={val_results['validation_f1']:.4f} Acc={val_results['validation_accuracy']:.4f}")

    train_history = trainer.state.log_history

    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test  [{lang_code}]: F1(macro)={test_results['eval_f1']:.4f} Acc={test_results['eval_accuracy']:.4f}")

    best_model_dir = f"{output_dir}/best"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"  Saved to: {best_model_dir}")

    import json
    history_path = f"{output_dir}/train_history.json"
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"  History: {history_path}")

    # Cleanup GPU
    import gc
    del trainer, tokenized_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Cleanup checkpoint-* dirs (keep best/ + train_history.json)
    import os
    import shutil
    deleted = 0
    for entry in os.listdir(output_dir):
        if entry.startswith("checkpoint-"):
            ckpt_path = os.path.join(output_dir, entry)
            if os.path.isdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                deleted += 1
    if deleted:
        print(f"  Deleted {deleted} checkpoint dirs")

    return {"train_results": train_results, "val_results": val_results,
            "test_results": test_results, "train_history": train_history}


if __name__ == "__main__":
    import json
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    nusax_ref = NUSAX_TABLE2[MODEL_SIZE]
    all_results = {}
    for lang in TARGET_LANGS:
        all_results[lang] = finetune(lang)

    # ── Summary + sanity-check vs NusaX Tabel 2 ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name} on NusaX-Senti ({'AUGMENTED' if USE_AUGMENTED else 'BASELINE'})")
    print(f"{'='*70}")
    print(f"{'lang':<5} {'train':>7} {'val':>7} {'test':>7}   {'NusaX-T2':>9} {'Δ':>7}  {'flag':>6}")
    for lang, res in all_results.items():
        train_f1 = res["train_results"]["train_f1"] * 100
        val_f1 = res["val_results"]["validation_f1"] * 100
        test_f1 = res["test_results"]["eval_f1"] * 100
        ref = nusax_ref.get(lang, 0)
        delta = test_f1 - ref
        # sanity-check hanya bermakna untuk BASELINE (augmented tak diharapkan match NusaX)
        flag = ""
        if not USE_AUGMENTED and ref:
            flag = "OK" if abs(delta) <= 2.0 else "CHECK"
        print(f"{lang:<5} {train_f1:>6.2f}% {val_f1:>6.2f}% {test_f1:>6.2f}%   {ref:>8.1f}% {delta:>+6.2f}  {flag:>6}")

    if not USE_AUGMENTED:
        print("\nSanity-check: baseline 'CHECK' (|Δ|>2pt vs NusaX Tabel 2) → "
              "recipe mungkin perlu ditinjau (mis. grid-search LR/batch). |Δ|≤2pt = OK (reproduksi wajar).")
        print("Catatan: deviasi wajar krn non-determinisme (seed/GPU/versi) + NusaX kemungkinan rata-rata banyak seed.")

    # Save summary
    summary_path = f"{OUTPUT_BASE_DIR}/results_summary.json"
    summary = {
        "hyperparams": {
            "model_checkpoint": MODEL_CHECKPOINT,
            "model_size": MODEL_SIZE,
            "use_augmented": USE_AUGMENTED,
            "target_langs": TARGET_LANGS,
            "input_max_length": INPUT_MAX_LENGTH,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_size_effective": TRAIN_BATCH_SIZE,
            "per_device_batch": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_grad_norm": MAX_GRAD_NORM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "linear",
            "seed": SEED,
            "source": "NusaX Winata 2023 Tabel 10 (LR/batch grid best; epochs/early-stop/max_norm fixed; wd absent=0; gamma undefined->linear)",
        },
        "results": {
            lang: {
                "train_f1": res["train_results"]["train_f1"],
                "train_accuracy": res["train_results"]["train_accuracy"],
                "val_f1": res["val_results"]["validation_f1"],
                "val_accuracy": res["val_results"]["validation_accuracy"],
                "test_f1": res["test_results"]["eval_f1"],
                "test_accuracy": res["test_results"]["eval_accuracy"],
            }
            for lang, res in all_results.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
