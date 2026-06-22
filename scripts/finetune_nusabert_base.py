import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from transformers import (
    AutoConfig,
    BertTokenizerFast,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# Config — NusaBERT-base hyperparams sesuai paper Table 4 + GitHub run_classification.sh
MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-base"
TARGET_LANGS = ["ace", "jav", "mad", "ban", "bjn", "min", "sun"]
# TARGET_LANGS = ["mad", "ban", "bjn", "jav", "ace", "min", "sun"]
# TARGET_LANGS = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]
# TARGET_LANGS = ["jav"]
INPUT_MAX_LENGTH = 128
NUM_TRAIN_EPOCHS = 100
LEARNING_RATE = 1e-5       # large = 2e-5 (base = 1e-5)
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 32        # paper = 32
EVAL_BATCH_SIZE = 64      # paper = 64, turunkan untuk VRAM
# GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = 4 * 4 = 16 (sama dengan paper)
EARLY_STOPPING_PATIENCE = 5 #3
SEED = 123
OUTPUT_BASE_DIR = "outputs/nusabert-sentiment-base-syn-123"


def finetune(lang_code: str):
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    print(f"\n{'='*50}")
    print(f"Fine-tuning {model_name} on NusaX-Senti [{lang_code}]")
    print(f"{'='*50}")

    # Load data
    data_dir = f"data/nusax_senti/{lang_code}"
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    valid_df = pd.read_csv(f"{data_dir}/valid.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    label_list = sorted(train_df["label"].unique().tolist())
    label2id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}
    num_labels = len(label_list)
    print(f"Labels: {label_list}")

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

    # Model
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label

    # Fix LayerNorm naming: NusaBERT uses gamma/beta, transformers expects weight/bias
    # We fix at state_dict level so checkpoints are also saved with correct naming
    from transformers import AutoModelForSequenceClassification as AutoMSC
    from collections import OrderedDict

    # Load raw state dict and rename gamma/beta → weight/bias
    raw_model = AutoMSC.from_pretrained(
        MODEL_CHECKPOINT, config=config, ignore_mismatched_sizes=True,
    )
    state_dict = raw_model.state_dict()
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
        new_state_dict[new_key] = value

    del raw_model
    total_params = sum(v.numel() for v in new_state_dict.values()) / 1e6
    print(f"Model: {model_name} (~{total_params:.0f}M params)")
    print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Tokenize
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
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        # gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        # warmup_ratio=0.1,
        # label_smoothing_factor=0.1,
        # max_grad_norm=1.0,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_available(),
        report_to="none",
        # logging_steps=32,
        seed=SEED,
        data_seed=SEED,
    )

    # model_init agar classification head di-initialize SETELAH seed di-set
    # Ini menjamin reproducibility — tanpa ini, classifier head random tiap run
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

    # Early-stop diagnostics — dihitung SEBELUM post-hoc evaluate() mengontaminasi
    # log_history (evaluate(test) pakai prefix default "eval" → ikut menambah entri
    # eval_f1 = skor TEST). Di titik ini log_history hanya berisi eval val per-epoch.
    # Model yang di-restore = checkpoint best-epoch (load_best_model_at_end=True);
    # tie-break Trainer (strict >) = epoch paling awal, sama dengan max() Python.
    _evals = [e for e in trainer.state.log_history if "eval_f1" in e]
    _best = max(_evals, key=lambda e: e["eval_f1"]) if _evals else {}
    stop_epoch = round(_evals[-1]["epoch"]) if _evals else None
    best_epoch = round(_best["epoch"]) if _evals else None
    best_val_f1 = _best.get("eval_f1") if _evals else None
    epochs_past_best = (stop_epoch - best_epoch) if _evals else None
    print(f"\nEarly-stop diagnostics [{lang_code}]:")
    print(f"  Stopped at epoch:    {stop_epoch} (patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Best/restored epoch: {best_epoch} (val_f1={best_val_f1})")
    print(f"  Epochs past best:    {epochs_past_best}")

    # Remove EarlyStoppingCallback before post-training evals to suppress false warnings
    # (callback looks for "eval_f1" but post-training prefix is "train"/"validation")
    trainer.remove_callback(EarlyStoppingCallback)

    # Evaluate pada train set (1x, untuk lihat train F1/accuracy final)
    train_results = trainer.evaluate(tokenized_dataset["train"], metric_key_prefix="train")
    print(f"\nTrain Results [{lang_code}] ({model_name}):")
    print(f"  Train F1:  {train_results['train_f1']:.4f}")
    print(f"  Train Acc: {train_results['train_accuracy']:.4f}")

    val_results = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="validation")
    print(f"\nValidation Results [{lang_code}] ({model_name}):")
    print(f"  F1:  {val_results['validation_f1']:.4f}")
    print(f"  Accuracy: {val_results['validation_accuracy']:.4f}")

    # Simpan training history
    train_history = trainer.state.log_history

    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"\nTest Results [{lang_code}] ({model_name}):")
    print(f"  F1 (macro): {test_results['eval_f1']:.4f}")
    print(f"  Accuracy:   {test_results['eval_accuracy']:.4f}")

    best_model_dir = f"{output_dir}/best"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"  Saved to: {best_model_dir}")

    # Simpan history ke JSON
    import json
    history_path = f"{output_dir}/train_history.json"
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"History saved to: {history_path}")

    # Cleanup GPU memory
    import gc
    del trainer, tokenized_dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory cleared. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

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
        print(f"Deleted {deleted} checkpoint dirs in {output_dir}")

    return {"train_results": train_results, "val_results": val_results, "test_results": test_results, "train_history": train_history,
            "stop_epoch": stop_epoch, "best_epoch": best_epoch, "best_val_f1": best_val_f1, "epochs_past_best": epochs_past_best}


if __name__ == "__main__":
    import json
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    all_results = {}
    for lang in TARGET_LANGS:
        result = finetune(lang)
        all_results[lang] = result

    # Paper reference (NusaBERT-base, from Table 6)
    paper_scores = {
        "ace": 76.5, "ban": 78.7, "bbc": 74.0, "bjn": 82.4, "bug": 71.6,
        "eng": 84.1, "ind": 89.7, "jav": 84.1, "mad": 75.6, "min": 80.8,
        "nij": 74.9, "sun": 85.2,
    }

    print(f"\nSUMMARY: {model_name} on NusaX-Senti")
    for lang, res in all_results.items():
        train_f1 = res["train_results"]["train_f1"] * 100
        val_f1   = res["val_results"]["validation_f1"] * 100
        test_f1  = res["test_results"]["eval_f1"] * 100
        paper_f1 = paper_scores.get(lang, 0)
        print(f"  {lang}: train={train_f1:.2f}% val={val_f1:.2f}% test={test_f1:.2f}% (paper - test set: {paper_f1}%) | stop@ep{res['stop_epoch']} best@ep{res['best_epoch']}")

    summary_path = f"{OUTPUT_BASE_DIR}/results_summary.json"
    summary = {
        "hyperparams": {
            "model_checkpoint": MODEL_CHECKPOINT,
            "target_langs": TARGET_LANGS,
            "input_max_length": INPUT_MAX_LENGTH,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            # "warmup_ratio": 0.1,
            # "label_smoothing_factor": 0.1,
            # "max_grad_norm": 1.0,
            "seed": SEED,
        },
        "results": {
            lang: {
                "train_f1": res["train_results"]["train_f1"],
                "train_accuracy": res["train_results"]["train_accuracy"],
                "val_f1": res["val_results"]["validation_f1"],
                "val_accuracy": res["val_results"]["validation_accuracy"],
                "test_f1": res["test_results"]["eval_f1"],
                "test_accuracy": res["test_results"]["eval_accuracy"],
                "stop_epoch": res["stop_epoch"],
                "best_epoch": res["best_epoch"],
                "best_val_f1": res["best_val_f1"],
                "epochs_past_best": res["epochs_past_best"],
            }
            for lang, res in all_results.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
