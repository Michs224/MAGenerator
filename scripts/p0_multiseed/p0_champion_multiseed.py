"""P0 — CHAMPION (LP-FT + all-layer LoRA) MULTI-SEED.

Recipe = champion kanonik PERSIS (= outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25,
patience 5). Seed 42 = folder kanonik itu (di-REUSE, tidak dilatih ulang); script ini
melatih seed LAINNYA.

Stage 1 Linear-Probe: mean-pool head only, backbone beku, LR 2e-4, 15 ep.
Stage 2 LoRA FT: backbone beku + LoRA (r16/a32, Q/K/V, ALL 24 layer, lora_drop 0.1) + head,
                 LR 5e-5, warmup 0.1, wd 0.05, hidden/attn dropout 0.25, bs 8, patience 5, 30 ep.

Cleanup champion: hapus checkpoint-*, SISAIN best/ (bobot model, buat deploy) + dump logits
test/val (.npy, buat kalibrasi/temperature-scaling nanti tanpa train ulang).
  KEEP_WEIGHTS = "all"         -> simpan bobot best/ tiap seed (sesuai instruksi; ~16GB/seed)
  KEEP_WEIGHTS = "deploy_only" -> simpan bobot best/ HANYA DEPLOY_SEED; seed lain cukup logits

Jalankan dari root:  uv run python scripts/p0_multiseed/p0_champion_multiseed.py
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
    BertModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model

# ---- Config (champion kanonik) ----
MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
TARGET_LANGS = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]  # 12 langs
SEEDS = [0, 1, 2, 3]            # seed 42 = champion kanonik (di-reuse)
DEPLOY_SEED = 42               # seed yang ditetapkan di muka untuk deploy (relevan saat KEEP_WEIGHTS=deploy_only)
KEEP_WEIGHTS = "all"           # "all" | "deploy_only"

INPUT_MAX_LENGTH = 128
LP_EPOCHS = 15
LP_LEARNING_RATE = 2e-4
FT_EPOCHS = 30
FT_LEARNING_RATE = 5e-5
LORA_ALL_LAYERS = True
WEIGHT_DECAY = 0.05
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5     # champion kanonik = 5 (BUKAN 3)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value"]
USE_DORA = False
HIDDEN_DROPOUT = 0.25
OUTPUT_ROOT = "outputs/p0-champion-multiseed"


def finetune(lang_code: str, seed: int, output_base_dir: str):
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    output_dir = f"{output_base_dir}/{model_name}-{lang_code}"

    done_marker = f"{output_dir}/train_history.json"
    if os.path.exists(done_marker):
        print(f"[seed {seed}][{lang_code}] SKIP (sudah ada {done_marker})")
        return None

    print(f"\n>>> champion LP-FT+LoRA [seed {seed}] {model_name} [{lang_code}]")

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
    config.hidden_dropout_prob = HIDDEN_DROPOUT
    config.attention_probs_dropout_prob = HIDDEN_DROPOUT

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

    class BertLinearProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.bert = BertModel(config, add_pooling_layer=False)
            self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
            out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            logits = self.classifier(pooled)
            loss = torch.nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

    # ---- Stage 1: Linear Probing ----
    _fixed_state_dict = {k: v.clone() for k, v in new_state_dict.items()}

    def model_init_lp():
        m = BertLinearProbe()
        m.load_state_dict(_fixed_state_dict, strict=False)
        for p in m.bert.parameters():
            p.requires_grad = False
        return m

    lp_args = TrainingArguments(
        output_dir=f"{output_dir}/lp", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        optim="adamw_torch_fused", learning_rate=LP_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        num_train_epochs=LP_EPOCHS, save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="f1", bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed,
    )
    lp_trainer = Trainer(
        model_init=model_init_lp, args=lp_args, train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], processing_class=tokenizer,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )
    lp_trainer.train()
    lp_state_dict = {k: v.clone() for k, v in lp_trainer.model.state_dict().items()}
    lp_history = lp_trainer.state.log_history
    del lp_trainer
    torch.cuda.empty_cache()

    # ---- Stage 2: FT (LoRA all-layer) ----
    n_layers = config.num_hidden_layers
    lora_layers = list(range(0, n_layers)) if LORA_ALL_LAYERS else list(range(n_layers - 4, n_layers))

    def model_init_ft():
        m = BertLinearProbe()
        m.load_state_dict(lp_state_dict)
        for p in m.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT,
            bias="none", layers_to_transform=lora_layers, layers_pattern="layer", use_dora=USE_DORA,
        )
        m.bert = get_peft_model(m.bert, lora_config)
        for p in m.classifier.parameters():
            p.requires_grad = True
        return m

    ft_args = TrainingArguments(
        output_dir=f"{output_dir}/ft", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        optim="adamw_torch_fused", warmup_ratio=0.1, learning_rate=FT_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        num_train_epochs=FT_EPOCHS, save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1",
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed,
    )
    trainer = Trainer(
        model_init=model_init_ft, args=ft_args, train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], processing_class=tokenizer,
        data_collator=data_collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()

    _evals = [e for e in trainer.state.log_history if "eval_f1" in e]
    _best = max(_evals, key=lambda e: e["eval_f1"]) if _evals else {}
    stop_epoch = round(_evals[-1]["epoch"]) if _evals else None
    best_epoch = round(_best["epoch"]) if _evals else None

    trainer.remove_callback(EarlyStoppingCallback)
    ft_history = trainer.state.log_history
    train_results = trainer.evaluate(tokenized_dataset["train"], metric_key_prefix="train")
    val_results = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="validation")
    test_results = trainer.evaluate(tokenized_dataset["test"])

    # Per-class + confusion + DUMP LOGITS (val & test) untuk kalibrasi nanti
    val_pred = trainer.predict(tokenized_dataset["validation"])
    test_pred = trainer.predict(tokenized_dataset["test"])
    y_true = test_pred.label_ids
    y_pred = np.argmax(test_pred.predictions, axis=1)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_labels)), zero_division=0)
    per_class = {label_list[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i in range(num_labels)}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels))).tolist()

    np.save(f"{output_dir}/test_logits.npy", test_pred.predictions)
    np.save(f"{output_dir}/test_labels.npy", test_pred.label_ids)
    np.save(f"{output_dir}/val_logits.npy", val_pred.predictions)
    np.save(f"{output_dir}/val_labels.npy", val_pred.label_ids)

    gap = train_results["train_f1"] - test_results["eval_f1"]
    print(f"[seed {seed}][{lang_code}] train={train_results['train_f1']*100:.2f} val={val_results['validation_f1']*100:.2f} "
          f"test={test_results['eval_f1']*100:.2f} gap={gap*100:.2f} | stop@ep{stop_epoch} best@ep{best_epoch}")

    # Simpan best/ (bobot) sesuai KEEP_WEIGHTS
    keep_weights = (KEEP_WEIGHTS == "all") or (seed == DEPLOY_SEED)
    if keep_weights:
        best_dir = f"{output_dir}/best"
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"  [champion] best/ disimpan (bobot)")
    else:
        print(f"  [champion] KEEP_WEIGHTS=deploy_only & seed!=DEPLOY_SEED -> bobot tidak disimpan (logits tetap ada)")

    with open(f"{output_dir}/train_history.json", "w") as fjson:
        json.dump({"lp": lp_history, "ft": ft_history}, fjson, indent=2)

    for sub in ("ft", "lp"):
        d = os.path.join(output_dir, sub)
        if os.path.isdir(d):
            for entry in os.listdir(d):
                if entry.startswith("checkpoint-") and os.path.isdir(os.path.join(d, entry)):
                    shutil.rmtree(os.path.join(d, entry))

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


def _write_summary(summary_path, seed, results):
    summary = {
        "hyperparams": {
            "method": f"LP-FT-alllayer-LoRA-r{LORA_R}-alpha{LORA_ALPHA}-QKV-LR{FT_LEARNING_RATE}-LP{LP_EPOCHS}",
            "model_checkpoint": MODEL_CHECKPOINT, "target_langs": TARGET_LANGS, "input_max_length": INPUT_MAX_LENGTH,
            "lp_epochs": LP_EPOCHS, "lp_learning_rate": LP_LEARNING_RATE, "ft_epochs": FT_EPOCHS,
            "ft_learning_rate": FT_LEARNING_RATE, "lora_all_layers": LORA_ALL_LAYERS, "ft_warmup_ratio": 0.1,
            "weight_decay": WEIGHT_DECAY, "train_batch_size": TRAIN_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "lora_target_modules": LORA_TARGET_MODULES, "use_dora": USE_DORA,
            "hidden_dropout_prob": HIDDEN_DROPOUT, "attention_probs_dropout_prob": HIDDEN_DROPOUT, "seed": seed,
        },
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def run_seed(seed: int):
    output_base_dir = f"{OUTPUT_ROOT}/seed_{seed}"
    os.makedirs(output_base_dir, exist_ok=True)
    summary_path = f"{output_base_dir}/results_summary.json"
    results = json.load(open(summary_path))["results"] if os.path.exists(summary_path) else {}

    for lang in TARGET_LANGS:
        res = finetune(lang, seed, output_base_dir)
        if res is not None:
            results[lang] = res
            _write_summary(summary_path, seed, results)

    _write_summary(summary_path, seed, results)
    print(f"\n[seed {seed}] summary -> {summary_path}")


def main():
    print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"P0 champion multi-seed | seeds={SEEDS} (42 di-reuse) | langs={len(TARGET_LANGS)} | USE_DORA={USE_DORA} | KEEP_WEIGHTS={KEEP_WEIGHTS} | out={OUTPUT_ROOT}")
    for seed in SEEDS:
        run_seed(seed)
    print("\nSELESAI. Lanjut: p0_aggregate.py")


if __name__ == "__main__":
    main()
