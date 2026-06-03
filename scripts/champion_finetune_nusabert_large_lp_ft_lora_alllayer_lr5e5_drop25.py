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
from peft import LoraConfig, get_peft_model

# LP-FT + LoRA ALL-LAYER (Q/K/V, r=16, α=32, LP=15, ALL 24 layers) — ABLATION
# Match paper convention (Hu et al. 2021): LoRA on ALL transformer layers.
# vs current champion (top-4 only): tests "is top-4 cap actually optimal?"
# Trainable: ~2.4M (vs top-4's 398K, 6x more) but still <1% of model.

MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
# TARGET_LANGS = ["jav", "bug", "nij", "ace", "ban", "bbc", "bjn", "eng", "ind", "mad", "min", "sun"]
TARGET_LANGS = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]
# TARGET_LANGS = ["bjn", "bug", "nij"]
INPUT_MAX_LENGTH = 128
LP_EPOCHS = 15  # R14 v2 winning longer probe
LP_LEARNING_RATE = 2e-4
FT_EPOCHS = 30
FT_LEARNING_RATE = 5e-5
LORA_ALL_LAYERS = True   # ABLATION: True=all 24 layers, False=top-4 only
WEIGHT_DECAY = 0.05
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value"]   # tambah "key" dari R11 v3
SEED = 42
USE_DORA = False   # vanilla LoRA
OUTPUT_BASE_DIR = "outputs/nusabert-sentiment-large-lpft-lora-alllayer-lr5e5-drop25-earlystop_3"


def finetune(lang_code: str):
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    print(f"\n{'='*50}")
    print(f"LP-FT LoRA-alllayer (LoRA r={LORA_R} α={LORA_ALPHA} target={LORA_TARGET_MODULES} LR={FT_LEARNING_RATE} LP={LP_EPOCHS}) {model_name} on NusaX-Senti [{lang_code}]")
    print(f"{'='*50}")

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

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label
    config.hidden_dropout_prob = 0.25
    config.attention_probs_dropout_prob = 0.25

    from transformers import AutoModelForSequenceClassification as AutoMSC
    from collections import OrderedDict

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
    clf_params = (config.hidden_size * num_labels + num_labels) / 1e6
    print(f"Model: {model_name} (~{total_params:.0f}M total params)")
    print(f"Trainable (clf head only): ~{clf_params*1e6:.0f} params ({clf_params/total_params*100:.2f}%)")
    print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

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

    output_dir = f"{OUTPUT_BASE_DIR}/{model_name}-{lang_code}"

    from transformers import BertModel
    from transformers.modeling_outputs import SequenceClassifierOutput

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

    print(f"\n--- Stage 1: Linear Probing (no pooler, head only, LR={LP_LEARNING_RATE}) ---")

    _fixed_state_dict = {k: v.clone() for k, v in new_state_dict.items()}

    def model_init_lp():
        m = BertLinearProbe()
        m.load_state_dict(_fixed_state_dict, strict=False)
        for param in m.bert.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen   = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        print(f"[LP] Pooler bypassed | Trainable (Linear only): {trainable:,} | Frozen: {frozen:,}")
        return m

    lp_args = TrainingArguments(
        output_dir=f"{output_dir}/lp",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        optim="adamw_torch_fused",
        learning_rate=LP_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=LP_EPOCHS,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    lp_trainer = Trainer(
        model_init=model_init_lp,
        args=lp_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    lp_trainer.train()
    lp_val = lp_trainer.evaluate(tokenized_dataset["validation"])
    print(f"[LP] Best val F1: {lp_val['eval_f1']:.4f}")

    lp_state_dict = {k: v.clone() for k, v in lp_trainer.model.state_dict().items()}
    lp_history = lp_trainer.state.log_history

    del lp_trainer
    torch.cuda.empty_cache()

    n_layers = config.num_hidden_layers
    lora_layers = list(range(0, n_layers)) if LORA_ALL_LAYERS else list(range(n_layers - 4, n_layers))
    print(f"\n--- Stage 2: FT (LoRA r={LORA_R} α={LORA_ALPHA} target={LORA_TARGET_MODULES} on layers {lora_layers}, LR={FT_LEARNING_RATE}) ---")

    def model_init_ft():
        m = BertLinearProbe()
        m.load_state_dict(lp_state_dict)

        for p in m.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            layers_to_transform=lora_layers,
            layers_pattern="layer",
            use_dora=USE_DORA,   # R15 v2: enable DoRA decomposition
        )
        m.bert = get_peft_model(m.bert, lora_config)

        for p in m.classifier.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen   = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        print(f"[FT-LoRA] LoRA on layers {lora_layers} ({LORA_TARGET_MODULES}) | head trainable")
        print(f"[FT-LoRA] Trainable: {trainable:,} ({trainable/(trainable+frozen)*100:.3f}%) | Frozen: {frozen:,}")
        return m

    ft_args = TrainingArguments(
        output_dir=f"{output_dir}/ft",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        optim="adamw_torch_fused",
        warmup_ratio=0.1,
        learning_rate=FT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=FT_EPOCHS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model_init=model_init_ft,
        args=ft_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    trainer.remove_callback(EarlyStoppingCallback)
    ft_history = trainer.state.log_history
    train_history = {"lp": lp_history, "ft": ft_history}

    train_results = trainer.evaluate(tokenized_dataset["train"], metric_key_prefix="train")
    print(f"\nTrain Results [{lang_code}]:")
    print(f"Train F1:  {train_results['train_f1']:.4f}")
    print(f"Train Acc: {train_results['train_accuracy']:.4f}")

    val_results = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="validation")
    print(f"\nValidation Results [{lang_code}]:")
    print(f"Val F1:  {val_results['validation_f1']:.4f}")
    print(f"Val Acc: {val_results['validation_accuracy']:.4f}")

    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"\nTest Results [{lang_code}]:")
    print(f"F1 (macro): {test_results['eval_f1']:.4f}")
    print(f"Accuracy:   {test_results['eval_accuracy']:.4f}")

    best_model_dir = f"{output_dir}/best"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Saved to: {best_model_dir}")

    import json
    history_path = f"{output_dir}/train_history.json"
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"History saved to: {history_path}")

    import gc
    del trainer, tokenized_dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory cleared. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    return {"train_results": train_results, "val_results": val_results, "test_results": test_results, "train_history": train_history}


if __name__ == "__main__":
    import json
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    all_results = {}
    for lang in TARGET_LANGS:
        result = finetune(lang)
        all_results[lang] = result

    print(f"\nSUMMARY: {model_name} LP-FT+LoRA r={LORA_R} target={LORA_TARGET_MODULES} LP={LP_EPOCHS} (LoRA-alllayer)")
    for lang, res in all_results.items():
        train_f1 = res["train_results"]["train_f1"] * 100
        val_f1   = res["val_results"]["validation_f1"] * 100
        test_f1  = res["test_results"]["eval_f1"] * 100
        print(f"  {lang}: train={train_f1:.2f}% val={val_f1:.2f}% test={test_f1:.2f}%")

    summary_path = f"{OUTPUT_BASE_DIR}/results_summary.json"
    summary = {
        "hyperparams": {
            "method": f"LP-FT-alllayer-LoRA-r{LORA_R}-alpha{LORA_ALPHA}-target-QKV-LR{FT_LEARNING_RATE}-LP{LP_EPOCHS}",
            "model_checkpoint": MODEL_CHECKPOINT,
            "target_langs": TARGET_LANGS,
            "input_max_length": INPUT_MAX_LENGTH,
            "lp_epochs": LP_EPOCHS,
            "lp_learning_rate": LP_LEARNING_RATE,
            "ft_epochs": FT_EPOCHS,
            "ft_learning_rate": FT_LEARNING_RATE,
            "lora_all_layers": LORA_ALL_LAYERS,
            "ft_warmup_ratio": 0.1,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "use_dora": USE_DORA,
            "hidden_dropout_prob": 0.25,
            "attention_probs_dropout_prob": 0.25,
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
            }
            for lang, res in all_results.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
