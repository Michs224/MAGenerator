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

# Config — LP-FT (Linear Probing then Fine-Tuning)
MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
# TARGET_LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
# TARGET_LANGS = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]
TARGET_LANGS = ["jav"]
INPUT_MAX_LENGTH = 128
# Stage 1: Linear Probing
LP_EPOCHS = 10
LP_LEARNING_RATE = 2e-4
# Stage 2: Fine-Tuning
FT_EPOCHS = 50
FT_LEARNING_RATE = 1e-5
FT_UNFREEZE_TOP_LAYERS = 4
WEIGHT_DECAY = 0.05
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5
SEED = 42
OUTPUT_BASE_DIR = "outputs/nusabert-sentiment-lpft"


def finetune(lang_code: str):
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    print(f"\n{'='*50}")
    print(f"LP-FT {model_name} on NusaX-Senti [{lang_code}]")
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
    config.hidden_dropout_prob = 0.2 # default 0.1
    config.attention_probs_dropout_prob = 0.2 # default 0.1
    
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

    # Hitung trainable params: encoder frozen, hanya classifier head
    total_params = sum(v.numel() for v in new_state_dict.values()) / 1e6
    # Classifier head: hidden_size * num_labels + num_labels (weight + bias)
    clf_params = (config.hidden_size * num_labels + num_labels) / 1e6
    print(f"Model: {model_name} (~{total_params:.0f}M total params)")
    print(f"Trainable (clf head only): ~{clf_params*1e6:.0f} params ({clf_params/total_params*100:.2f}%)")
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

    # Custom model: bypass pooler sepenuhnya, [CLS] hidden state → Linear langsung
    # Sesuai paper probing (Peters et al., Tenney et al.) — pooler tidak dipakai sama sekali
    from transformers import BertModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class BertLinearProbe(torch.nn.Module):
        """Mask-aware mean pooling → Linear (no pooler). Exclude padding tokens dari rata-rata."""
        def __init__(self):
            super().__init__()
            self.config = config  # expose config for HuggingFace Trainer
            self.bert = BertModel(config, add_pooling_layer=False)
            self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
            out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            logits = self.classifier(pooled)
            loss = torch.nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

    # ── Stage 1: Linear Probing (freeze all BERT, train head only) ──
    print(f"\n--- Stage 1: Linear Probing (no pooler, head only, LR={LP_LEARNING_RATE}) ---")

    _fixed_state_dict = {k: v.clone() for k, v in new_state_dict.items()}

    def model_init_lp():
        m = BertLinearProbe()
        m.load_state_dict(_fixed_state_dict, strict=False)  # load encoder/emb, skip pooler+clf keys
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
        metric_for_best_model="loss",
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

    # Save LP state dict for Stage 2
    lp_state_dict = {k: v.clone() for k, v in lp_trainer.model.state_dict().items()}
    lp_history = lp_trainer.state.log_history

    del lp_trainer
    torch.cuda.empty_cache()

    # ── Stage 2: Fine-Tuning (unfreeze top layers, low LR) ──
    print(f"\n--- Stage 2: Fine-Tuning (top-{FT_UNFREEZE_TOP_LAYERS} layers, LR={FT_LEARNING_RATE}) ---")

    def model_init_ft():
        m = BertLinearProbe()
        m.load_state_dict(lp_state_dict)  # same architecture, strict load
        n_layers = len(m.bert.encoder.layer)
        freeze_until = n_layers - FT_UNFREEZE_TOP_LAYERS

        for param in m.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(m.bert.encoder.layer):
            for param in layer.parameters():
                param.requires_grad = (i >= freeze_until)
        # classifier head: always trainable (no pooler)

        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen   = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        print(f"[FT] Frozen: 0-{freeze_until-1} | Trainable: {freeze_until}-{n_layers-1} + head (no pooler)")
        print(f"[FT] Trainable: {trainable:,} | Frozen: {frozen:,}")
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
        metric_for_best_model="loss",
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

    # Evaluate train set
    train_results = trainer.evaluate(tokenized_dataset["train"], metric_key_prefix="train")
    print(f"\nTrain Results [{lang_code}] ({model_name}, LP-FT):")
    print(f"Train F1:  {train_results['train_f1']:.4f}")
    print(f"Train Acc: {train_results['train_accuracy']:.4f}")

    val_results = trainer.evaluate(tokenized_dataset["validation"], metric_key_prefix="validation")
    print(f"\nValidation Results [{lang_code}] ({model_name}, LP-FT):")
    print(f"Val F1:  {val_results['validation_f1']:.4f}")
    print(f"Val Acc: {val_results['validation_accuracy']:.4f}")

    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"\nTest Results [{lang_code}] ({model_name}, LP-FT):")
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

    print(f"\nSUMMARY: {model_name} LP-FT")
    for lang, res in all_results.items():
        train_f1 = res["train_results"]["train_f1"] * 100
        val_f1   = res["val_results"]["validation_f1"] * 100
        test_f1  = res["test_results"]["eval_f1"] * 100
        print(f"  {lang}: train={train_f1:.2f}% val={val_f1:.2f}% test={test_f1:.2f}%")

    summary_path = f"{OUTPUT_BASE_DIR}/results_summary.json"
    summary = {
        "hyperparams": {
            "method": "LP-FT",
            "model_checkpoint": MODEL_CHECKPOINT,
            "target_langs": TARGET_LANGS,
            "input_max_length": INPUT_MAX_LENGTH,
            "lp_epochs": LP_EPOCHS,
            "lp_learning_rate": LP_LEARNING_RATE,
            "ft_epochs": FT_EPOCHS,
            "ft_learning_rate": FT_LEARNING_RATE,
            "ft_unfreeze_top_layers": FT_UNFREEZE_TOP_LAYERS,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "hidden_dropout_prob": 0.2,
            "attention_probs_dropout_prob": 0.2,
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
