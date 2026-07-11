"""E4 — CH-Cloze / PET verbalizer fine-tuning (SV-grounding iterasi-2).

IDE: ganti head klasifikasi linear -> prediksi lewat MLM head NusaBERT. Input dibungkus template
'[teks]. Sentimen: [MASK].', loss = CE atas 3 logit verbalizer {negatif, netral, positif} di posisi [MASK].
MLM head + embedding FROZEN (decoder tied) -> kelas di-anchor ke embedding pretrained Indonesia masif,
BUKAN dipelajari head dari ~167 contoh/kelas. Mekanisme paling langsung nyerang recall-NEGATIVE ban/bjn.

BEDA dari drop06 (arm pembanding): SATU perubahan konseptual = head->verbalizer. Segala HP lain = drop06 PERSIS
(init pissa, lora_dropout 0.06, r16/a32 QKV all-layer, lr5e-5, hidden 0.25, wd0.05, bs8, patience5, maxlen128).
TANPA stage LP (head random tak ada -> rationale LP-FT gugur). Prediksi = argmax 3 token verbalizer.

Verbalizer (label sorted [negative,neutral,positive] -> id 0,1,2): negatif=3778, netral=11855, positif=2508 (semua single-token, verified).

SCREENING (default): 4 bahasa kontes (ace/ban/bjn/mad) x seed 42, KEEP=none. Kalau lolos -> confirm 7 x 5 seed.
Env override (opsional): PET_LANGS (json), PET_SEEDS (json), PET_KEEP (none|all), PET_OUT.

Jalankan dari root:  uv run python scripts/sv_grounding_2/pet_cloze.py
"""
import os
import gc
import json
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import evaluate
from transformers import (
    AutoConfig, BertTokenizerFast, AutoModelForMaskedLM, DataCollatorWithPadding,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model

MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
TARGET_LANGS = ["ace", "ban", "bjn", "mad"]   # screening kontes; confirm -> 7 scope
SEEDS = [42]
KEEP_WEIGHTS = "none"
OUTPUT_ROOT = "outputs/sv2-pet-drop06"

TEMPLATE_SECOND = "Sentimen: [MASK]."
VERBALIZER = ["negatif", "netral", "positif"]   # index = label id (sorted: negative/neutral/positive)
INPUT_MAX_LENGTH = 128
FT_EPOCHS = 30
FT_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.06            # = drop06
LORA_TARGET_MODULES = ["query", "key", "value"]
HIDDEN_DROPOUT = 0.25
INIT_LORA_WEIGHTS = "pissa"    # = drop06


class PetClozeModel(torch.nn.Module):
    """Wrap BertForMaskedLM (LoRA di .bert, .cls frozen) -> 3-way classifier via verbalizer di [MASK]."""
    def __init__(self, mlm, verbalizer_ids, mask_id):
        super().__init__()
        self.mlm = mlm
        self.register_buffer("verbalizer_ids", verbalizer_ids, persistent=False)
        self.mask_id = mask_id

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kw):
        out = self.mlm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = out.logits                                  # (B, L, V)
        is_mask = input_ids == self.mask_id
        # posisi [MASK] pertama per baris (template pasti punya 1, truncation only_first jaga template)
        mask_pos = is_mask.float().argmax(dim=1)             # (B,)
        B = input_ids.size(0)
        mask_logits = logits[torch.arange(B, device=logits.device), mask_pos]   # (B, V)
        verb_logits = mask_logits[:, self.verbalizer_ids]    # (B, 3)
        loss = torch.nn.CrossEntropyLoss()(verb_logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=verb_logits)


def finetune(lang_code, seed, output_base_dir, tokenizer, verbalizer_ids, mask_id):
    model_name = MODEL_CHECKPOINT.split("/")[-1].lower()
    output_dir = f"{output_base_dir}/{model_name}-{lang_code}"
    print(f"\n>>> PET [seed {seed}] [{lang_code}] init={INIT_LORA_WEIGHTS} drop={LORA_DROPOUT} lr={FT_LEARNING_RATE}")

    data_dir = f"data/nusax_senti/{lang_code}"
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    valid_df = pd.read_csv(f"{data_dir}/valid.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    label_list = sorted(train_df["label"].unique().tolist())   # [negative, neutral, positive]
    assert label_list == ["negative", "neutral", "positive"], f"label order tak terduga: {label_list}"
    label2id = {v: i for i, v in enumerate(label_list)}
    num_labels = len(label_list)
    for df in [train_df, valid_df, test_df]:
        df["label"] = df["label"].map(label2id)

    features = Features({"id": Value("int64"), "text": Value("string"), "label": ClassLabel(names=label_list)})
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, features=features, preserve_index=False),
        "validation": Dataset.from_pandas(valid_df, features=features, preserve_index=False),
        "test": Dataset.from_pandas(test_df, features=features, preserve_index=False),
    })

    def preprocess(examples):
        return tokenizer(examples["text"], [TEMPLATE_SECOND] * len(examples["text"]),
                         truncation="only_first", max_length=INPUT_MAX_LENGTH)
    tokenized = dataset.map(preprocess, batched=True)
    # guard: tiap contoh WAJIB punya tepat 1 [MASK]
    for split in ("train", "validation", "test"):
        bad = sum(1 for ids in tokenized[split]["input_ids"] if ids.count(mask_id) != 1)
        assert bad == 0, f"{split}: {bad} contoh tanpa/lebih-dari-1 [MASK]"
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        return {"f1": round(f1["f1"], 4), "accuracy": round(acc["accuracy"], 4)}

    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    config.hidden_dropout_prob = HIDDEN_DROPOUT
    config.attention_probs_dropout_prob = HIDDEN_DROPOUT
    n_layers = config.num_hidden_layers
    lora_layers = list(range(0, n_layers))

    def model_init():
        mlm = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT, config=config)
        # UNTIE decoder MLM dari word-embeddings + predictions.bias (dua-duanya frozen -> nilai identik,
        # cuma pisah storage). transformers 5.5.3 save checkpoint pakai safetensors yg NOLAK shared-memory tensor.
        mlm.cls.predictions.decoder.weight = torch.nn.Parameter(mlm.cls.predictions.decoder.weight.detach().clone())
        if getattr(mlm.cls.predictions.decoder, "bias", None) is not None:
            mlm.cls.predictions.decoder.bias = torch.nn.Parameter(mlm.cls.predictions.decoder.bias.detach().clone())
        for p in mlm.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT,
            bias="none", layers_to_transform=lora_layers, layers_pattern="layer",
            init_lora_weights=INIT_LORA_WEIGHTS,
        )
        mlm = get_peft_model(mlm, lora_config)   # .cls (MLM head) + embedding tetap frozen
        return PetClozeModel(mlm, verbalizer_ids, mask_id)

    ft_args = TrainingArguments(
        output_dir=f"{output_dir}/ft", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        optim="adamw_torch_fused", warmup_ratio=0.1, learning_rate=FT_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        num_train_epochs=FT_EPOCHS, save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1",
        bf16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed,
    )
    trainer = Trainer(model_init=model_init, args=ft_args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized["validation"], processing_class=tokenizer,
                      data_collator=data_collator, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(EARLY_STOPPING_PATIENCE)])
    trainer.train()
    _evals = [e for e in trainer.state.log_history if "eval_f1" in e]
    _best = max(_evals, key=lambda e: e["eval_f1"]) if _evals else {}
    stop_epoch = round(_evals[-1]["epoch"]) if _evals else None
    best_epoch = round(_best["epoch"]) if _evals else None
    trainer.remove_callback(EarlyStoppingCallback)
    ft_history = trainer.state.log_history

    train_results = trainer.evaluate(tokenized["train"], metric_key_prefix="train")
    val_results = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    test_results = trainer.evaluate(tokenized["test"])

    val_pred = trainer.predict(tokenized["validation"])
    test_pred = trainer.predict(tokenized["test"])
    y_true = test_pred.label_ids
    y_pred = np.argmax(test_pred.predictions, axis=1)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_labels)), zero_division=0)
    per_class = {label_list[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i in range(num_labels)}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels))).tolist()

    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/test_logits.npy", test_pred.predictions)
    np.save(f"{output_dir}/test_labels.npy", test_pred.label_ids)
    np.save(f"{output_dir}/val_logits.npy", val_pred.predictions)
    np.save(f"{output_dir}/val_labels.npy", val_pred.label_ids)

    gap = train_results["train_f1"] - test_results["eval_f1"]
    print(f"[seed {seed}][{lang_code}] train={train_results['train_f1']*100:.2f} val={val_results['validation_f1']*100:.2f} "
          f"test={test_results['eval_f1']*100:.2f} gap={gap*100:.2f} | negR={per_class['negative']['recall']*100:.1f} | stop@{stop_epoch} best@{best_epoch}")

    if KEEP_WEIGHTS == "all":
        best_dir = f"{output_dir}/best"
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)

    with open(f"{output_dir}/train_history.json", "w") as fj:
        json.dump({"ft": ft_history}, fj, indent=2)
    d = os.path.join(output_dir, "ft")
    if os.path.isdir(d):
        for entry in os.listdir(d):
            if entry.startswith("checkpoint-") and os.path.isdir(os.path.join(d, entry)):
                shutil.rmtree(os.path.join(d, entry))

    del trainer, tokenized
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "train_f1": train_results["train_f1"], "val_f1": val_results["validation_f1"],
        "test_f1": test_results["eval_f1"], "gap": gap, "stop_epoch": stop_epoch, "best_epoch": best_epoch,
        "per_class_test": per_class, "confusion_matrix_test": cm, "label_order": label_list,
    }


def main():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    mask_id = tokenizer.mask_token_id
    verb_ids = torch.tensor([tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in VERBALIZER])
    assert all(len(tokenizer(w, add_special_tokens=False)["input_ids"]) == 1 for w in VERBALIZER), "verbalizer bukan single-token"
    print(f"CUDA={torch.cuda.is_available()} | verbalizer {VERBALIZER}->{verb_ids.tolist()} mask={mask_id} | langs={TARGET_LANGS} seeds={SEEDS} keep={KEEP_WEIGHTS}")

    for seed in SEEDS:
        out = f"{OUTPUT_ROOT}/seed_{seed}"
        os.makedirs(out, exist_ok=True)
        sp = f"{out}/results_summary.json"
        results = json.load(open(sp))["results"] if os.path.exists(sp) else {}
        for lang in TARGET_LANGS:
            if lang in results:
                print(f"[seed {seed}][{lang}] SKIP (sudah ada)")
                continue
            res = finetune(lang, seed, out, tokenizer, verb_ids, mask_id)
            results[lang] = res
            json.dump({"config": {"init": INIT_LORA_WEIGHTS, "lora_dropout": LORA_DROPOUT, "template": TEMPLATE_SECOND,
                                  "verbalizer": VERBALIZER, "no_lp": True, "seed": seed}, "results": results},
                      open(sp, "w"), indent=2)
        print(f"[seed {seed}] -> {sp}")
    print("PET SELESAI.")


def _apply_env():
    for k, ek, fn in [("TARGET_LANGS", "PET_LANGS", json.loads), ("SEEDS", "PET_SEEDS", json.loads),
                      ("KEEP_WEIGHTS", "PET_KEEP", str), ("OUTPUT_ROOT", "PET_OUT", str)]:
        if ek in os.environ:
            globals()[k] = fn(os.environ[ek])


if __name__ == "__main__":
    _apply_env()
    main()
