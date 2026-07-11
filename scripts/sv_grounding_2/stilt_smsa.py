"""E6 STAGE-0 — STILTs: intermediate full-FT NusaBERT-large di SmSA (sentimen Indonesia ~11K), leakage-filtered.

IDE: tanam decision boundary sentimen (termasuk NEGATIVE) dari 11K contoh Indonesia DULU (task sama, domain
IDENTIK -- NusaX = subset SmSA yg diterjemah), sebelum FT per-bahasa. Stage bahasa daerah tinggal adaptasi
leksikal -> tekanan memorisasi turun -> gap turun + recall-NEGATIVE naik. Bukti DeFT-X: ban 84.8 achievable.

⚠️ LEAKAGE FATAL kalau lalai: NusaX (12 bahasa PARALEL) berasal dari SmSA. Kalimat ind NusaX = versi Indonesia
dari kalimat test lokal. Kalau SmSA train memuat kalimat itu -> model liat "original" test -> bocor. Maka:
BUANG tiap baris SmSA yg exact (hard-norm: buang semua non-alnum) atau near-dup (Levenshtein edit-ratio >=0.6,
via difflib, char-trigram cuma dipakai buat PRUNE kandidat) vs kolom ind NusaX SEMUA split (train+val+test,
paralel = 1 set ~1000 kalimat cover 12 bahasa). TIERED: exact+edit>=0.85 = TIER-1 bersih (~nol false-positive,
mekanis: detok/recase/ejaan); edit 0.6-0.85 = TIER-2 ambigu (konservatif, shorten/substitusi-entitas). Sudah
DIBUKTIKAN (6+ teknik dicoba: token-Jaccard/char-trigram/LCS/containment/length-ratio/semantik) separasi
SEMPURNA (nol false-positive) MUSTAHIL -- SmSA punya kalimat template ("saya tidak kecewa dengan X") yg beda
entitas tapi skor tinggi di semua metrik permukaan. Ini batas keras data (butuh id-provenance yg tak
dipublikasi), bukan kurang teknik. Threshold 0.6 = optimum praktis (condong recall, false-positive murah, false-
negative fatal). CATATAN: NusaX (paper) = 1000 stratified dari SmSA + FILTER abusive + diterjemah; sebagian
NusaX tak persis di TSV IndoNLU ini (versi/cleaning beda) -> exact maksimal ~537-728/1000, TAPI utk leakage
cukup nangkep yg ADA di SmSA-train (itulah yg dilakukan filter ini). Detail lengkap + tabel di
agent_doc/sv_grounding_2/results_2.md bagian E6.

Pipeline script ini (STAGE-0 saja):
  1. fetch 3 TSV SmSA (github IndoNLU) -> cache data/smsa/
  2. leakage-filter TIERED vs NusaX ind (hard-norm exact + Levenshtein edit-ratio, lihat leakage_filter())
  3. full-FT NusaBERT-large di SmSA terfilter (HP warisan FT-repro: LR2e-5/bs16/wd0.01/<=3ep/early-stop) -> outputs/sv2-smsa-stage0/best
  4. GATE zero-shot: eval checkpoint stage-0 di 7 NusaX test set (menit) -> ukur transfer ind->lokal SEBELUM bayar stage-1

STAGE-1 (jalankan TERPISAH setelah gate OK) = resep drop06 per-bahasa, init dari stage-0:
  P1_OVERRIDES='{"INIT_LORA_WEIGHTS":"pissa","LORA_DROPOUT":0.06,"MODEL_CHECKPOINT":"outputs/sv2-smsa-stage0/best"}' \
  P1_OUTPUT_ROOT=outputs/sv2-stilt P1_SEEDS='[42]' P1_TARGET_LANGS='["ace","ban","bjn","mad"]' P1_KEEP_WEIGHTS=none \
  uv run python scripts/sv_grounding/p1_sweep/champion_p1_base.py

Jalankan stage-0 dari root:  uv run python scripts/sv_grounding_2/stilt_smsa.py
"""
import os
import csv
import json
import urllib.request

import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset, Features, Value, ClassLabel
from sklearn.metrics import f1_score
from transformers import (
    AutoConfig, BertTokenizerFast, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback,
)

MODEL_CHECKPOINT = "LazarusNLP/NusaBERT-large"
LABEL_LIST = ["negative", "neutral", "positive"]
LABEL2ID = {v: i for i, v in enumerate(LABEL_LIST)}
SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
SMSA_BASE = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa"
SMSA_DIR = "data/smsa"
STAGE0_OUT = "outputs/sv2-smsa-stage0"
MAXLEN = 128
EDIT_RATIO_THRESHOLD = 0.6     # Levenshtein/difflib ratio buat near-dup. Bukti inspeksi: bahkan ratio 0.63 masih
                               # kalimat SAMA yg diedit NusaX (foodcourt->tempat makan, kalimat dipendekin); <0.6 baru beda.
TRIGRAM_BLOCK = 0.3            # prune kandidat via char-trigram SEBELUM edit-distance (biar cepat, bukan O(n*m) penuh)
SEED = 42

f1m = evaluate.load("f1")
accm = evaluate.load("accuracy")


def fetch_smsa():
    os.makedirs(SMSA_DIR, exist_ok=True)
    out = {}
    for split, fname in [("train", "train_preprocess"), ("valid", "valid_preprocess"), ("test", "test_preprocess")]:
        path = f"{SMSA_DIR}/{fname}.tsv"
        if not os.path.exists(path):
            print(f"  download {fname}.tsv ...")
            urllib.request.urlretrieve(f"{SMSA_BASE}/{fname}.tsv", path)
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"],
                         quoting=csv.QUOTE_NONE, on_bad_lines="skip")
        df = df[df["label"].isin(LABEL_LIST)].reset_index(drop=True)
        out[split] = df
        print(f"  SmSA {split}: {len(df)} baris")
    return out


import re
from difflib import SequenceMatcher
_ALNUM = re.compile(r"[^0-9a-z]")


def _norm_hard(s):
    # Buang SEMUA non-alfanumerik + lowercase. SmSA = teks pre-tokenized (tanda baca dispasikan,
    # hyphenasi beda) vs NusaX ind = teks bersih. Normalisasi 'lunak' (strip punct saja) under-catch
    # (exact cuma 280/1000); normalisasi KERAS ini bikin "tahu-nya, tidak" == "tahu nya , tidak" (exact 537/1000).
    return _ALNUM.sub("", str(s).lower())


def _trigrams(s):
    h = _norm_hard(s)
    return set(h[i:i + 3] for i in range(len(h) - 2)) if len(h) >= 3 else ({h} if h else set())


def leakage_filter(smsa_df):
    """Buang baris SmSA yg exact (hard-norm) atau near-dup vs kolom ind NusaX (train+val+test).

    TEKNIK (cocok utk kasus kita = deteksi kalimat-SAMA-beda-format, satu bahasa Indonesia):
    - hard-norm exact: nangkep beda format murni (spasi/tanda-baca/tokenisasi) -> 537/1000, presisi 100%.
    - Levenshtein edit-ratio (difflib): ukur % karakter yg harus diedit. Ini metrik PRINSIPIL utk 'kalimat sama
      + edit kecil' (NusaX meng-clean/mempersingkat sumber SmSA). BUKAN embedding semantik (itu salah utk kasus
      ini -- dua review negatif beda = mirip makna tapi bukan duplikat -> false-positive).
    - char-trigram cuma dipakai buat PRUNE kandidat (blocking) biar edit-distance nggak O(n*m) penuh.
    Bukti threshold: inspeksi manual -> ratio 0.63 masih kalimat sama diedit; <0.6 baru beda -> pakai 0.6."""
    ind_texts = []
    for split in ["train", "valid", "test"]:
        ind_texts += pd.read_csv(f"data/nusax_senti/ind/{split}.csv")["text"].tolist()
    ind_hard = [_norm_hard(t) for t in ind_texts]
    ind_hard_set = set(ind_hard)
    ind_tri = [_trigrams(t) for t in ind_texts]
    inv = {}                                   # inverted index char-trigram -> idx kalimat ind (buat blocking)
    for i, tg in enumerate(ind_tri):
        for g in tg:
            inv.setdefault(g, set()).add(i)

    # TIERED: klasifikasi tiap baris dibuang -> tier1 (edit mekanis, ~nol false-pos) vs tier2 (edit konten, ambigu).
    # Alasan: transformasi NusaX = (a) mekanis reversibel [detok/recase/ejaan] -> exact/edit>=0.85 = bersih;
    # (b) konten lossy [shorten/substitusi entitas] -> edit 0.6-0.85 = campur leak-berat & template-false-positive
    # yg MUSTAHIL dipisah (substitusi bikin "prabowo"->"produk apple" identik permukaan dgn kalimat template beda).
    TIER1_MIN = 0.85
    keep_idx, n_exact, n_tier1, n_tier2 = [], 0, 0, 0
    for ridx, text in enumerate(smsa_df["text"]):
        h = _norm_hard(text)
        if h in ind_hard_set:
            n_exact += 1
            continue
        tg = _trigrams(text)
        cand = set()
        for g in tg:
            cand |= inv.get(g, set())
        best = 0.0
        for ci in cand:
            it = ind_tri[ci]
            if len(tg & it) / max(1, len(tg | it)) < TRIGRAM_BLOCK:   # prune kandidat jauh
                continue
            r = SequenceMatcher(None, h, ind_hard[ci]).ratio()
            if r > best:
                best = r
                if best >= TIER1_MIN:
                    break                                            # cukup: sudah tier-1, tak perlu cari lebih
        if best >= TIER1_MIN:
            n_tier1 += 1
            continue
        if best >= EDIT_RATIO_THRESHOLD:
            n_tier2 += 1
            continue
        keep_idx.append(ridx)
    filtered = smsa_df.iloc[keep_idx].reset_index(drop=True)
    n_removed = n_exact + n_tier1 + n_tier2
    n_clean = n_exact + n_tier1
    print(f"  leakage-filter (hard-norm exact + edit-ratio, TIERED): buang {n_removed} | sisa {len(filtered)}/{len(smsa_df)}")
    print(f"    TIER-1 bersih (~nol false-pos): exact {n_exact} + edit>={TIER1_MIN} {n_tier1} = {n_clean}")
    print(f"    TIER-2 ambigu (edit {EDIT_RATIO_THRESHOLD}-{TIER1_MIN}, konservatif): {n_tier2}")
    return filtered, {"removed_total": n_removed, "removed_exact": n_exact, "removed_tier1_mechanical": n_tier1,
                      "removed_tier2_ambiguous": n_tier2, "clean_highconf": n_clean, "kept": len(filtered),
                      "orig": len(smsa_df), "norm": "hard(alnum-only)", "near_metric": "levenshtein-edit-ratio(difflib)",
                      "tier1_min": TIER1_MIN, "threshold": EDIT_RATIO_THRESHOLD}


def make_ds(df, tok):
    df = df.copy()
    df["label"] = df["label"].map(LABEL2ID)
    feats = Features({"text": Value("string"), "label": ClassLabel(names=LABEL_LIST)})
    d = Dataset.from_pandas(df[["text", "label"]], features=feats, preserve_index=False)
    return d.map(lambda e: tok(e["text"], max_length=MAXLEN, truncation=True), batched=True)


def compute_metrics(ep):
    pred, lab = ep
    pred = np.argmax(pred, axis=1)
    return {"f1": round(f1m.compute(predictions=pred, references=lab, average="macro")["f1"], 4),
            "accuracy": round(accm.compute(predictions=pred, references=lab)["accuracy"], 4)}


def train_stage0(smsa, tok):
    cfg = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    cfg.num_labels = 3; cfg.label2id = LABEL2ID; cfg.id2label = {i: v for i, v in enumerate(LABEL_LIST)}
    ds_tr = make_ds(smsa["train_filtered"], tok)
    ds_va = make_ds(smsa["valid"], tok)
    collator = DataCollatorWithPadding(tokenizer=tok)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=cfg, ignore_mismatched_sizes=True)

    args = TrainingArguments(
        output_dir=f"{STAGE0_OUT}/run", eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        per_device_train_batch_size=16, per_device_eval_batch_size=64, optim="adamw_torch_fused",
        learning_rate=2e-5, weight_decay=0.01, num_train_epochs=3, warmup_ratio=0.1,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1",
        save_only_model=True,   # full-FT (bukan LoRA): optimizer.pt AdamW momentum+variance ~2.2GB/checkpoint
                                # utk 337M param -- kita cuma butuh bobot akhir (bukan resume training).
                                # Tanpa ini: crash disk-full (optimizer.pt gagal ditulis, lihat handoff).
        bf16=torch.cuda.is_available(), report_to="none", seed=SEED, data_seed=SEED)
    trainer = Trainer(model_init=model_init, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                      processing_class=tok, data_collator=collator, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(2)])
    trainer.train()
    val = trainer.evaluate(ds_va)
    print(f"  stage-0 SmSA val-F1 = {val['eval_f1']*100:.2f}")
    os.makedirs(f"{STAGE0_OUT}/best", exist_ok=True)
    trainer.save_model(f"{STAGE0_OUT}/best"); tok.save_pretrained(f"{STAGE0_OUT}/best")
    import shutil
    d = f"{STAGE0_OUT}/run"
    if os.path.isdir(d):
        for e in os.listdir(d):
            if e.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(d, e), ignore_errors=True)
    return trainer, float(val["eval_f1"])


def gate_zeroshot(trainer, tok):
    """Eval stage-0 zero-shot di 7 NusaX test set (transfer ind->lokal)."""
    print("\n=== GATE zero-shot stage-0 di NusaX test (transfer ind->lokal) ===")
    res = {}
    for lang in SCOPE:
        te = pd.read_csv(f"data/nusax_senti/{lang}/test.csv")
        ds = make_ds(te, tok)
        pred = trainer.predict(ds)
        f1 = f1_score(pred.label_ids, np.argmax(pred.predictions, 1), average="macro") * 100
        res[lang] = round(f1, 2)
        print(f"  {lang}: zero-shot test-F1 = {f1:.2f}")
    print(f"  scope-mean zero-shot = {np.mean(list(res.values())):.2f}")
    return res


def main():
    print(f"CUDA={torch.cuda.is_available()} | E6 STILTs stage-0 (SmSA -> NusaBERT-large)")
    tok = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    smsa = fetch_smsa()
    filtered, filt_stats = leakage_filter(smsa["train"])
    smsa["train_filtered"] = filtered
    os.makedirs(STAGE0_OUT, exist_ok=True)
    json.dump(filt_stats, open(f"{STAGE0_OUT}/leakage_filter_stats.json", "w"), indent=2)

    trainer, s0_val = train_stage0(smsa, tok)
    gate = gate_zeroshot(trainer, tok)
    json.dump({"stage0_smsa_val_f1": s0_val, "leakage_filter": filt_stats, "gate_zeroshot_nusax": gate},
              open(f"{STAGE0_OUT}/stage0_summary.json", "w"), indent=2)
    print(f"\nSTAGE-0 SELESAI -> {STAGE0_OUT}/best")
    print("Gate OK? -> jalankan STAGE-1 (lihat docstring: champion_p1_base + MODEL_CHECKPOINT override).")


if __name__ == "__main__":
    main()
