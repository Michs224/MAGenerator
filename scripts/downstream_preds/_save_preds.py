"""Simpan prediksi test per-kalimat (y_true, y_pred) + per-class + confusion.

Dipakai skrip downstream_preds/* supaya confusion matrix & per-class F1 tersedia
TANPA perlu re-train (finetune biasa hanya menyimpan F1 skalar; prediksi mentah dibuang).

Ditulis ke {output_dir}/predictions.json (sebelah train_history.json tiap bahasa).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def save_test_predictions(trainer, tokenized_test, test_df, label_list, output_dir, lang, seed):
    """test_df: kolom 'id' + 'label' (label SUDAH di-map ke id 0..K-1). label_list: nama urut id."""
    pred = trainer.predict(tokenized_test)
    y_pred = np.argmax(pred.predictions, axis=1).astype(int).tolist()
    y_true = [int(x) for x in test_df["label"].tolist()]
    ids = [int(x) for x in test_df["id"].tolist()]

    rep = classification_report(y_true, y_pred, target_names=label_list,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_list)))).tolist()

    out = {
        "lang": lang,
        "seed": seed,
        "label_list": label_list,               # urutan = indeks id
        "id": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "macro_f1": round(rep["macro avg"]["f1-score"] * 100, 2),
        "per_class": {c: {k: round(rep[c][k], 4)
                          for k in ("precision", "recall", "f1-score", "support")}
                      for c in label_list},
        "confusion": cm,                        # baris = true, kolom = pred (urut label_list)
    }
    path = Path(output_dir) / "predictions.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Predictions saved -> {path}  (macro_f1={out['macro_f1']}, n={len(ids)})")
    return out
