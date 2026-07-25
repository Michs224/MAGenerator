# downstream_preds — varian finetune yang MENYIMPAN prediksi

Salinan **faithful** dari `scripts/finetune_*.py` (resep training identik → angka F1 sama persis)
dengan tambahan: menyimpan **prediksi test per-kalimat** ke `predictions.json`. Berguna karena
skrip finetune biasa hanya menyimpan F1 skalar dan **membuang** pasangan (y_true, y_pred), sehingga
confusion matrix / per-class F1 butuh re-train. Skrip ini menghindari itu untuk run berikutnya.

## File
- `_save_preds.py` — helper bersama (satu-satunya logika baru): tulis `predictions.json`.
- `nusabert_base_preds.py`  — = `finetune_nusabert_base.py` + simpan prediksi.
- `nusabert_large_preds.py` — = `finetune_nusabert_large.py` + simpan prediksi.
- `xlmr_preds.py`           — = `finetune_xlmr.py` + simpan prediksi (base/large via `XLMR_SIZE`).

## Cara pakai (dari root project, sama seperti skrip asli)
```
uv run python scripts/downstream_preds/nusabert_base_preds.py
uv run python scripts/downstream_preds/nusabert_large_preds.py
set XLMR_SIZE=base  && uv run python scripts/downstream_preds/xlmr_preds.py
set XLMR_SIZE=large && uv run python scripts/downstream_preds/xlmr_preds.py
```
Toggle `SYNTH_SOURCE` (`full`/`single_llm`) & `USE_AUGMENTED` sama seperti skrip asli.
Output ke **`outputs/downstream_<source>_preds/`** (TERPISAH dari hasil utama `downstream_<source>/`,
supaya tak resume-skip / menimpa) — isinya `results_summary.json` + `predictions.json` per bahasa.
Hasil downstream utama TIDAK tersentuh.

## Output: `{out_base}/<model>-<lang>/predictions.json`
```
{ "lang", "seed", "label_list": ["negative","neutral","positive"],
  "id": [...], "y_true": [...], "y_pred": [...],   # sejajar, urut test.csv
  "macro_f1": 87.32,
  "per_class": { "<label>": {precision, recall, f1-score, support} },
  "confusion": [[...],[...],[...]] }               # baris=true, kolom=pred (urut label_list)
```

## Caveat
- **Folder terpisah** (`downstream_<source>_preds/`, kosong di awal) → run pertama melatih SEMUA
  bahasa dari nol + simpan `predictions.json`, tanpa menyentuh `downstream_<source>/` hasil utama.
  Resume tetap berlaku DI DALAM folder preds (re-run skip bahasa yang sudah ada di situ).
- Tidak mengubah pipeline / skrip asli sama sekali (folder terpisah, bisa dihapus kapan saja).
