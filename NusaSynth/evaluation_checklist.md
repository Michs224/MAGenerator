# Evaluasi NusaSynth - Checklist Bab IV

## Evaluasi Wajib

### 1. Rasio Penyaringan (Filter Rate)
- Per bahasa (jav, sun, ace, bbc), dari metadata pipeline:
  - Acceptance Rate: % sampel langsung lolos kedua validator
  - Rejection Rate Sentiment Validator
  - Rejection Rate Linguistic Validator
  - Avg Retry Count per sampel
  - Disqualified Rate: % gagal setelah max retry

### 2. Konsistensi Label Sentimen
- Sentiment Validator Pass Rate (dari filter rate di atas)
- Tabel distribusi label (pos/neu/neg) data sintetis vs data asli per bahasa

### 3. Linguistic Purity Rate (NLLB-LID)
- Hanya untuk: **jav, sun, ace** (Batak Toba tidak di-support NLLB-LID)
- Metrik: % kalimat sintetis yang dikenali sebagai bahasa target oleh NLLB-LID
- Baseline dari train set data asli NusaX:
  - jav: **94.4%** (472/500 dikenali sebagai Jawa)
  - sun: **92.0%** (460/500 dikenali sebagai Sunda)
  - ace: **98.0%** (490/500 dikenali sebagai Aceh)
  - bbc: **N/A** (Batak Toba tidak ada di 218 bahasa NLLB-LID)
- Tampilkan tabel Bab IV:

  | Bahasa | Data Asli (Baseline) | Data Sintetis |
  |---|---|---|
  | Jawa | 94.4% | ?% |
  | Sunda | 92.0% | ?% |
  | Aceh | 98.0% | ?% |
  | Batak Toba | N/A | N/A |

  Jika angka sintetis mendekati baseline → kualitas linguistik setara data asli.
- Tampilkan juga: histogram distribusi + contoh kalimat yang terdeteksi code-mixing
- Untuk **bbc**: tulis bahwa evaluasi linguistic purity otomatis tidak dimungkinkan,
  kualitas dinilai via downstream F1-Score

### 4. Downstream F1-Score (UTAMA)
- Fine-tune IndoBERT dan XLM-R per bahasa
- Skenario:
  - Baseline: 500 data asli
  - +1x: 500 asli + 500 sintetis
  - +2x: 500 asli + 1000 sintetis
  - +3x: 500 asli + 1500 sintetis
- Test set: 400 data NusaX asli per bahasa
- Metrik: F1-Score (macro), Accuracy, Per-class F1
- Tampilkan: tabel F1 + line chart + confusion matrix + perbandingan IndoBERT vs XLM-R
- Total: 4 skenario x 4 bahasa x 2 model = 32 eksperimen

### 5. Analisis Feedback Loop
- Efektivitas mekanisme feedback loop
- Alasan rejection paling umum per bahasa
- Apakah kualitas meningkat setelah regenerasi

---

## Evaluasi Optional

### 6. Distributional Similarity (Data Asli vs Sintetis)
- Panjang teks: mean & std karakter/kata, asli vs sintetis
- Distribusi label: proporsi pos/neu/neg asli vs sintetis
- Vocabulary overlap: % kata sintetis yang juga ada di data asli
- Format: tabel + boxplot per bahasa
