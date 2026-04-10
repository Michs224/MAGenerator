# Strategi Evaluasi NusaSynth - Analisis Bahasa Target

## Konteks

Tesis ini menargetkan 4 bahasa: **Jawa (jav)**, **Sunda (sun)**, **Aceh (ace)**, dan **Batak Toba (bbc)**.
Evaluasi linguistic purity menggunakan NLLB-LID (Meta, 218 bahasa) menunjukkan:

| Bahasa      | NLLB-LID Top-1 Accuracy | Status                |
| ----------- | ------------------------ | --------------------- |
| Jawa (jav)  | ~94.4%                     | Valid sebagai evaluator |
| Sunda (sun) | ~92%                     | Valid sebagai evaluator |
| Aceh (ace)  | ~98%                     | Valid sebagai evaluator |
| Batak Toba  | **0%**                   | **Tidak di-support**  |

Batak Toba **tidak termasuk** dalam 218 bahasa yang dilatih oleh NLLB-LID.
Ini bukan kegagalan model, melainkan bahasa ini memang tidak ada di training data NLLB/FLORES-200.

---

## Kenapa Tetap Pakai Batak Toba?

### 1. Representasi Spektrum Sumber Daya

| Bahasa     | Penutur  | NLLB-LID Support | Rumpun/Wilayah |
| ---------- | -------- | ---------------- | -------------- |
| Jawa       | ~82 juta | Ya (94.4%)         | Jawa |
| Sunda      | ~40 juta | Ya (92%)         | Jawa |
| Aceh       | ~3.5 juta | Ya (98%)        | Sumatra |
| Batak Toba | ~2 juta  | Tidak            | Sumatra |

Keempat bahasa ini memberi variasi: dua bahasa dari Jawa dan dua dari Sumatra,
dengan jumlah penutur yang bervariasi. Batak Toba menjadi kasus khusus dimana
bahkan tool language identification terbesar (NLLB, 218 bahasa) tidak meng-cover-nya.
Menghapus Batak Toba sepenuhnya menghilangkan representasi
extremely low-resource, yang justru menjadi fokus utama tesis.

### 2. Temuan NLLB-LID = Bukti Empiris

Fakta bahwa NLLB-LID (tool terbesar di dunia, 218 bahasa) tidak support Batak Toba
adalah **temuan empiris** yang memperkuat argumen tesis:
- Mengonfirmasi status extremely low-resource secara kuantitatif
- Memotivasi pengembangan metode evaluasi alternatif
- Menunjukkan bahwa tooling NLP untuk bahasa daerah Indonesia masih sangat terbatas

### 3. Downstream F1-Score = Bukti Terkuat

Evaluasi linguistic purity adalah metrik **intermediate** (mengukur kualitas teks itu sendiri).
Downstream F1-Score adalah metrik **ultimate** (mengukur apakah data sintetis berguna).

Logikanya:
- Data sintetis yang terkontaminasi code-mixing → model belajar pola campur kode
  → F1 **turun** pada test set asli (fenomena *capacity dilution*)
- Data sintetis yang murni → model belajar pola bahasa yang benar
  → F1 **naik** pada test set asli

Jika F1 naik, itu bukti **tidak terbantahkan** bahwa data sintetis berkualitas,
terlepas dari metrik intermediate manapun.

---

## Alur Evaluasi Lengkap

### Fase 1: Kalibrasi (Sebelum Generate)

Jalankan semua metrik evaluasi pada **data NusaX asli** untuk mendapatkan baseline.
Data asli sudah divalidasi native speaker, jadi ini menjadi ground truth.

```
Data NusaX Asli (train+valid+test per bahasa)
  ↓
┌──────────────────────────────┐
│ Jalankan metrik evaluasi:    │
│ • NLLB-LID (jav, sun, ace)  │
│ • Statistik teks (panjang,   │
│   distribusi label, dll)     │
│                              │
│ Output: baseline score       │
└──────────────────────────────┘
```

### Fase 2: Generate Data Sintetis (NusaSynth Pipeline)

```
Per bahasa (jav, sun, ace, bbc):

  Orchestrator (Python) iterasi seed 1-500:
    ├─ Pilih 3 few-shot (random, label sama)
    ├─ Contextualizer (LLM): analisis seed → instruksi variasi
    ├─ Generator (LLM): seed + instruksi + few-shot → 3 kalimat
    ├─ Sentiment Validator (LLM): per kalimat, PASS/REJECT
    └─ Linguistic Validator (LLM): per kalimat, PASS/REJECT
       Shared retry counter max 3x per kalimat (SV + LV berbagi)
       Feedback dari kedua validator digabung saat retry

  Output: ~1200-1400 kalimat valid per bahasa + metadata pipeline
```

### Fase 3: Evaluasi Kualitas Data Sintetis

```
Data Sintetis
  ↓
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  A. Linguistic Purity Rate (NLLB-LID)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ JAWA, SUNDA, ACEH:                                      │  │
│  │   • Jalankan NLLB-LID pada setiap kalimat sintetis      │  │
│  │   • Hitung % yang diprediksi sebagai bahasa target      │  │
│  │   • Bandingkan dengan baseline data asli:               │  │
│  │     jav 95%, sun 93%, ace 98%                           │  │
│  │                                                          │  │
│  │ BATAK TOBA:                                             │  │
│  │   • NLLB-LID tidak support bahasa ini                   │  │
│  │   • Kualitas dinilai via downstream F1-Score saja       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  B. Rasio Penyaringan (Filter Rate)                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Dari metadata pipeline NusaSynth:                        │  │
│  │   • % sampel ditolak Sentiment Validator                │  │
│  │   • % sampel ditolak Linguistic Validator               │  │
│  │   • Rata-rata retry count per sampel                    │  │
│  │   • % sampel didiskualifikasi (gagal setelah max retry) │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Fase 4: Evaluasi Downstream Task

```
┌──────────────────────────────────────────────────────────────┐
│ Fine-tune IndoBERT (dan/atau XLM-R) per bahasa:              │
│                                                              │
│ Skenario A: 500 data asli (baseline)                         │
│ Skenario B: 500 data asli + 500 sintetis (1x augmentasi)    │
│ Skenario C: 500 data asli + 1000 sintetis (2x augmentasi)   │
│ Skenario D: 500 data asli + 1500 sintetis (3x augmentasi)   │
│                                                              │
│ Test set: 400 data NusaX asli per bahasa                     │
│ Metrik: F1-Score (macro), Accuracy                           │
│                                                              │
│ Harapan: F1 naik dari A → B → C (→ D mungkin saturasi)      │
└──────────────────────────────────────────────────────────────┘
```

---

## Ringkasan Keputusan

| Aspek | Jawa, Sunda, Aceh | Batak Toba |
| ----- | ----------------- | ---------- |
| Linguistic Purity | NLLB-LID (93-98%) | Tidak bisa dievaluasi secara otomatis |
| Sentiment Consistency | Sentiment Validator Agent | Sentiment Validator Agent |
| Filter Rate & Pipeline Metrics | Ya | Ya |
| **Downstream F1 (Ultimate)** | **Ya** | **Ya (satu-satunya bukti kualitas)** |

Untuk Batak Toba, downstream F1-Score menjadi **satu-satunya bukti kualitas data**:
- F1 naik → data sintetis berkualitas (model belajar pola Batak Toba yang benar)
- F1 turun → data terkontaminasi (capacity dilution akibat code-mixing)

Evaluasi linguistic purity otomatis untuk Batak Toba tidak dimungkinkan karena
NLLB-LID tidak support bahasa ini (tidak ada di 218 bahasa NLLB/FLORES-200).

**Catatan**: Keputusan untuk mempertahankan atau menghapus Batak Toba dari target
penelitian perlu didiskusikan dengan dosen pembimbing.
