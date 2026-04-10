# NusaSynth - Task Breakdown

## Keputusan yang Sudah Final

- **Judul**: NusaSynth: A Multi-Agent Framework for Synthetic Sentiment Data Generation
  in Indonesian Low-Resource Languages (Javanese, Sundanese, Acehnese, and Toba Batak)
- **Bahasa target**: jav, sun, ace, bbc (tanpa bahasa Indonesia)
- **Prompt**: Bahasa Indonesia/Inggris → Output bahasa daerah target
- **Evaluasi linguistic purity**: NLLB-LID untuk jav/sun/ace, downstream F1 only untuk bbc
- **Downstream model**: IndoBERT dan/atau XLM-R
- **Skenario augmentasi**: Baseline (500 asli), +1x (500 sintetis), +2x (1000), +3x (1500)
- **LLM backbone**: Keluarga model Gemini (komersial, via API)

---

## Arsitektur Multi-Agent (Final)

### Komponen

```
Orchestrator (Python) → Contextualizer (LLM) → Generator (LLM) → SV (LLM) → LV (LLM)
```

| Komponen | Tipe | Tugas |
|---|---|---|
| **Orchestrator** | Python | Iterasi seed, pilih few-shot, tracking, retry logic |
| **Contextualizer Agent** | LLM | Analisis seed → identifikasi aspek-aspek yang bisa divariasikan |
| **Generator Agent** | LLM | Generate N kalimat sintetis per seed (N configurable) |
| **Sentiment Validator** | LLM | Cek konsistensi label per kalimat |
| **Linguistic Validator** | LLM | Cek kemurnian bahasa per kalimat |

### Alur Pipeline (per bahasa)

```
Orchestrator iterasi seed 1 sampai 500:
│
├─ Pilih 3 few-shot (random.sample dari pool seed, label sama, beda dari seed saat ini)
│
├─ Contextualizer Agent
│   Input: 1 seed text + label + bahasa target
│   Tugas: Analisis seed → identifikasi aspek-aspek yang bisa divariasikan
│          (Contextualizer sendiri yang tentukan aspek apa dan berapa banyak,
│           minimal 3, sesuaikan dengan kekayaan konteks seed)
│   Output: analisis singkat + daftar aspek variasi (adaptif per seed)
│
├─ Generator Agent
│   Input: seed + output Contextualizer + few-shot + constraint bahasa
│   Output: N kalimat sintetis (N = configurable, misal 3)
│   Generator pilih dari aspek variasi yang diberikan Contextualizer
│
├─ Validasi per kalimat (shared retry counter, max 3x total per kalimat):
│   │
│   ├─ Sentiment Validator
│   │   Input: 1 kalimat + target label
│   │   Output: PASS/REJECT + feedback
│   │   Jika REJECT → retry ke Generator (counter +1)
│   │
│   └─ Linguistic Validator (hanya jika SV PASS)
│       Input: 1 kalimat + target bahasa
│       Output: PASS/REJECT + feedback
│       Jika REJECT → retry ke Generator (counter +1)
│       Jika PASS → simpan ke dataset
│
│   Retry: feedback dari KEDUA validator digabung ke Generator
│   agar Generator perbaiki semua masalah sekaligus (hindari ping-pong)
│   Jika counter = 3 dan masih gagal → DISCARD + log metadata
```

### Kenapa Arsitektur Ini?

1. **Iterasi semua 500 seed secara berurutan** (bukan random sampling)
   - Semua seed pasti kena, tidak ada yang terlewat
   - Distribusi label otomatis terjaga (192 neg, 119 neu, 189 pos)

2. **Few-shot random per iterasi** dari pool seed (label sama)
   - Memberi contoh gaya bahasa yang bervariasi tiap iterasi
   - Generator lihat referensi berbeda → output berbeda

3. **Contextualizer adaptif** (bukan predefined transformation)
   - Membaca seed → menentukan sendiri aspek apa saja yang bisa divariasikan
   - Jumlah aspek menyesuaikan kekayaan konteks seed (minimal 3)
   - Seed kaya detail → lebih banyak aspek variasi
   - Seed simpel → lebih sedikit tapi tetap cukup
   - Instruksinya berbeda per seed karena konteksnya berbeda

4. **N output per seed** (configurable, default 3)
   - 500 × 3 = 1500 kalimat mentah
   - Setelah validator filter: ~1200-1400 kalimat valid
   - Skenario +1x: ambil 500 pertama (1 per seed)
   - Skenario +2x: ambil 1000 (2 per seed)
   - Skenario +3x: ambil semua
   - Jumlah bisa diubah ke 5 tanpa mengubah arsitektur

5. **Validator per kalimat** (bukan batch)
   - Kualitas kontrol ketat
   - Retry individual, bukan regenerasi semua 3

6. **Tidak perlu dedup** (post-hoc similarity check tidak diperlukan)
   - 500 seed unik + random few-shot + Contextualizer adaptif = duplikasi sangat unlikely
   - Validator sudah cukup sebagai quality gate

---

## Task 1: Persiapan Data

### 1.1 Download & Eksplorasi NusaX-Senti
- [x] Download dataset via HuggingFace datasets
- [x] Eksplorasi distribusi label, panjang teks, karakteristik per bahasa
- [x] Konfirmasi data paralel (ID sama = makna sama antar bahasa)

### 1.2 Pre-Analysis Seed Data (Python, bukan LLM)
- [ ] Per bahasa (jav, sun, ace, bbc), hitung:
  - Distribusi label (pos/neu/neg count & %)
  - Statistik panjang teks (mean, std, min, max kata)
- [ ] Siapkan fungsi random few-shot sampler:
  - Input: seed_pool, current_seed_id, target_label, n=3
  - Filter: label sama + beda dari seed saat ini
  - Output: random.sample(candidates, n)

### 1.3 Validasi NLLB-LID
- [x] Install fasttext + download NLLB-LID model
- [x] Jalankan pada data NusaX asli: jav 95%, sun 93%, ace 98%, bbc 0%
- [x] Konfirmasi bbc tidak ada di NLLB/FLORES-200

---

## Task 2: Implementasi Multi-Agent Pipeline

### 2.1 Orchestrator (Python)
- [ ] Load seed data per bahasa
- [ ] Implementasi loop: iterasi semua 500 seed berurutan
- [ ] Implementasi few-shot sampler (random, label sama)
- [ ] Retry logic: per kalimat, max 3x, lalu discard
- [ ] Metadata logging: seed_id, retry_count, sv_result, lv_result per kalimat
- [ ] Progress tracking & checkpointing (bisa resume jika crash)
- [ ] Counter tracking: berapa pos/neu/neg sudah valid

### 2.2 Contextualizer Agent (LLM)
- [ ] Desain system prompt:
  - Tugas: analisis seed → identifikasi aspek-aspek yang bisa divariasikan
  - Tentukan sendiri berapa banyak aspek variasi (minimal 3, sesuai kekayaan seed)
  - Output: analisis singkat + daftar aspek variasi
- [ ] Desain user prompt:
  - Input: 1 seed text + label + bahasa target
  - Output: adaptif per seed (LLM decide aspek apa dan berapa banyak)
- [ ] TIDAK hardcode jumlah variasi atau transformation type

### 2.3 Generator Agent (LLM)
- [ ] Desain system prompt per bahasa target:
  - Constraint bahasa murni (DILARANG code-mixing)
  - N kalimat harus berbeda satu sama lain (N configurable)
- [ ] Desain user prompt per iterasi:
  - Seed text + output Contextualizer + few-shot examples
  - Generator pilih dari aspek variasi yang diberikan Contextualizer
  - Jika retry: tambahkan feedback dari validator
- [ ] Output: N kalimat sintetis (default 3)

### 2.4 Sentiment Validator Agent (LLM)
- [ ] Desain prompt: terima 1 kalimat + target label
- [ ] Output: PASS/REJECT + feedback spesifik jika reject
- [ ] Dijalankan per kalimat (bukan batch)

### 2.5 Linguistic Validator Agent (LLM)
- [ ] Desain prompt: terima 1 kalimat + target bahasa
- [ ] Cek: code-mixing, kealamian gramatikal, token asing
- [ ] Output: PASS/REJECT + token asing yang ditemukan jika reject
- [ ] Dijalankan per kalimat (bukan batch)

### 2.6 Retry Mechanism
- [ ] Shared retry counter: max 3x total per kalimat (SV + LV berbagi counter)
- [ ] Feedback dari kedua validator DIGABUNG ke Generator saat retry
- [ ] Mencegah ping-pong (perbaiki bahasa → rusak sentimen → ulang)
- [ ] Jika counter = 3 dan masih gagal → DISCARD + log metadata

---

## Task 3: Generasi Data Sintetis

### 3.1 Eksekusi Pipeline
- [ ] Jalankan pipeline per bahasa: jav, sun, ace, bbc
- [ ] Target per bahasa: 1500 data sintetis (untuk skenario +3x)
- [ ] Distribusi label: seimbang atau mengikuti distribusi asli
- [ ] Monitor filter rate selama generasi
- [ ] Simpan dataset sintetis + metadata

### 3.2 Quality Check Awal
- [ ] Cek distribusi label output vs target
- [ ] Cek statistik panjang teks (tidak terlalu pendek/panjang)
- [ ] Spot check manual beberapa sampel per bahasa

---

## Task 4: Evaluasi Kualitas Data Sintetis (Bab IV bagian A)

### 4.1 Filter Rate & Pipeline Metrics
- [ ] Tabel per bahasa: acceptance rate, rejection rate (SV & LV), avg retry, disqualified rate
- [ ] Analisis: bahasa mana yang paling sulit di-generate?

### 4.2 Konsistensi Label Sentimen
- [ ] Distribusi label data sintetis vs data asli per bahasa
- [ ] Sentiment Validator pass rate

### 4.3 Linguistic Purity Rate (NLLB-LID)
- [ ] Jalankan NLLB-LID pada data sintetis jav, sun, ace
- [ ] Bandingkan dengan baseline data asli (jav 95%, sun 93%, ace 98%)
- [ ] Histogram confidence distribution: asli vs sintetis
- [ ] Error analysis: contoh kalimat yang terdeteksi code-mixing
- [ ] Untuk bbc: tulis bahwa NLLB-LID tidak support → evaluasi via downstream F1

---

## Task 5: Evaluasi Downstream Task (Bab IV bagian B - INTI)

### 5.1 Fine-tuning Setup
- [ ] Pilih model: IndoBERT (dan/atau XLM-R)
- [ ] Setup hyperparameter: learning rate, batch size, epochs, early stopping
- [ ] Pastikan setup SAMA untuk semua skenario (fair comparison)

### 5.2 Eksperimen Utama
Per bahasa (jav, sun, ace, bbc) x per model:
- [ ] Skenario A (Baseline): train 500 data asli → test 400 NusaX
- [ ] Skenario B (+1x): train 500 asli + 500 sintetis → test 400 NusaX
- [ ] Skenario C (+2x): train 500 asli + 1000 sintetis → test 400 NusaX
- [ ] Skenario D (+3x): train 500 asli + 1500 sintetis → test 400 NusaX
- [ ] Catat: F1-Score (macro), Accuracy, Per-class F1
- [ ] Total: 4 skenario × 4 bahasa × 1-2 model = 16-32 eksperimen

### 5.3 Analisis Hasil
- [ ] Tabel F1-Score per skenario per bahasa
- [ ] Line chart: F1 vs volume augmentasi (apakah naik/saturasi?)
- [ ] Confusion matrix skenario terbaik per bahasa
- [ ] Perbandingan antar bahasa: bahasa mana yang paling diuntungkan?
- [ ] Analisis Batak Toba: apakah F1 naik? (bukti kualitas data sintetis)

---

## Task 6: Analisis & Pembahasan (Bab IV bagian C)

### 6.1 Analisis Feedback Loop
- [ ] Efektivitas mekanisme feedback (apakah kualitas membaik setelah retry?)
- [ ] Alasan rejection paling umum per bahasa
- [ ] Perbandingan kualitas iterasi 1 vs iterasi 2-3

### 6.2 Error Analysis
- [ ] Contoh kasus Generator gagal → diperbaiki oleh feedback
- [ ] Contoh kasus yang lolos validator tapi sebenarnya kurang bagus
- [ ] Diskusi limitasi

---

## Evaluasi Optional (Jika Waktu Cukup)

- [ ] Distributional similarity: panjang teks, vocabulary overlap asli vs sintetis
- [ ] Human evaluation: sampling + penilaian manual (jika ada akses native speaker)
- [ ] Diversity metrics: distinct n-gram antar kalimat sintetis
