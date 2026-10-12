"""Uji kemampuan Gemini membaca sentimen di NusaX-Senti, dengan klien PERSIS SV produksi.

PERTANYAAN: apakah gemini-3.1-flash-lite (config SV produksi) andal membaca sentimen
            bahasa-bahasa target, TANPA sinyal NusaBERT dan TANPA prompt SV?

Ini validasi pemilihan MODEL, bukan evaluasi AGEN SV:
  - Klien Gemini PERSIS klien SV produksi (model, temperature, thinking, Vertex, retry parse).
    Nilai diambil dari NusaSynth.config lalu DIKUNCI: kalau berbeda dari produksi, skrip menolak jalan.
  - Prompt BUKAN prompt SV (prompt SV = verifikasi ya/tidak + sinyal NusaBERT + seed).
  - Satu kalimat per panggilan.

PROTOKOL FINAL (dipakai untuk TEST): 6 template prompt dari literatur, diambil VERBATIM dari PDF-nya
(L1-L3 SEACrowd, L4 Zhang dkk. 2024, L5 LexC-Gen, L6 BHASA), zero-shot, tanpa system prompt, persona,
definisi, maupun contoh. SEMUA template dilaporkan, plus rata-rata/SD/rentangnya (--aggregate),
seperti SEACrowd/NusaCrowd/Koto dkk. 2024. Template terbaik di pilot VALID (L4) boleh disebut
sebagai pembanding, tidak dipilih dari hasil test. Satu-satunya adaptasi: jawaban dibaca lewat keluaran
terstruktur (pembanding teks bebas L4_teks/L6_teks di valid: 96-98% jawaban identik).

Varian v1-v7 di bawah = jejak eksplorasi di pilot VALID (tidak dipakai untuk test):
VARIAN PROMPT (tiap varian mengubah SATU komponen dari v1, supaya atribusinya bersih):
  v1  dasar: anotator buta, diadaptasi dari exp_ablation_final.py (batch -> 1 kalimat)
  v2  v1 + pedoman umum definisi kelas (sentimen = evaluasi penulis, bukan baik-buruknya topik;
      tanpa evaluasi = netral; evaluasi bisa tersirat; campuran -> yang dominan, seimbang -> netral)
  v3  v1 + few-shot dari split TRAIN (3 contoh per label, acak berbiji tetap, bahasa yang sama)
  v4  v1 + alasan singkat sebelum label (seperti field CoT keluaran SV produksi)
  --- ronde 2 (setelah ronde 1 di pilot valid: v2 & v3 membantu, v4 tidak) ---
  v5  v2 + v3 (pedoman + 3 contoh/label)
  v6  v5 dengan 5 contoh/label
  v7  v5 + ambang gaya SV: polaritas hanya bila evaluasi penulis jelas, selain itu netral

DISIPLIN SPLIT (sama dengan seleksi champion NusaBERT, by-VAL):
  1. Cari varian terbaik di split VALID (default). Boleh diulang.
  2. Bekukan varian terpilih, commit.
  3. Jalankan split TEST SEKALI (--split test). Angka test tidak dipakai mengubah prompt.

Metrik (ditetapkan sebelum test): akurasi 3 kelas, macro-F1, kappa Cohen, presisi/recall/F1 per kelas
(recall per kelas = "akurasi per kelas"), akurasi kalimat pos/neg + presisi prediksi pos/neg, confusion
matrix, tingkat pembalikan polaritas (negatif<->positif). CI95 gabungan = bootstrap klaster per kalimat
sumber (7 bahasa = terjemahan kalimat yang sama). Banding: NusaBERT-large champion SV, split yang sama.

Jalankan TEST (sekali), lalu agregat:
  for v in L1 L2 L3 L4 L5 L6: uv run python scripts/gemini_zero_shot_nusax.py --variant $v --split test --workers 16
  uv run python scripts/gemini_zero_shot_nusax.py --split test --aggregate

Output: outputs/gemini_zero_shot/<split>/<varian>/
  predictions.jsonl  sumber utama, ditulis PER KALIMAT begitu jawaban masuk (aman bila proses putus; resume)
  predictions.csv    sama, untuk Excel/spreadsheet (utf-8-sig)
  hasil.xlsx         sheet ringkasan / confusion / prediksi / salah / prompt_config
  summary.json       semua metrik + config + commit git
  prompts_used.json  system prompt persis per bahasa + human prompt
  logs/run_*.log     salinan seluruh output konsol tiap run
  usage/<run_id>/    token per panggilan (biaya)

Jalankan dari root:
  uv run python scripts/gemini_zero_shot_nusax.py --variant v2 --dry-run
  uv run python scripts/gemini_zero_shot_nusax.py --variant v2 --show-errors 20   # VALID
  uv run python scripts/gemini_zero_shot_nusax.py --compare v1 v2 v3 v4            # bandingkan di VALID
  uv run python scripts/gemini_zero_shot_nusax.py --variant <terpilih> --split test # TEST, sekali
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from NusaSynth.config import (DATA_DIR, GEMINI_LOCATION, GEMINI_MODEL, GEMINI_PROJECT,
                              GEMINI_TEMP_VALIDATOR, GEMINI_THINK_VALIDATOR, GEMINI_USE_VERTEXAI,
                              LABEL_ORDER, LANG_NAMES, PARSE_MAX_RETRY)

LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
SPLIT_FILE = {"valid": "valid.csv", "test": "test.csv"}
NUSABERT_KEY = {"valid": "val_f1", "test": "test_f1"}
NUSABERT_SUMMARY = ROOT / "outputs" / "pissa-drop06-full" / "seed_42" / "results_summary.json"
OUTER_RETRY_WAIT = (30, 60, 120)  # detik; lapis retry ke-3 (lihat run_predictions)

# Nilai klien SV pada run produksi 2026-07-17_02-40-52. Bukti: model + thinking dari usage log
# (14.917/14.917 call flash-lite, reasoning token 0), temperature dari config.py commit 0234885
# yang tidak ditimpa .env. Skrip menolak jalan kalau config saat ini menyimpang dari sini.
PRODUCTION = {
    "GEMINI_MODEL": "gemini-3.1-flash-lite",
    "GEMINI_TEMP_VALIDATOR": 0.0,
    "GEMINI_THINK_VALIDATOR": "minimal",
    "GEMINI_USE_VERTEXAI": True,
    "GEMINI_LOCATION": "global",
    "PARSE_MAX_RETRY": 2,
}

BASE_PROMPT = ("You are a sentiment annotator for {lang_name} text. Classify the sentiment of the "
               "sentence as negative, neutral, or positive. Judge only the sentence itself.")
# Pedoman umum (definisi kelas), BUKAN aturan per kalimat. Inti: sentimen = evaluasi penulis,
# bukan baik-buruknya topik (pembedaan subjektif vs objektif, Liu 2012).
GUIDELINES = ("\n\nGuidelines:\n"
              "- Sentiment is the writer's own evaluation, not whether the topic is good or bad news. "
              "A report of a pleasant or unpleasant event without the writer's evaluation is neutral.\n"
              "- Neutral also covers sentences with no evaluation, such as a factual statement, a "
              "question, a request, an offer, or an announcement.\n"
              "- An evaluation can be stated directly or implied by how the writer describes their "
              "own experience.\n"
              "- If both favorable and unfavorable evaluations appear, choose the one that dominates. "
              "If neither dominates, choose neutral.")
# Ambang gaya SV produksi ("CLEARLY and UNAMBIGUOUSLY"): polaritas hanya bila evaluasinya jelas.
STRICT_LINE = ("\n- Choose positive or negative only when the writer clearly expresses that evaluation. "
               "If the evaluation is absent or unclear, choose neutral.")
HUMAN_PROMPT = "Sentence: {text}"

# ── Template prompt dari LITERATUR (dibaca langsung dari PDF-nya) ─────────────────────────────────
# Diambil apa adanya: tanpa system prompt, tanpa definisi kelas, tanpa contoh. Satu-satunya adaptasi:
# slot jawaban ([LABEL_CHOICE]) dikosongkan karena Gemini menjawab lewat teks, bukan skor log-likelihood;
# jawabannya dibaca dari keluaran terstruktur (setara "string matching for commercial baselines",
# SEACrowd hal. 30). [OPTIONS] diisi daftar label NusaX.
OPTIONS = "negative, neutral, positive"
LIT_TEMPLATES = {
    # SEACrowd (Lovenia dkk., 2024), Tabel 23 hal. 34. Zero-shot, 3 template per tugas NLU (§3.3 hal. 5).
    "L1": ("SEACrowd #1 (Lovenia dkk., 2024)",
           "Classify the sentiment of the text below.\n{text} => Sentiment ({options}):"),
    "L2": ("SEACrowd #2 (Lovenia dkk., 2024)",
           "Predict the sentiment of the following text.\nText: {text}\nAnswer with {options}:"),
    "L3": ("SEACrowd #3 (Lovenia dkk., 2024)",
           "{text}\nWhat would be the sentiment of the text above? {options}?"),
    # Zhang dkk. (2024), Lampiran A.1 hal. 20 (SC Twitter, 3 kelas); contoh 1-shot dibuang = setelan
    # zero-shot mereka (Figure 1 hal. 7: "would be removed under the zero-shot setting").
    "L4": ("Zhang dkk. (2024)",
           ("Please perform Sentiment Classification task. Given the sentence, assign a sentiment label "
            "from ['negative', 'neutral', 'positive']. Return label only without any other text.\n"
            "Sentence: {text}\nLabel:")),
    # Yong dkk. (2024, LexC-Gen), Lampiran E hal. 16: prompt GPT-4o zero-shot di NusaX (urutan label asli).
    "L5": ("Yong dkk. (2024) LexC-Gen",
           "Does this sentence have positive, negative, neutral sentiment? {text}"),
    # BHASA (Leong dkk., 2023), hal. 66: prompt Inggris untuk NusaX (id), zero-shot, LLM generatif.
    "L6": ("BHASA (Leong dkk., 2023)",
           ("What is the sentiment of the following sentence? {text}\n"
            "Answer only with a single word: Positive/Negative/Neutral")),
}
LIT_SET = list(LIT_TEMPLATES)  # semua template literatur; dilaporkan semua + rata-ratanya

# Zhang (L4) + definisi tugas minimal, masih dalam kerangka Zhang sendiri (§4.2 arXiv v1 hal. 7:
# "The task definition is constructed based on each task's definition and annotation guidelines").
# Eksplorasi di VALID, satu klausa per varian; BUKAN bagian rata-rata template literatur (LIT_SET).
# DITOLAK (tidak dipakai di test): klausa "use 'neutral' when..." memberi aturan untuk SATU kelas saja
# (tidak simetris) dan terbukti menggeser jawaban ke netral (tebakan netral L4 48 -> def1 58 -> def2 60
# dari 210; neg 79->72, pos 83->78). Kenaikan akurasi def2 +1,9 tidak bermakna (McNemar p=0,45),
# perkiraan di proporsi test hanya +0,6. Prinsip: gaya prompt tetap bawaan; pengayaan harus adil 3 kelas.
ZHANG_DEF_TEMPLATES = {
    "L4_def1": ("Zhang + definisi pendapat penulis",
                ("Please perform Sentiment Classification task. Given the sentence, assign a sentiment label "
                 "from ['negative', 'neutral', 'positive'] based on the opinion expressed by the writer; use "
                 "'neutral' when the writer expresses no opinion. Return label only without any other text.\n"
                 "Sentence: {text}\nLabel:")),
    "L4_def2": ("Zhang + definisi + bukan baik-buruknya topik",
                ("Please perform Sentiment Classification task. Given the sentence, assign a sentiment label "
                 "from ['negative', 'neutral', 'positive'] based on the opinion expressed by the writer, not on "
                 "whether the topic is good or bad news; use 'neutral' when the writer expresses no opinion. "
                 "Return label only without any other text.\nSentence: {text}\nLabel:")),
    # Pengayaan ADIL (usul user): ketiga label didefinisikan sejajar & setara dalam gaya Zhang; netral =
    # "bukan keduanya" (tanpa kriteria tambahan). Aturan lolos (ditetapkan SEBELUM hasil): akurasi 3 kelas
    # >= L4 + 3 poin, akurasi pos/neg tidak turun, sebaran tebakan tidak condong ke satu kelas.
    # HASIL pilot VALID: TIDAK LOLOS (3 kelas 83,33 vs L4 83,81 = -0,48; pos/neg 93,6 vs 95,7; sebaran
    # tidak condong, geser maks 5/210). Kesimpulan: definisi/pilihan "if..." tidak menambah pemahaman,
    # hanya menggeser keputusan -> prompt test = template literatur apa adanya (L4 Zhang terbaik di valid).
    "L4_adil": ("Zhang + definisi setara 3 label",
                ("Please perform Sentiment Classification task. Given the sentence, assign a sentiment label "
                 "from ['negative', 'neutral', 'positive']: 'negative' if the sentence expresses an unfavorable "
                 "opinion, 'positive' if it expresses a favorable opinion, and 'neutral' if it expresses neither. "
                 "Return label only without any other text.\nSentence: {text}\nLabel:")),
}
# Pengayaan UMUM tanpa kondisi (arahan user): Zhang asli + NAMA BAHASA saja, tanpa definisi/pilihan label.
# Preseden: Rathje dkk. 2024 PNAS ("Is the sentiment of this (Arabic/Swahili/…) text positive, neutral, or
# negative?") dan AfroBench Findings ACL 2025 ("Does this {{ language }} statement ..."). Satu perubahan.
ZHANG_LANG_TEMPLATES = {
    "L4_bahasa": ("Zhang + nama bahasa",
                  ("Please perform Sentiment Classification task. Given the {language} sentence, assign a "
                   "sentiment label from ['negative', 'neutral', 'positive']. Return label only without any "
                   "other text.\nSentence: {text}\nLabel:")),
}
# Pengayaan UMUM kedua tanpa kondisi (arahan user): Zhang asli + "according to its overall sentiment
# polarity" = definisi tugas versi Zhang sendiri (hal. 3: SC "aims at determining the overall sentiment
# polarity of a given text"); frasa sama dipakai di prompt FreeAL (EMNLP 2023, hal. 16). Satu klausa.
ZHANG_OVERALL_TEMPLATES = {
    "L4_overall": ("Zhang + sentimen keseluruhan",
                   ("Please perform Sentiment Classification task. Given the sentence, assign a sentiment label "
                    "from ['negative', 'neutral', 'positive'] according to its overall sentiment polarity. "
                    "Return label only without any other text.\nSentence: {text}\nLabel:")),
}
ALL_TEMPLATES = {**LIT_TEMPLATES, **ZHANG_DEF_TEMPLATES, **ZHANG_LANG_TEMPLATES, **ZHANG_OVERALL_TEMPLATES}

# Mode TEKS BEBAS (tanpa structured output): hanya untuk menguji apakah format terstruktur mengubah
# jawaban. Label dibaca dengan pencocokan string seperti SEACrowd untuk model komersial (hal. 30).
LABEL_WORD = re.compile(r"\b(negative|neutral|positive)\b", re.IGNORECASE)

VARIANTS = {
    "v1": {"guidelines": False, "fewshot": 0, "reasoning": False, "strict": False},
    "v2": {"guidelines": True,  "fewshot": 0, "reasoning": False, "strict": False},
    "v3": {"guidelines": False, "fewshot": 3, "reasoning": False, "strict": False},
    "v4": {"guidelines": False, "fewshot": 0, "reasoning": True,  "strict": False},
    "v5": {"guidelines": True,  "fewshot": 3, "reasoning": False, "strict": False},
    "v6": {"guidelines": True,  "fewshot": 5, "reasoning": False, "strict": False},
    "v7": {"guidelines": True,  "fewshot": 3, "reasoning": False, "strict": True},
    # Uji sensitivitas v5 terhadap PILIHAN contoh (biji acak lain). Ditetapkan SEBELUM dijalankan:
    # yang dipakai tetap v5 (biji 42); dua ini hanya mengukur seberapa jauh angka bergeser.
    "v5_s43": {"guidelines": True, "fewshot": 3, "reasoning": False, "strict": False, "fs_seed": 43},
    "v5_s44": {"guidelines": True, "fewshot": 3, "reasoning": False, "strict": False, "fs_seed": 44},
    # Protokol FINAL: kelima template literatur, SEMUA dilaporkan + dirata-ratakan (seperti SEACrowd,
    # NusaCrowd, Koto dkk. 2024). Tidak ada template yang dipilih/dibuang berdasarkan hasil.
    **{t: {"template": t, "guidelines": False, "fewshot": 0, "reasoning": False, "strict": False}
       for t in ALL_TEMPLATES},
    # Uji pembanding format: template yang aslinya meminta jawaban teks ("Return label only",
    # "Answer only with a single word"), dijalankan TANPA structured output.
    "L4_teks": {"template": "L4", "free_text": True, "guidelines": False, "fewshot": 0, "reasoning": False,
                "strict": False},
    "L6_teks": {"template": "L6", "free_text": True, "guidelines": False, "fewshot": 0, "reasoning": False,
                "strict": False},
}


def parse_free_text(text: str) -> str:
    """Label diambil bila jawaban menyebut tepat SATU label; selain itu 'unparsed' (dihitung salah,
    final, tidak diulang), sama seperti evaluasi berbasis pencocokan string di literatur."""
    found = {m.lower() for m in LABEL_WORD.findall(text or "")}
    return found.pop() if len(found) == 1 else "unparsed"


def invoke_free_text(llm, messages, *, lang: str, gold: str) -> str | None:
    """Panggilan tanpa structured output. None = gagal API (diulang saat skrip dijalankan lagi)."""
    from NusaSynth.usage_tracker import record_usage
    for attempt in range(PARSE_MAX_RETRY + 1):
        try:
            msg = llm.invoke(messages)
        except Exception as e:
            print(f"ZeroShot-teks WARNING: API error (attempt {attempt + 1}: {type(e).__name__}). Coba lagi.")
            continue
        record_usage(lang=lang, label=gold, agent="zeroshot_text", seed_id=None, attempt=attempt, ok=True,
                     usage=getattr(msg, "usage_metadata", None), model=GEMINI_MODEL)
        return str(msg.text)
    return None


class Prediction(BaseModel):
    sentiment: Literal["negative", "neutral", "positive"] = Field(
        description="The sentiment of the sentence: negative, neutral, or positive.")


class PredictionWithReason(BaseModel):
    reasoning: str = Field(description="One or two sentences explaining the writer's overall sentiment.")
    sentiment: Literal["negative", "neutral", "positive"] = Field(
        description="The sentiment of the sentence: negative, neutral, or positive.")


def out_dir(split: str, variant: str) -> Path:
    return ROOT / "outputs" / "gemini_zero_shot" / split / variant


def check_production_config() -> None:
    current = {
        "GEMINI_MODEL": GEMINI_MODEL,
        "GEMINI_TEMP_VALIDATOR": GEMINI_TEMP_VALIDATOR,
        "GEMINI_THINK_VALIDATOR": GEMINI_THINK_VALIDATOR,
        "GEMINI_USE_VERTEXAI": GEMINI_USE_VERTEXAI,
        "GEMINI_LOCATION": GEMINI_LOCATION,
        "PARSE_MAX_RETRY": PARSE_MAX_RETRY,
    }
    beda = {k: (current[k], v) for k, v in PRODUCTION.items() if current[k] != v}
    if beda:
        for k, (now, prod) in beda.items():
            print(f"  BEDA {k}: sekarang={now!r}  produksi={prod!r}")
        sys.exit("Config Gemini tidak sama dengan produksi. Cek env var / .env, lalu ulangi.")
    print("Config Gemini = produksi (dikunci): " + ", ".join(f"{k}={v!r}" for k, v in current.items()))


def fewshot_examples(lang: str, k: int, seed: int = 42) -> list[tuple[str, str]]:
    """k contoh per label dari split TRAIN (acak berbiji tetap), diselang-seling neg/neu/pos."""
    tr = pd.read_csv(DATA_DIR / lang / "train.csv")
    per = {lab: tr[tr["label"] == lab].sample(n=k, random_state=seed)["text"].tolist() for lab in LABEL_ORDER}
    return [(per[lab][i], lab) for i in range(k) for lab in LABEL_ORDER]


def system_prompt(variant: str, lang: str) -> str:
    v = VARIANTS[variant]
    if v.get("template"):
        return ""  # template literatur: tanpa system prompt
    s = BASE_PROMPT.format(lang_name=LANG_NAMES[lang])
    if v["guidelines"]:
        s += GUIDELINES
        if v["strict"]:
            s += STRICT_LINE
    if v["fewshot"]:
        ex = "\n".join(f"- {t} => {lab}" for t, lab in fewshot_examples(lang, v["fewshot"], v.get("fs_seed", 42)))
        s += f"\n\nAnnotated examples from the training data:\n{ex}"
    return s


def human_template(variant: str) -> str:
    t = VARIANTS[variant].get("template")
    return ALL_TEMPLATES[t][1] if t else HUMAN_PROMPT


def build_messages(variant: str, text: str, lang: str) -> list[tuple[str, str]]:
    human = ("human", human_template(variant).format(text=text, options=OPTIONS, language=LANG_NAMES[lang]))
    if VARIANTS[variant].get("template"):
        return [human]
    return [("system", system_prompt(variant, lang)), human]


def load_data(split: str, lang: str, per_label: int | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / lang / SPLIT_FILE[split])
    if per_label:
        # NusaX paralel (id & label sama di 7 bahasa). Biji per bahasa beda supaya sampel pilot
        # mencakup banyak kalimat sumber, bukan 1 set kalimat yang diterjemahkan 7 kali.
        seed = 42 + LANGS.index(lang)
        df = pd.concat([g.sample(n=min(per_label, len(g)), random_state=seed)
                        for _, g in df.groupby("label")])
    return df


def load_done(split: str, variant: str) -> dict[tuple[str, str], dict]:
    path = out_dir(split, variant) / "predictions.jsonl"
    done: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["lang"], str(r["id"]))] = r
    return done


def run_predictions(split: str, variant: str, langs: list[str], per_label: int | None, workers: int) -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from NusaSynth.usage_tracker import init_usage_tracker, invoke_structured_tracked

    od = out_dir(split, variant)
    init_usage_tracker(od / "usage")
    # Klien PERSIS SV produksi (NusaSynth/sv_graph.py baris 66-67). Param lain sengaja tidak diisi.
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=GEMINI_TEMP_VALIDATOR,
                                 thinking_level=GEMINI_THINK_VALIDATOR, vertexai=GEMINI_USE_VERTEXAI,
                                 project=GEMINI_PROJECT, location=GEMINI_LOCATION)
    schema = PredictionWithReason if VARIANTS[variant]["reasoning"] else Prediction
    structured_llm = llm.with_structured_output(schema, include_raw=True)

    done = load_done(split, variant)
    jobs = []
    for lang in langs:
        for row in load_data(split, lang, per_label).itertuples(index=False):
            prev = done.get((lang, str(row.id)))
            if prev is not None and (prev["pred"] in LABEL_ORDER or prev["pred"] == "unparsed"):
                continue
            jobs.append((lang, str(row.id), row.text, row.label))
    print(f"[{split.upper()} / {variant}] memproses {len(jobs)} kalimat ({workers} worker), "
          f"sudah ada {sum(1 for r in done.values() if r['pred'] in LABEL_ORDER)}.")
    if not jobs:
        return

    od.mkdir(parents=True, exist_ok=True)
    pred_path = od / "predictions.jsonl"
    lock = threading.Lock()

    free = VARIANTS[variant].get("free_text", False)

    def one(job):
        lang, sid, text, gold = job
        messages = build_messages(variant, text, lang)
        t0 = time.time()
        # Retry 3 lapis: (1) SDK Google, 408/429/5xx jeda 1-16 dtk; (2) wrapper, +2 percobaan;
        # (3) di sini, kalau masih gagal (mis. kuota habis lama), tunggu 30/60/120 dtk lalu ulang.
        # Sisa yang gagal tersimpan pred=None dan diproses lagi saat skrip dijalankan ulang.
        for wait in (*OUTER_RETRY_WAIT, None):
            if free:
                res = invoke_free_text(llm, messages, lang=lang, gold=gold)
            else:
                res = invoke_structured_tracked(structured_llm, messages,
                                                agent_display="ZeroShot", fail_consequence="Kalimat dilewati.",
                                                lang=lang, label=gold, model=GEMINI_MODEL,
                                                parse_max_retry=PARSE_MAX_RETRY)
            if res is not None or wait is None:
                break
            print(f"  [{lang} {sid}] gagal, tunggu {wait} dtk lalu ulang")
            time.sleep(wait)
        if free:
            pred = parse_free_text(res) if res is not None else None
        else:
            pred = res.sentiment if res is not None else None
        rec = {"lang": lang, "id": sid, "text": text, "gold": gold, "pred": pred,
               "latency_s": round(time.time() - t0, 2)}
        if free and res is not None:
            rec["raw_text"] = res
        if res is not None and VARIANTS[variant]["reasoning"]:
            rec["reasoning"] = res.reasoning
        with lock:
            with open(pred_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result()["pred"] not in LABEL_ORDER:
                n_fail += 1
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} selesai (gagal parse: {n_fail})")


def macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for c in LABEL_ORDER:
        tp = np.sum((gold == c) & (pred == c))
        fp = np.sum((gold != c) & (pred == c))
        fn = np.sum((gold == c) & (pred != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) * 100


def boot_samples(n: int, groups, n_boot: int, rng) -> list[np.ndarray]:
    """Indeks bootstrap. Dengan groups (id kalimat sumber) dipakai bootstrap KLASTER: ketujuh bahasa
    NusaX adalah terjemahan kalimat sumber yang sama, jadi kesalahannya berkorelasi dan bootstrap per
    kalimat akan membuat CI terlalu sempit."""
    if groups is None:
        return list(rng.integers(0, n, size=(n_boot, n)))
    gr = np.asarray(groups)
    members = [np.flatnonzero(gr == u) for u in np.unique(gr)]
    return [np.concatenate([members[j] for j in rng.integers(0, len(members), size=len(members))])
            for _ in range(n_boot)]


def cohen_kappa(g: np.ndarray, p: np.ndarray) -> float:
    """Kappa Cohen terhadap label emas; 'unparsed' dihitung sebagai kategori tersendiri (salah)."""
    po = float(np.mean(g == p))
    pe = sum(float(np.mean(g == c)) * float(np.mean(p == c)) for c in LABEL_ORDER + ["unparsed"])
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def scores(gold: list[str], pred: list[str], n_boot: int = 1000, groups: list[str] | None = None) -> dict:
    """Prediksi gagal parse ('unparsed') dihitung salah untuk kelas emasnya."""
    g, p = np.array(gold), np.array(pred)
    per_class, prec_c, rec_c = {}, {}, {}
    for c in LABEL_ORDER:
        tp = np.sum((g == c) & (p == c)); fp = np.sum((g != c) & (p == c)); fn = np.sum((g == c) & (p != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = round(2 * prec * rec / (prec + rec) * 100, 2) if prec + rec else 0.0
        prec_c[c], rec_c[c] = round(prec * 100, 2), round(rec * 100, 2)
    rng = np.random.default_rng(42)
    samples = boot_samples(len(g), groups, n_boot, rng)
    polar = (g == "negative") | (g == "positive")
    flips = int(np.sum(((g == "negative") & (p == "positive")) | ((g == "positive") & (p == "negative"))))

    def ci(fn):
        v = np.array([fn(g[i], p[i]) for i in samples])
        return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]

    def polar_acc(gg, pp):
        m = (gg == "negative") | (gg == "positive")
        return float(np.mean(gg[m] == pp[m])) * 100

    def polar_prec(gg, pp):
        m = (pp == "negative") | (pp == "positive")
        return float(np.mean(gg[m] == pp[m])) * 100 if m.any() else 0.0

    return {
        "n": int(len(g)),
        "unparsed": int(np.sum(~np.isin(p, LABEL_ORDER))),
        # CI95: bootstrap per kalimat SUMBER bila groups diberikan (gabungan 7 bahasa), lihat boot_samples
        "ci_method": "bootstrap klaster per kalimat sumber" if groups is not None else "bootstrap per kalimat",
        "accuracy": round(float(np.mean(g == p)) * 100, 2),
        "accuracy_ci95": ci(lambda gg, pp: float(np.mean(gg == pp)) * 100),
        "macro_f1": round(macro_f1(g, p), 2),
        "macro_f1_ci95": ci(macro_f1),
        "kappa": round(cohen_kappa(g, p), 4),
        "kappa_ci95": ci(lambda gg, pp: cohen_kappa(gg, pp)),
        "f1_per_class": per_class,
        "precision_per_class": prec_c,
        "recall_per_class": rec_c,
        # Kalimat yang dilabeli manusia positif/negatif: berapa persen ditebak PERSIS labelnya
        # (= recall gabungan kelas positif+negatif; prediksi netral dihitung salah). Dilaporkan BERSAMA
        # akurasi 3 kelas DAN presisi per kelas (seberapa sering prediksi pos/neg Gemini memang benar).
        "polar_accuracy": round(polar_acc(g, p), 2),
        "polar_accuracy_ci95": ci(polar_acc),
        # Pasangannya: dari prediksi positif/negatif Gemini, berapa persen labelnya tepat sama dengan emas.
        "polar_precision": round(polar_prec(g, p), 2),
        "polar_precision_ci95": ci(polar_prec),
        "polarity_flips": flips,
        "polar_n": int(np.sum(polar)),
        "polarity_flip_rate": round(flips / int(np.sum(polar)) * 100, 2),
        "confusion": {gl: {pl: int(np.sum((g == gl) & (p == pl))) for pl in LABEL_ORDER + ["unparsed"]}
                      for gl in LABEL_ORDER},
    }


def summarize(split: str, variant: str, langs: list[str], show_errors: int) -> dict | None:
    done = load_done(split, variant)
    nb = json.loads(NUSABERT_SUMMARY.read_text(encoding="utf-8"))["results"]
    out = {"split": split, "variant": variant, "variant_spec": VARIANTS[variant],
           "config": PRODUCTION, "system_prompt_example_jav": system_prompt(variant, "jav"),
           "human_prompt": human_template(variant), "per_lang": {}}
    all_g, all_p, all_rows = [], [], []
    print(f"\n[{split.upper()} / {variant}]")
    print(f"{'bahasa':<7}{'n':>5}{'akurasi':>9}{'macro-F1':>10}{'NusaBERT':>10}{'selisih':>9}"
          f"{'flip%':>7}{'pos/neg%':>10}   F1 neg / neu / pos")
    for lang in langs:
        rows = [r for (lg, _), r in done.items() if lg == lang]
        if not rows:
            continue
        g = [r["gold"] for r in rows]
        p = [r["pred"] if r["pred"] in LABEL_ORDER else "unparsed" for r in rows]
        s = scores(g, p, n_boot=200)
        s["nusabert_f1"] = round(nb[lang][NUSABERT_KEY[split]] * 100, 2)
        out["per_lang"][lang] = s
        all_g += g; all_p += p; all_rows += rows
        pc = s["f1_per_class"]
        print(f"{lang:<7}{s['n']:>5}{s['accuracy']:>9.2f}{s['macro_f1']:>10.2f}{s['nusabert_f1']:>10.2f}"
              f"{s['macro_f1'] - s['nusabert_f1']:>+9.2f}{s['polarity_flip_rate']:>7.1f}{s['polar_accuracy']:>10.1f}"
              f"   {pc['negative']:.1f} / {pc['neutral']:.1f} / {pc['positive']:.1f}")
    if not all_g:
        print("Belum ada prediksi.")
        return None
    pooled = scores(all_g, all_p, groups=[str(r["id"]) for r in all_rows])
    done_langs = list(out["per_lang"])
    pooled["mean_macro_f1_over_langs"] = round(np.mean([out["per_lang"][l]["macro_f1"] for l in done_langs]), 2)
    pooled["nusabert_mean_f1"] = round(np.mean([out["per_lang"][l]["nusabert_f1"] for l in done_langs]), 2)
    out["pooled"] = pooled
    pc = pooled["f1_per_class"]
    print(f"{'gabung':<7}{pooled['n']:>5}{pooled['accuracy']:>9.2f}{pooled['macro_f1']:>10.2f}"
          f"{pooled['nusabert_mean_f1']:>10.2f}{'':>9}{pooled['polarity_flip_rate']:>7.1f}{pooled['polar_accuracy']:>10.1f}"
          f"   {pc['negative']:.1f} / {pc['neutral']:.1f} / {pc['positive']:.1f}")
    print(f"akurasi 3 kelas {pooled['accuracy']:.2f} CI95 {pooled['accuracy_ci95']}  |  "
          f"macro-F1 {pooled['macro_f1']:.2f} CI95 {pooled['macro_f1_ci95']}  |  gagal parse {pooled['unparsed']}")
    print(f"kalimat positif/negatif ditebak tepat {pooled['polar_accuracy']:.2f}% CI95 {pooled['polar_accuracy_ci95']}"
          f"  |  prediksi pos/neg yang tepat (presisi) {pooled['polar_precision']:.2f}% CI95 "
          f"{pooled['polar_precision_ci95']}  |  pembalikan polaritas {pooled['polarity_flips']}/{pooled['polar_n']}")
    pr, rc = pooled["precision_per_class"], pooled["recall_per_class"]
    print(f"presisi neg/neu/pos {pr['negative']:.1f}/{pr['neutral']:.1f}/{pr['positive']:.1f}  |  recall "
          f"{rc['negative']:.1f}/{rc['neutral']:.1f}/{rc['positive']:.1f}  |  kappa {pooled['kappa']:.3f} CI95 "
          f"{pooled['kappa_ci95']}  |  CI: {pooled['ci_method']}")
    print("Confusion (baris = emas, kolom = prediksi):")
    print(f"  {'':<10}" + "".join(f"{c:>10}" for c in LABEL_ORDER + ['unparsed']))
    for gl in LABEL_ORDER:
        print(f"  {gl:<10}" + "".join(f"{pooled['confusion'][gl][c]:>10}" for c in LABEL_ORDER + ['unparsed']))
    if show_errors:
        wrong = [r for r in all_rows if r["pred"] != r["gold"]]
        print(f"\nContoh salah ({min(show_errors, len(wrong))} dari {len(wrong)}):")
        for r in wrong[:show_errors]:
            why = f"  | alasan: {r['reasoning'][:110]}" if r.get("reasoning") else ""
            print(f"  [{r['lang']}] emas={r['gold']:<8} prediksi={str(r['pred']):<8} {r['text'][:120]}{why}")
    od = out_dir(split, variant)
    od.mkdir(parents=True, exist_ok=True)
    out["git"] = git_state()
    (od / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    export_results(split, variant, out, all_rows)
    return out


def git_state() -> dict:
    """Commit kode saat ringkasan dibuat, untuk jejak audit (aturan pencatatan eksperimen)."""
    def git(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "script_uncommitted": bool(git("status", "--porcelain", "--", "scripts/gemini_zero_shot_nusax.py"))}


def export_results(split: str, variant: str, out: dict, rows: list[dict]) -> None:
    """Simpan hasil dalam bentuk yang bisa dibuka tanpa Python: CSV, Excel (beberapa sheet), dan
    salinan prompt yang dipakai per bahasa. predictions.jsonl tetap sumber utama (ditulis per kalimat)."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    od = out_dir(split, variant)
    df = pd.DataFrame(rows)
    df["benar"] = (df["pred"] == df["gold"]).astype(int)
    df = df.sort_values(["lang", "id"], key=lambda s: s.astype(str)).reset_index(drop=True)
    cols = [c for c in ["lang", "id", "text", "gold", "pred", "benar", "reasoning", "raw_text", "latency_s"]
            if c in df.columns]
    df = df[cols]
    df.to_csv(od / "predictions.csv", index=False, encoding="utf-8-sig")  # utf-8-sig: aman dibuka di Excel

    ringkas = []
    for lang, s in list(out["per_lang"].items()) + [("GABUNGAN", out["pooled"])]:
        ringkas.append({"bahasa": lang, "n": s["n"], "akurasi_3kelas": s["accuracy"],
                        "akurasi_ci95": str(s.get("accuracy_ci95", "")), "macro_f1": s["macro_f1"],
                        "macro_f1_ci95": str(s["macro_f1_ci95"]), "f1_negative": s["f1_per_class"]["negative"],
                        "f1_neutral": s["f1_per_class"]["neutral"], "f1_positive": s["f1_per_class"]["positive"],
                        "akurasi_kalimat_pos_neg": s["polar_accuracy"],
                        "akurasi_pos_neg_ci95": str(s.get("polar_accuracy_ci95", "")),
                        "presisi_prediksi_pos_neg": s.get("polar_precision"),
                        "akurasi_neg": s.get("recall_per_class", {}).get("negative"),
                        "akurasi_neu": s.get("recall_per_class", {}).get("neutral"),
                        "akurasi_pos": s.get("recall_per_class", {}).get("positive"),
                        "presisi_neg": s.get("precision_per_class", {}).get("negative"),
                        "presisi_neu": s.get("precision_per_class", {}).get("neutral"),
                        "presisi_pos": s.get("precision_per_class", {}).get("positive"),
                        "kappa": s.get("kappa"), "pembalikan_polaritas": s["polarity_flips"],
                        "kalimat_pos_neg": s["polar_n"], "gagal_parse": s["unparsed"],
                        "nusabert_macro_f1": s.get("nusabert_f1", s.get("nusabert_mean_f1"))})
    conf = pd.DataFrame(out["pooled"]["confusion"]).T
    conf.index.name = "emas \\ prediksi"
    prompts = {lang: system_prompt(variant, lang) for lang in LANGS}
    meta = [{"kunci": k, "nilai": str(v)} for k, v in
            [("split", split), ("varian", variant), ("spec", VARIANTS[variant]), *PRODUCTION.items(),
             ("git_commit", out["git"]["commit"]), ("skrip_belum_commit", out["git"]["script_uncommitted"]),
             ("human_prompt", human_template(variant))]]
    meta += [{"kunci": f"system_prompt_{lang}", "nilai": p} for lang, p in prompts.items()]

    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.apply(lambda c: c.map(lambda x: ILLEGAL_CHARACTERS_RE.sub("", x) if isinstance(x, str) else x))

    with pd.ExcelWriter(od / "hasil.xlsx", engine="openpyxl") as xw:
        pd.DataFrame(ringkas).to_excel(xw, sheet_name="ringkasan", index=False)
        conf.to_excel(xw, sheet_name="confusion")
        clean(df).to_excel(xw, sheet_name="prediksi", index=False)
        clean(df[df["benar"] == 0]).to_excel(xw, sheet_name="salah", index=False)
        clean(pd.DataFrame(meta)).to_excel(xw, sheet_name="prompt_config", index=False)
    (od / "prompts_used.json").write_text(json.dumps(
        {"variant": variant, "spec": VARIANTS[variant], "config": PRODUCTION,
         "system_prompt_per_lang": prompts, "human_prompt": human_template(variant),
         "options": OPTIONS if VARIANTS[variant].get("template") else None}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"Tersimpan: {od / 'predictions.csv'} | {od / 'hasil.xlsx'} | {od / 'summary.json'} | prompts_used.json")


class Tee:
    """Salin semua print ke layar DAN ke file log (tetap print, tanpa modul logging)."""

    def __init__(self, stream, fh, lock):
        self.stream, self.fh, self.lock = stream, fh, lock

    def write(self, s):
        with self.lock:
            self.stream.write(s)
            self.fh.write(s)
            self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def start_log(split: str, variant: str) -> Path:
    log_dir = out_dir(split, variant) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    fh = open(path, "a", encoding="utf-8")
    lock = threading.Lock()
    sys.stdout = Tee(sys.stdout, fh, lock)
    sys.stderr = Tee(sys.stderr, fh, lock)
    return path


def compare(split: str, variants: list[str]) -> None:
    """Tabel per varian + perbandingan BERPASANGAN terhadap varian pertama (kalimat yang sama):
    'baik' = salah di acuan jadi benar, 'rusak' = benar di acuan jadi salah."""
    ref = load_done(split, variants[0])
    print(f"\nPerbandingan varian di split {split.upper()} (berpasangan vs {variants[0]}):")
    print(f"{'varian':<8}{'n':>6}{'akurasi':>9}{'macro-F1':>10}{'CI95':>17}{'flip%':>7}{'gagal':>7}"
          f"{'baik':>6}{'rusak':>7}   F1 neg / neu / pos   spec")
    for v in variants:
        p = out_dir(split, v) / "summary.json"
        if not p.exists():
            print(f"{v:<8} (belum ada summary)")
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        q = s["pooled"]; pc = q["f1_per_class"]
        cur = load_done(split, v)
        both = [k for k in cur if k in ref]
        fixed = sum(1 for k in both if ref[k]["pred"] != ref[k]["gold"] and cur[k]["pred"] == cur[k]["gold"])
        broke = sum(1 for k in both if ref[k]["pred"] == ref[k]["gold"] and cur[k]["pred"] != cur[k]["gold"])
        print(f"{v:<8}{q['n']:>6}{q['accuracy']:>9.2f}{q['macro_f1']:>10.2f}{str(q['macro_f1_ci95']):>17}"
              f"{q['polarity_flip_rate']:>7.1f}{q['unparsed']:>7}{fixed:>6}{broke:>7}   {pc['negative']:.1f} / "
              f"{pc['neutral']:.1f} / {pc['positive']:.1f}   {s['variant_spec']}")


def aggregate(split: str, variants: list[str]) -> None:
    """Rata-rata lintas template (protokol SEACrowd/NusaCrowd/Koto dkk.): tiap template dilaporkan,
    lalu rata-rata, SD, dan rentangnya. Disimpan ke <split>/aggregate/ (json + csv + xlsx)."""
    keys = [("accuracy", "akurasi 3 kelas"), ("macro_f1", "macro-F1"), ("kappa", "kappa Cohen"),
            ("polar_accuracy", "akurasi kalimat pos/neg"), ("polar_precision", "presisi prediksi pos/neg"),
            ("polarity_flips", "pembalikan polaritas")]
    rows, per_lang = [], {}
    for v in variants:
        p = out_dir(split, v) / "summary.json"
        if not p.exists():
            print(f"{v}: belum ada summary, dilewati")
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        q = s["pooled"]
        name = ALL_TEMPLATES[VARIANTS[v]["template"]][0] if VARIANTS[v].get("template") else v
        rows.append({"varian": v, "sumber": name, "n": q["n"], "gagal_parse": q["unparsed"],
                     **{k: q[k] for k, _ in keys},
                     # rincian per kelas: akurasi per kelas (= recall) dan presisi
                     **{f"akurasi_{c[:3]}": q["recall_per_class"][c] for c in LABEL_ORDER},
                     **{f"presisi_{c[:3]}": q["precision_per_class"][c] for c in LABEL_ORDER}})
        for lang, ls in s["per_lang"].items():
            d = per_lang.setdefault(lang, {})
            for k_out, val in [("akurasi_3kelas", ls["accuracy"]), ("macro_f1", ls["macro_f1"]),
                               ("akurasi_neg", ls["recall_per_class"]["negative"]),
                               ("akurasi_neu", ls["recall_per_class"]["neutral"]),
                               ("akurasi_pos", ls["recall_per_class"]["positive"]),
                               ("akurasi_pos_neg", ls["polar_accuracy"]),
                               ("presisi_pos_neg", ls["polar_precision"]),
                               ("pembalikan", ls["polarity_flips"])]:
                d.setdefault(k_out, []).append(val)
    if not rows:
        return
    df = pd.DataFrame(rows)
    print(f"\nAgregat {len(df)} template di split {split.upper()}:")
    print(df.to_string(index=False))
    summ = {k: {"mean": round(float(df[k].mean()), 2), "sd": round(float(df[k].std(ddof=1)), 2) if len(df) > 1 else 0.0,
                "min": round(float(df[k].min()), 2), "max": round(float(df[k].max()), 2)} for k, _ in keys}
    print("\nRata-rata lintas template (SD; rentang):")
    for k, label in keys:
        m = summ[k]
        print(f"  {label:<26} {m['mean']:>7.2f}  (SD {m['sd']:.2f}; {m['min']:.2f}–{m['max']:.2f})")
    print("\nRincian per kelas, rata-rata lintas template (akurasi per kelas = recall | presisi):")
    for c in LABEL_ORDER:
        a, p = df[f"akurasi_{c[:3]}"], df[f"presisi_{c[:3]}"]
        print(f"  {c:<9} akurasi {a.mean():6.2f} ({a.min():.1f}–{a.max():.1f})  |  presisi {p.mean():6.2f} "
              f"({p.min():.1f}–{p.max():.1f})")
    best = df.loc[df["accuracy"].idxmax()]
    print(f"\nTemplate tertinggi (akurasi 3 kelas): {best['varian']} {best['sumber']} = {best['accuracy']:.2f}")
    lang_tbl = pd.DataFrame({lang: {k: round(float(np.mean(vals)), 2) for k, vals in d.items()}
                             for lang, d in per_lang.items()}).T
    # Pembanding: NusaBERT-large champion SV di split yang sama, bahasa yang sama
    nb = json.loads(NUSABERT_SUMMARY.read_text(encoding="utf-8"))["results"]
    acc_key = "test_accuracy" if split == "test" else "val_accuracy"
    lang_tbl["nusabert_akurasi"] = [round(nb[l][acc_key] * 100, 2) for l in lang_tbl.index]
    lang_tbl["nusabert_macro_f1"] = [round(nb[l][NUSABERT_KEY[split]] * 100, 2) for l in lang_tbl.index]
    lang_tbl.loc["RATA-RATA"] = lang_tbl.mean().round(2)
    lang_tbl.index.name = "bahasa"
    print("\nPer bahasa (rata-rata lintas template; pembanding NusaBERT di split yang sama):")
    print(lang_tbl.to_string())
    # Set template literatur -> folder "aggregate"; set lain -> folder sendiri agar tidak saling menimpa
    sub = "aggregate" if list(variants) == LIT_SET else "aggregate_" + "_".join(variants)
    ad = ROOT / "outputs" / "gemini_zero_shot" / split / sub
    ad.mkdir(parents=True, exist_ok=True)
    (ad / "aggregate.json").write_text(json.dumps(
        {"split": split, "variants": variants, "per_template": rows, "mean_over_templates": summ,
         "per_lang_mean": lang_tbl.reset_index().to_dict(orient="records"), "git": git_state()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(ad / "per_template.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(ad / "aggregate.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="per_template", index=False)
        pd.DataFrame(summ).T.to_excel(xw, sheet_name="rata_rata")
        lang_tbl.to_excel(xw, sheet_name="per_bahasa")
    print(f"Tersimpan: {ad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--variant", choices=list(VARIANTS), default="v1")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--per-label", type=int, default=None, help="sampel acak N kalimat per label per bahasa")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--show-errors", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--compare", nargs="+", metavar="VARIAN", help="bandingkan summary beberapa varian")
    ap.add_argument("--aggregate", nargs="*", metavar="VARIAN",
                    help="rata-rata lintas template (tanpa argumen = semua template literatur L1..L6)")
    args = ap.parse_args()

    if args.compare:
        compare(args.split, args.compare)
        return
    if args.aggregate is not None:
        aggregate(args.split, args.aggregate or LIT_SET)
        return
    if not args.dry_run:
        print(f"Log konsol disalin ke: {start_log(args.split, args.variant)}")
    check_production_config()
    if args.split == "test" and not args.summarize_only and not args.dry_run:
        print("\n*** SPLIT TEST: jalankan SEKALI, dengan varian yang sudah dibekukan dan di-commit. ***")
    if args.dry_run:
        sample = pd.read_csv(DATA_DIR / "jav" / "valid.csv").iloc[0]
        print(f"\nVarian {args.variant} {VARIANTS[args.variant]} — contoh pesan (valid jav):")
        for role, content in build_messages(args.variant, sample["text"], "jav"):
            print(f"[{role}]\n{content}\n")
        return
    if not args.summarize_only:
        run_predictions(args.split, args.variant, args.langs, args.per_label, args.workers)
    summarize(args.split, args.variant, args.langs, args.show_errors)


if __name__ == "__main__":
    main()
