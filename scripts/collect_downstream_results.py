"""Kumpulkan hasil fine-tune downstream -> Excel + Markdown.

Sumber kebenaran = `outputs/*/results_summary.json` (ditulis tiap skrip finetune).
Skrip ini cuma MENGUMPULKAN, jadi aman dijalankan berkali-kali (rebuild dari nol,
tidak pernah dobel). Jalankan tiap selesai satu fine-tune.

Keluaran:
  - outputs/downstream_results.xlsx      (plain, 2 sheet: per-bahasa + ringkas)
  - agent_doc/thesis/downstream_results.md

Jalankan dari root project:  uv run python scripts/collect_downstream_results.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCOPE = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]  # 7 bahasa target tesis
XLSX = ROOT / "outputs" / "downstream_results.xlsx"
MD = ROOT / "agent_doc" / "thesis" / "downstream_results.md"

# ── Angka published buat pembanding (KUTIPAN, BUKAN hasil kami) ──────────────
# SUMBER TUNGGAL: NusaBERT paper (Wongso dkk. 2025) **Tabel 6**, macro-F1 (%) NusaX-Senti.
# Tabel 6 memuat baris NusaBERT DAN XLM-R sekaligus; baris XLM-R di situ dikutip Wongso
# dari Winata dkk. 2023 ("Baseline results are obtained from Winata et al. (2023)").
# ⚠️ DIVERIFIKASI FIRST-HAND dari agent_doc/_paper_pdfs/nusabert_arxiv2403.01817.pdf (2026-07-17).
#    Versi sebelumnya diisi dari INGATAN -> 27/28 angka SALAH (XLM-R large bjn meleset 6.8 poin).
#    JANGAN pernah isi tabel ini tanpa buka PDF-nya.
# Baris mentah Tabel 6 (urutan: ace ban bbc bjn bug eng ind jav mad min nij sun mu):
#   XLM-RBASE      73.9 72.8 62.3 76.6 66.6 90.8 88.4 78.9 69.7 79.1 75.0 80.1 | 76.2
#   XLM-RLARGE     75.9 77.1 65.5 86.3 70.0 92.6 91.6 84.2 74.9 83.1 73.3 86.0 | 80.0
#   NusaBERTBASE   76.5 78.7 74.0 82.4 71.6 84.1 89.7 84.1 75.6 80.8 74.9 85.2 | 79.8
#   NusaBERTLARGE  81.8 82.8 74.7 86.5 73.4 84.6 93.3 87.2 82.5 83.5 77.7 82.7 | 82.6
# mu paper = rata-rata 12 bahasa; scope kita 7 -> mean di skrip ini DIHITUNG ULANG atas 7 saja.
PAPER_NUSABERT = {
    "base":  {"ace": 76.5, "ban": 78.7, "bjn": 82.4, "jav": 84.1, "mad": 75.6, "min": 80.8, "sun": 85.2},
    "large": {"ace": 81.8, "ban": 82.8, "bjn": 86.5, "jav": 87.2, "mad": 82.5, "min": 83.5, "sun": 82.7},
}
PAPER_XLMR = {
    "base":  {"ace": 73.9, "ban": 72.8, "bjn": 76.6, "jav": 78.9, "mad": 69.7, "min": 79.1, "sun": 80.1},
    "large": {"ace": 75.9, "ban": 77.1, "bjn": 86.3, "jav": 84.2, "mad": 74.9, "min": 83.1, "sun": 86.0},
}


def infer_model(ckpt: str) -> tuple[str, str]:
    """(family, size) dari model_checkpoint."""
    c = ckpt.lower()
    fam = "NusaBERT" if "nusabert" in c else ("XLM-R" if "xlm-roberta" in c else "?")
    size = "large" if "large" in c else ("base" if "base" in c else "?")
    return fam, size


def main() -> None:
    rows = []
    for js in sorted((ROOT / "outputs").glob("downstream_*/*/results_summary.json")):
        try:
            d = json.loads(js.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {js.parent.name}: {e}")
            continue
        hp, res = d.get("hyperparams", {}), d.get("results", {})
        source = js.parent.parent.name.replace("downstream_", "")  # "full" | "single_llm"
        fam, size = infer_model(hp.get("model_checkpoint", ""))
        if fam == "?":
            continue
        aug = hp.get("use_augmented")
        if aug is None:  # skrip lama tak menulis flag -> tebak dari nama folder
            aug = "syn" in js.parent.name
        paper = (PAPER_NUSABERT if fam == "NusaBERT" else PAPER_XLMR).get(size, {})
        for lang, r in res.items():
            if lang not in SCOPE:
                continue  # buang bahasa di luar scope (bbc/bug/nij/eng/ind)
            test_f1 = round(r.get("test_f1", 0) * 100, 2) if r.get("test_f1", 0) <= 1 else round(r.get("test_f1", 0), 2)
            val_f1 = round(r.get("val_f1", 0) * 100, 2) if r.get("val_f1", 0) <= 1 else round(r.get("val_f1", 0), 2)
            train_f1 = round(r.get("train_f1", 0) * 100, 2) if r.get("train_f1", 0) <= 1 else round(r.get("train_f1", 0), 2)
            acc = r.get("test_accuracy", 0)
            rows.append({
                "model": fam, "size": size,
                "skenario": "Augmentasi" if aug else "Baseline",
                "lang": lang,
                "test_f1": test_f1,
                "test_acc": round(acc * 100, 2) if acc <= 1 else round(acc, 2),
                "val_f1": val_f1, "train_f1": train_f1,
                "gap_train_test": round(train_f1 - test_f1, 2),
                "paper_f1": paper.get(lang),
                "delta_vs_paper": round(test_f1 - paper[lang], 2) if lang in paper else None,
                "seed": hp.get("seed"), "lr": hp.get("learning_rate"),
                "batch": hp.get("train_batch_size"), "patience": hp.get("early_stopping_patience"),
                "source": source, "run_dir": js.parent.name,
            })

    if not rows:
        print("Belum ada results_summary.json yang cocok.")
        return

    df = pd.DataFrame(rows).sort_values(["source", "model", "size", "skenario", "lang"]).reset_index(drop=True)

    # Ringkasan: mean atas 7 bahasa, PER RUN_DIR.
    # WAJIB group by run_dir — outputs/ berisi banyak eksperimen lama (multi-seed 0/42/123,
    # varian rasio 40/50/60, sweep earlystop_3). Tanpa ini semuanya kerata-rata jadi satu = ngawur.
    summ = (df.groupby(["source", "model", "size", "skenario", "seed", "patience", "run_dir"])
              .agg(n_lang=("lang", "count"), mean_test_f1=("test_f1", "mean"),
                   mean_paper_f1=("paper_f1", "mean"), mean_delta=("delta_vs_paper", "mean"),
                   mean_gap=("gap_train_test", "mean"))
              .round(2).reset_index()
              .sort_values(["source", "model", "size", "skenario", "run_dir"]))
    # n_lang != 7 -> run belum lengkap / scope beda; ditandai supaya tak salah dibaca
    summ["lengkap"] = summ["n_lang"].map(lambda n: "OK" if n == len(SCOPE) else f"!! {n}/7")

    # ── Sheet tabel Bab IV: mean +- std LINTAS-SEED per (model, size, skenario)
    # std = simpangan baku SAMPEL (pembagi n-1). pandas .std() default ddof=1 -> benar.
    # (numpy.std default ddof=0 = populasi -> JANGAN dipakai, beda ~19% di n=3)
    agg = (df.groupby(["source", "model", "size", "skenario", "lang"])
             .agg(mean=("test_f1", "mean"), std=("test_f1", "std"),
                  n_seed=("seed", "nunique"), paper=("paper_f1", "first"))
             .round(2).reset_index())
    agg["mean_pm_std"] = agg.apply(
        lambda r: f"{r['mean']:.2f}" if pd.isna(r["std"]) else f"{r['mean']:.2f}±{r['std']:.2f}", axis=1)
    agg["delta_vs_paper"] = (agg["mean"] - agg["paper"]).round(2)
    # pivot: baris = model, kolom = bahasa (bentuk tabel Bab IV)
    tabel = agg.pivot_table(index=["source", "model", "size", "skenario"], columns="lang",
                            values="mean_pm_std", aggfunc="first").reset_index()
    mu = (agg.groupby(["source", "model", "size", "skenario"])
            .agg(mu=("mean", "mean"), mu_paper=("paper", "mean"), n_seed=("n_seed", "max"))
            .round(2).reset_index())
    mu["delta_mu"] = (mu["mu"] - mu["mu_paper"]).round(2)
    tabel = tabel.merge(mu, on=["source", "model", "size", "skenario"])

    XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        tabel.to_excel(w, sheet_name="tabel_bab_iv", index=False)
        agg.to_excel(w, sheet_name="per_model", index=False)
        df.to_excel(w, sheet_name="per_bahasa", index=False)
        summ.to_excel(w, sheet_name="ringkas", index=False)
    print(f"XLSX -> {XLSX}  ({len(df)} baris, {len(summ)} run, {len(tabel)} config tabel)")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        "# Hasil Downstream — NusaSynth (auto-generated)",
        "",
        f"> Dibuat ulang otomatis oleh `scripts/collect_downstream_results.py` — **jangan edit tangan**.",
        f"> Terakhir: {stamp}. Sumber: `outputs/*/results_summary.json`. Scope: 7 bahasa ({', '.join(SCOPE)}).",
        "> `paper_f1` = KUTIPAN published (NusaBERT: Wongso dkk. 2025; XLM-R: Winata dkk. 2023 Tabel 2), **bukan** hasil kami.",
        "> ⚠️ Klaim yang sah = *kompetitif/mengungguli baseline published*, BUKAN kausal *'augmentasi naik +Xpp'*",
        "> (lihat `thesis_writing_notes.md` §Keputusan FINAL).",
        "> `mean±std` = rata-rata & **simpangan baku sampel (pembagi n−1)** atas seed; `n_seed` = jumlah seed.",
        "> Kedua paper acuan melaporkan sentimen dari **satu kali proses tanpa mencantumkan seed** → baris published tak punya ±.",
        "", "## Tabel Bab IV (mean±std lintas-seed)", "",
        tabel.to_markdown(index=False), "",
        "> ⚠️ Satu baris ringkas = SATU `run_dir` (satu seed). JANGAN dirata-rata manual lintas run_dir —",
        "> `outputs/` bisa berisi eksperimen lain (varian rasio, sweep patience).",
        "", "## Ringkas per run (satu baris = satu seed)", "",
        summ.to_markdown(index=False), "",
        "## Per bahasa", "",
        df.to_markdown(index=False), "",
    ]
    MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"MD   -> {MD}")

    print("\n" + summ.to_string(index=False))


if __name__ == "__main__":
    main()
