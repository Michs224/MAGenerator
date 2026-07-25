"""Analisis KENAPA single-LLM kompetitif vs full pipeline.
Bagian 1: dinamika training (dari results_summary) + sisi-data + leakage/novelty.
Semua angka via code. Tanpa load model.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path("y:/Michh/Python/Projects/MAGenerator")
sys.path.insert(0, str(ROOT))
from NusaSynth.tools import jaccard_bigram

LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
LABELS = ["negative", "neutral", "positive"]
SEEDS = [42, 0, 123]
FULL_SYN = ROOT / "outputs/synthetic/2026-07-17_02-40-52"
SINGLE_SYN = ROOT / "outputs/synthetic_single_llm/2026-07-23_22-12-08"


def wc(s):
    return len(str(s).split())


# ============================================================
# BAGIAN A — DINAMIKA TRAINING (results_summary, semua model/seed)
# ============================================================
MODELS = [("nusabert-sentiment-base", "NusaBERT-base"),
          ("nusabert-sentiment-large", "NusaBERT-large"),
          ("xlmr-sentiment-base", "XLM-R-base")]


def dyn():
    print("=" * 78)
    print("BAGIAN A — DINAMIKA TRAINING (mean 7 bahasa × 3 seed)")
    print("=" * 78)
    print(f"{'model':16} {'src':7} {'train':>7} {'val':>7} {'test':>7} {'ovf(tr-te)':>10} {'val-te':>7} {'best_ep':>7} {'stop':>6}")
    for folder, name in MODELS:
        for src in ["full", "single_llm"]:
            tr, va, te, bep, sep = [], [], [], [], []
            for seed in SEEDS:
                p = ROOT / f"outputs/downstream_{src}/{folder}-syn-seed{seed}/results_summary.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())["results"]
                for lang in LANGS:
                    r = d[lang]
                    tr.append(r["train_f1"]); va.append(r["val_f1"]); te.append(r["test_f1"])
                    if "best_epoch" in r: bep.append(r["best_epoch"]); sep.append(r["stop_epoch"])
            if not tr:
                continue
            tr, va, te = np.mean(tr) * 100, np.mean(va) * 100, np.mean(te) * 100
            be = f"{np.mean(bep):7.1f}" if bep else f"{'-':>7}"
            se = f"{np.mean(sep):6.1f}" if sep else f"{'-':>6}"
            print(f"{name:16} {src:7} {tr:7.2f} {va:7.2f} {te:7.2f} {tr-te:10.2f} {va-te:7.2f} {be} {se}")
    # per-lang test single-full utk nusabert-base (di mana single menang/kalah)
    print("\n-- NusaBERT-base: test_f1 per bahasa (mean 3 seed), single - full --")
    for src in ["full", "single_llm"]:
        pass
    rows = {}
    for src in ["full", "single_llm"]:
        per = {l: [] for l in LANGS}
        for seed in SEEDS:
            d = json.loads((ROOT / f"outputs/downstream_{src}/nusabert-sentiment-base-syn-seed{seed}/results_summary.json").read_text())["results"]
            for l in LANGS:
                per[l].append(d[l]["test_f1"] * 100)
        rows[src] = {l: np.mean(v) for l, v in per.items()}
    print(f"{'lang':5} {'full':>7} {'single':>7} {'Δ(s-f)':>8}")
    for l in LANGS:
        f, s = rows["full"][l], rows["single_llm"][l]
        print(f"{l:5} {f:7.2f} {s:7.2f} {s-f:+8.2f}")


# ============================================================
# BAGIAN B — SISI DATA (length, diversity, label-dist)
# ============================================================
def bigrams(s):
    w = str(s).lower().split()
    return set(zip(w, w[1:]))


def distinct2(texts):
    allbg, tot = set(), 0
    for t in texts:
        w = str(t).lower().split()
        bg = list(zip(w, w[1:]))
        allbg.update(bg); tot += len(bg)
    return len(allbg) / tot if tot else 0


def greedy_duprate(texts):
    """fraksi yang DIBUANG dedup pipeline (jaccard bigram >=0.5 vs yang sudah disimpan)."""
    kept, index = [], {}  # bigram -> list kept idx
    dropped = 0
    for t in texts:
        bg = bigrams(t)
        cand = set()
        for g in bg:
            cand.update(index.get(g, ()))
        dup = False
        for ci in cand:
            kb = kept[ci]
            if len(bg | kb) and len(bg & kb) / len(bg | kb) >= 0.5:
                dup = True; break
        if dup:
            dropped += 1
        else:
            ki = len(kept); kept.append(bg)
            for g in bg:
                index.setdefault(g, []).append(ki)
    return dropped / len(texts) if texts else 0


def max_jac_vs(texts, ref_texts):
    """utk tiap text, max jaccard-bigram vs koleksi ref (inverted index). return list of max."""
    ref_bg = [bigrams(r) for r in ref_texts]
    index = {}
    for i, bg in enumerate(ref_bg):
        for g in bg:
            index.setdefault(g, []).append(i)
    out = []
    for t in texts:
        bg = bigrams(t)
        cand = set()
        for g in bg:
            cand.update(index.get(g, ()))
        mx = 0.0
        for ci in cand:
            rb = ref_bg[ci]
            u = len(bg | rb)
            if u:
                j = len(bg & rb) / u
                if j > mx:
                    mx = j
        out.append(mx)
    return np.array(out)


def data_side():
    print("\n" + "=" * 78)
    print("BAGIAN B — SISI DATA (full vs single, mean 7 bahasa)")
    print("=" * 78)
    agg = {"full": [], "single": []}
    perlang = []
    for lang in LANGS:
        seed = pd.read_csv(ROOT / f"data/nusax_senti/{lang}/train.csv")
        full = pd.read_csv(FULL_SYN / lang / "synthetic.csv")
        single = pd.read_csv(SINGLE_SYN / lang / "synthetic.csv")
        row = {"lang": lang}
        for name, df in [("full", full), ("single", single)]:
            texts = df["text"].dropna().astype(str).tolist()
            # length R_sigma & Delta_mu (pooled semua label vs seed pooled)
            slen = seed["text"].map(wc); dlen = df["text"].map(wc)
            rsig = dlen.std() / slen.std()
            dmu = dlen.mean() - slen.mean()
            row[f"{name}_len_mu"] = dlen.mean()
            row[f"{name}_Rsig"] = rsig
            row[f"{name}_Dmu"] = dmu
            row[f"{name}_d2"] = distinct2(texts)
            row[f"{name}_dup"] = greedy_duprate(texts)
            lc = df["label"].value_counts()
            row[f"{name}_neu%"] = 100 * lc.get("neutral", 0) / len(df)
            row[f"{name}_n"] = len(df)
        perlang.append(row)
    dfp = pd.DataFrame(perlang)
    seed_len = pd.concat([pd.read_csv(ROOT / f"data/nusax_senti/{l}/train.csv")["text"].map(wc) for l in LANGS])
    print(f"seed len rata2 (7 bhs): mu={seed_len.mean():.1f} std={seed_len.std():.1f}\n")
    cols = ["lang", "full_len_mu", "single_len_mu", "full_Rsig", "single_Rsig",
            "full_Dmu", "single_Dmu", "full_d2", "single_d2", "full_dup", "single_dup",
            "full_neu%", "single_neu%"]
    print(dfp[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n-- MEAN 7 bahasa --")
    for m in ["len_mu", "Rsig", "Dmu", "d2", "dup", "neu%", "n"]:
        f, s = dfp[f"full_{m}"].mean(), dfp[f"single_{m}"].mean()
        print(f"  {m:8}: full={f:8.3f}   single={s:8.3f}   (Δ s-f={s-f:+.3f})")
    return dfp


# ============================================================
# BAGIAN C — LEAKAGE (vs TEST) & NOVELTY (vs SEED)
# ============================================================
def leakage():
    print("\n" + "=" * 78)
    print("BAGIAN C — LEAKAGE vs TEST & NOVELTY vs SEED (KRITIS)")
    print("  leak = fraksi syn dgn max jaccard-bigram >=0.5 ke SOME test text (mirip test)")
    print("  seed-sim = fraksi syn dgn max jaccard >=0.5 ke SOME seed text (kurang novel)")
    print("=" * 78)
    print(f"{'lang':5} | {'full leak%':>10} {'sngl leak%':>10} | {'full meanJ':>10} {'sngl meanJ':>10} | {'full seedsim%':>13} {'sngl seedsim%':>13}")
    agg = {}
    for lang in LANGS:
        seed = pd.read_csv(ROOT / f"data/nusax_senti/{lang}/train.csv")["text"].astype(str).tolist()
        test = pd.read_csv(ROOT / f"data/nusax_senti/{lang}/test.csv")["text"].astype(str).tolist()
        out = {}
        for name, path in [("full", FULL_SYN), ("single", SINGLE_SYN)]:
            syn = pd.read_csv(path / lang / "synthetic.csv")["text"].dropna().astype(str).tolist()
            mj_test = max_jac_vs(syn, test)
            mj_seed = max_jac_vs(syn, seed)
            out[name] = dict(leak=100 * (mj_test >= 0.5).mean(), meanJ=mj_test.mean(),
                             exact=int((mj_test >= 0.999).sum()), seedsim=100 * (mj_seed >= 0.5).mean())
        agg[lang] = out
        print(f"{lang:5} | {out['full']['leak']:10.2f} {out['single']['leak']:10.2f} | "
              f"{out['full']['meanJ']:10.3f} {out['single']['meanJ']:10.3f} | "
              f"{out['full']['seedsim']:13.2f} {out['single']['seedsim']:13.2f}")
    for k in ["leak", "meanJ", "seedsim", "exact"]:
        f = np.mean([agg[l]["full"][k] for l in LANGS])
        s = np.mean([agg[l]["single"][k] for l in LANGS])
        print(f"  MEAN {k:8}: full={f:.3f}  single={s:.3f}")


def noise_check():
    print("\n" + "=" * 78)
    print("BAGIAN A2 — GAP single-vs-full DALAM NOISE SEED?")
    print("  seed_std = std(3 seed) per bahasa, dirata2 7 bhs = tipikal ayun run-to-run")
    print("  |gap μ| = |μ_single − μ_full|  → kalau gap << seed_std ⇒ beda = NOISE")
    print("=" * 78)
    print(f"{'model':16} {'μ_full':>7} {'μ_single':>8} {'|gap μ|':>8} {'seed_std_full':>13} {'seed_std_single':>15} {'gap<noise?':>11}")
    for folder, name in MODELS:
        mu, sstd = {}, {}
        for src in ["full", "single_llm"]:
            per = {l: [] for l in LANGS}
            ok = True
            for seed in SEEDS:
                p = ROOT / f"outputs/downstream_{src}/{folder}-syn-seed{seed}/results_summary.json"
                if not p.exists():
                    ok = False; continue
                d = json.loads(p.read_text())["results"]
                for l in LANGS:
                    per[l].append(d[l]["test_f1"] * 100)
            lang_mu = [np.mean(per[l]) for l in LANGS]
            lang_sd = [np.std(per[l], ddof=1) for l in LANGS]
            mu[src] = np.mean(lang_mu); sstd[src] = np.mean(lang_sd)
        gap = abs(mu["single_llm"] - mu["full"])
        noise = max(sstd["full"], sstd["single_llm"])
        print(f"{name:16} {mu['full']:7.2f} {mu['single_llm']:8.2f} {gap:8.2f} "
              f"{sstd['full']:13.2f} {sstd['single_llm']:15.2f} {('YA' if gap < noise else 'tidak'):>11}")


def cleanvolume_check():
    """Cek apakah single-LLM ikut hukum clean-volume ΔF1 = a - b*baseline (base)."""
    print("\n" + "=" * 78)
    print("BAGIAN A3 — CLEAN-VOLUME LAW utk SINGLE-LLM (NusaBERT-base)")
    print("  ΔF1 = F1(augmented single) − F1(baseline real-only). Regres vs baseline_F1.")
    print("  Repo: full base slope=−0.534, r=−0.867. Kalau single mirip ⇒ mekanisme sama.")
    print("=" * 78)
    BASE_REAL = {"ace": 72.61, "ban": 78.00, "bjn": 83.06, "jav": 86.18, "mad": 73.35, "min": 83.88, "sun": 82.94}
    for src in ["full", "single_llm"]:
        per = {l: [] for l in LANGS}
        for seed in SEEDS:
            d = json.loads((ROOT / f"outputs/downstream_{src}/nusabert-sentiment-base-syn-seed{seed}/results_summary.json").read_text())["results"]
            for l in LANGS:
                per[l].append(d[l]["test_f1"] * 100)
        base = np.array([BASE_REAL[l] for l in LANGS])
        aug = np.array([np.mean(per[l]) for l in LANGS])
        delta = aug - base
        b, a = np.polyfit(base, delta, 1)
        r = np.corrcoef(base, delta)[0, 1]
        print(f"  {src:10}: ΔF1 = {a:.1f} {b:+.3f}·baseline | r={r:+.3f} | meanΔ={delta.mean():+.2f} | "
              f"per-lang Δ: " + " ".join(f"{l}={d:+.1f}" for l, d in zip(LANGS, delta)))


if __name__ == "__main__":
    dyn()
    noise_check()
    cleanvolume_check()
    data_side()
    leakage()
