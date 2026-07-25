"""Perbandingan KUALITAS DATA (Bab III Tabel 3.7): NusaSynth penuh vs Single-LLM.
Fungsi Self-BLEU / BLEU-vs-Seed disalin VERBATIM dari bab_iv_metrics.ipynb (komparabilitas).
Pipeline Outcome (pers. 3.1-3.8) + Conflict Rate (3.15-3.16) dari pipeline_log (FULL-only).
Length R_sigma/Delta_mu (3.13-3.14) per (bahasa x label). GlotLID ID Rate (3.9): pakai audit
(single) + purity re-run (full), keduanya post-hoc di output -> ADIL.

  PYTHONUTF8=1 uv run python scripts/ablation_dataquality.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

ROOT = Path("y:/Michh/Python/Projects/MAGenerator")
sys.path.insert(0, str(ROOT))
from NusaSynth.config import synthetic_dir, single_llm_synthetic_dir

LANGS = ["ace", "ban", "bjn", "jav", "mad", "min", "sun"]
LABELS = ["negative", "neutral", "positive"]

# ---- fungsi metrik VERBATIM dari bab_iv_metrics.ipynb ----
smoothie = SmoothingFunction().method1
ORDERS = (2, 3, 4, 5)
WEIGHTS = [tuple([1.0 / n] * n) for n in ORDERS]
SAMPLE_SEED = 42
REF = 4  # BLEU-4 orde acuan (pers. 3.11/3.12)


def tokenize(t): return t.lower().split()
def _bleu(refs, hyp): return dict(zip(ORDERS, sentence_bleu(refs, hyp, WEIGHTS, smoothing_function=smoothie)))


def self_bleu(sents):
    if len(sents) < 2: return {n: 0.0 for n in ORDERS}
    T = [tokenize(s) for s in sents]; acc = {n: [] for n in ORDERS}
    for i in range(len(T)):
        d = _bleu([T[j] for j in range(len(T)) if j != i], T[i])
        for n in ORDERS: acc[n].append(d[n])
    return {n: float(np.mean(acc[n])) for n in ORDERS}


def bleu_vs_ref(hyps, refs):
    if not hyps or not refs: return {n: 0.0 for n in ORDERS}
    R = [tokenize(r) for r in refs]; acc = {n: [] for n in ORDERS}
    for h in hyps:
        d = _bleu(R, tokenize(h))
        for n in ORDERS: acc[n].append(d[n])
    return {n: float(np.mean(acc[n])) for n in ORDERS}


def match_sample(items, n):
    if len(items) <= n: return list(items)
    rng = np.random.default_rng(SAMPLE_SEED)
    return [items[i] for i in rng.choice(len(items), n, replace=False)]


def diversity_novelty(resolver):
    """Self-BLEU-4 sintetis + seed baseline + BLEU-vs-Seed-4, mu 7 bahasa (pers. 3.11/3.12)."""
    ss, sd, bs = [], [], []
    for lang in LANGS:
        seed_df = pd.read_csv(ROOT / f"data/nusax_senti/{lang}/train.csv")
        syn_df = pd.read_csv(resolver(lang) / "synthetic.csv")
        p_ss, p_sd, p_bs = [], [], []
        for label in LABELS:
            seed_t = seed_df[seed_df.label == label]["text"].dropna().tolist()
            syn_all = syn_df[syn_df.label == label]["text"].dropna().tolist()
            if len(syn_all) < 2 or len(seed_t) < 2: continue
            p_ss.append(self_bleu(match_sample(syn_all, len(seed_t)))[REF])
            p_sd.append(self_bleu(seed_t)[REF])
            p_bs.append(bleu_vs_ref(syn_all, seed_t)[REF])
        ss.append(np.mean(p_ss)); sd.append(np.mean(p_sd)); bs.append(np.mean(p_bs))
    return np.mean(ss), np.mean(sd), np.mean(bs)


def length_match(resolver):
    """R_sigma + Delta_mu per (bahasa x label), pers. 3.13/3.14. Return mean semua sel + per label."""
    rsig = {c: [] for c in LABELS}; dmu = {c: [] for c in LABELS}
    for lang in LANGS:
        seed_df = pd.read_csv(ROOT / f"data/nusax_senti/{lang}/train.csv")
        syn_df = pd.read_csv(resolver(lang) / "synthetic.csv")
        seed_df["wc"] = seed_df.text.str.split().str.len()
        syn_df["wc"] = syn_df.text.str.split().str.len()
        for c in LABELS:
            ss = seed_df[seed_df.label == c]["wc"]; sy = syn_df[syn_df.label == c]["wc"]
            if len(ss) > 1 and len(sy) > 1:
                rsig[c].append(sy.std() / ss.std()); dmu[c].append(sy.mean() - ss.mean())
    all_r = [v for c in LABELS for v in rsig[c]]; all_d = [v for c in LABELS for v in dmu[c]]
    return (np.mean(all_r), np.mean(all_d),
            {c: (np.mean(rsig[c]), np.mean(dmu[c])) for c in LABELS})


def pipeline_outcome():
    """FULL pipeline_log -> Metrik 1 (pers 3.1-3.8) + Conflict 6/7 (3.15-3.16). Single = trivial."""
    from collections import Counter
    oc = Counter(); acc_retry = Counter(); nb_conf = 0; gl_conf = 0; acc = 0
    base = synthetic_dir(LANGS[0]).parent
    for lang in LANGS:
        for ln in open(base / lang / "pipeline_log.jsonl", encoding="utf-8"):
            d = json.loads(ln); oc[d["outcome"]] += 1
            if d["outcome"] == "accepted":
                acc += 1; acc_retry[int(d["retry_count"])] += 1
                if d["nusabert_label"] != d["target_label"]: nb_conf += 1
                if d["glotlid_lang"] != d["target_lang"]: gl_conf += 1
    terminal = oc["accepted"] + oc["discarded"] + oc["dedup_filtered"]
    return dict(terminal=terminal, accepted=acc,
                a0=100 * acc_retry[0] / terminal, a1=100 * acc_retry[1] / terminal, a2=100 * acc_retry[2] / terminal,
                discard=100 * oc["discarded"] / terminal, dedup=100 * oc["dedup_filtered"] / terminal,
                filter_rate=100 * acc / terminal,
                nb_conflict=100 * nb_conf / acc, gl_conflict=100 * gl_conf / acc,
                raw=dict(oc))


if __name__ == "__main__":
    print("=" * 74)
    print("KUALITAS DATA — NusaSynth penuh vs Single-LLM (Bab III Tabel 3.7)")
    print("=" * 74)

    print("\n## Kelompok 2 — Keberagaman & Kebaruan (mu 7 bahasa, BLEU-4)")
    for name, res in [("PENUH", synthetic_dir), ("SINGLE-LLM", single_llm_synthetic_dir)]:
        ss, sd, bs = diversity_novelty(res)
        print(f"  {name:11}: Self-BLEU sintetis={ss:.4f}  (Self-BLEU seed={sd:.4f})  BLEU-vs-Seed={bs:.4f}")

    print("\n## Kelompok 2 — Length Distribution Match (mean sel bahasa×label, pers 3.13/3.14)")
    for name, res in [("PENUH", synthetic_dir), ("SINGLE-LLM", single_llm_synthetic_dir)]:
        r, d, per = length_match(res)
        print(f"  {name:11}: R_sigma={r:.3f}  Delta_mu={d:+.2f}  | per label " +
              " ".join(f"{c[:3]}(R={per[c][0]:.2f},Δ={per[c][1]:+.1f})" for c in LABELS))

    print("\n## Kelompok 1 — Pipeline Outcome (pers 3.1-3.8) + Kelompok 3 Conflict (3.15/3.16) — FULL only")
    po = pipeline_outcome()
    print(f"  N_terminal={po['terminal']}  accepted={po['accepted']}")
    print(f"  Filter Rate = {po['filter_rate']:.2f}%  (Attempt0={po['a0']:.2f}  Attempt1={po['a1']:.2f}  Attempt2={po['a2']:.2f})")
    print(f"  Discard = {po['discard']:.2f}%   Cross-batch Dedup = {po['dedup']:.2f}%")
    print(f"  NusaBERT-LLM Conflict = {po['nb_conflict']:.2f}%   GlotLID-LLM Conflict = {po['gl_conflict']:.2f}%")
    print(f"  (Single-LLM: tanpa pipeline -> Attempt0=100%, retry/discard/dedup=0, tanpa validator)")
    print(f"\n  raw outcome: {po['raw']}")
    print("\nGlotLID ID Rate (Metrik 2, post-hoc): PENUH 97.72% | SINGLE 90.50% (dari purity_compare + audit)")
