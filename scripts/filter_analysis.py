"""Eksperimen filter metric untuk synthetic Javanese.

Menjalankan analisis empiris berbagai filter dari literature:
1. OOV coverage vs test (Phase 1 verification)
2. Distinct-N diversity (Li et al. 2016)
3. Type-Token Ratio (TTR + MTLD-inspired)
4. Per-sentence novelty score (NEW vocab vs seed pool)
5. Length distribution comparison (KL-style)
6. Vocabulary entropy (Shannon)

Output: print ke console untuk evaluasi cepat.
"""
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path('y:/Michh/Python/Projects/MAGenerator')
SEED_TRAIN = ROOT / 'data/nusax_senti/jav/train.csv'
SEED_TEST  = ROOT / 'data/nusax_senti/jav/test.csv'
SEED_VALID = ROOT / 'data/nusax_senti/jav/valid.csv'
SYN_FULL   = ROOT / 'outputs/synthetic/jav/synthetic.csv'

# Tokenization (consistent across analyses)
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

def vocab(texts):
    v = set()
    for t in texts:
        v |= set(tokenize(t))
    return v

def all_tokens(texts):
    toks = []
    for t in texts:
        toks.extend(tokenize(t))
    return toks

# ── Load
print("="*70)
print("LOAD DATA")
print("="*70)
seed_train = pd.read_csv(SEED_TRAIN)
seed_test  = pd.read_csv(SEED_TEST)
seed_valid = pd.read_csv(SEED_VALID)
syn = pd.read_csv(SYN_FULL)
print(f"Seed train : {len(seed_train)} rows")
print(f"Seed valid : {len(seed_valid)} rows")
print(f"Seed test  : {len(seed_test)} rows")
print(f"Synthetic  : {len(syn)} rows")

train_texts = seed_train['text'].dropna().tolist()
valid_texts = seed_valid['text'].dropna().tolist()
test_texts  = seed_test['text'].dropna().tolist()
syn_texts   = syn['text'].dropna().tolist()

# ===================================================================
# EXPERIMENT 1: OOV COVERAGE vs TEST
# ===================================================================
print("\n" + "="*70)
print("EXPERIMENT 1 — OOV COVERAGE vs TEST SET")
print("="*70)
print("Goal: berapa % unique words di test set yang tercover oleh train pool?")
print("Metric: |test_vocab ∩ pool_vocab| / |test_vocab|")
print()

train_vocab = vocab(train_texts)
test_vocab  = vocab(test_texts)
syn_vocab   = vocab(syn_texts)

baseline_pool   = train_vocab
augmented_pool  = train_vocab | syn_vocab

baseline_cov  = len(test_vocab & baseline_pool)  / len(test_vocab)
augmented_cov = len(test_vocab & augmented_pool) / len(test_vocab)

print(f"Test vocab unique words : {len(test_vocab):,}")
print(f"Train vocab             : {len(train_vocab):,}")
print(f"Synthetic vocab         : {len(syn_vocab):,}")
print(f"Synthetic NEW vs train  : {len(syn_vocab - train_vocab):,} ({100*len(syn_vocab - train_vocab)/len(syn_vocab):.1f}%)")
print()
print(f"Baseline coverage  (train only)        : {baseline_cov:.1%}  ({len(test_vocab & baseline_pool):,} / {len(test_vocab):,})")
print(f"Augmented coverage (train + synthetic) : {augmented_cov:.1%}  ({len(test_vocab & augmented_pool):,} / {len(test_vocab):,})")
print(f"IMPROVEMENT                             : +{(augmented_cov-baseline_cov)*100:.1f} pp")
print()

# What test words does synthetic NEWLY cover?
syn_only_cover = (test_vocab & syn_vocab) - train_vocab
print(f"Test words NEWLY covered by synthetic (not in train): {len(syn_only_cover):,}")
print(f"  -> {(len(syn_only_cover)/len(test_vocab))*100:.1f}% of test vocab")
print(f"  Sample 30: {sorted(list(syn_only_cover))[:30]}")
print()

# Wasted vocab? (synthetic vocab not in test)
syn_unused = syn_vocab - test_vocab - train_vocab
print(f"Synthetic NEW vocab NOT in test (potentially 'wasted'): {len(syn_unused):,} ({100*len(syn_unused)/len(syn_vocab):.1f}% of syn vocab)")
print(f"  Note: tidak benar-benar wasted - bisa berguna untuk generalization")

# ===================================================================
# EXPERIMENT 2: DISTINCT-N (Li et al. 2016)
# ===================================================================
print("\n" + "="*70)
print("EXPERIMENT 2 — DISTINCT-N DIVERSITY (Li et al. 2016)")
print("="*70)
print("Reference: 'A Diversity-Promoting Objective Function for Neural Conversation Models'")
print("Metric: |unique_n_grams| / |total_n_grams|. Higher = more diverse.")
print()

def distinct_n(texts, n=1):
    grams = []
    for t in texts:
        toks = tokenize(t)
        if len(toks) >= n:
            grams.extend([tuple(toks[i:i+n]) for i in range(len(toks)-n+1)])
    if not grams:
        return 0.0, 0, 0
    unique = len(set(grams))
    total  = len(grams)
    return unique / total, unique, total

print(f"{'Dataset':<20} {'Distinct-1':>15} {'Distinct-2':>15} {'Distinct-3':>15}")
for name, texts in [('Seed train', train_texts), ('Synthetic', syn_texts), ('Combined', train_texts + syn_texts)]:
    d1, u1, t1 = distinct_n(texts, 1)
    d2, u2, t2 = distinct_n(texts, 2)
    d3, u3, t3 = distinct_n(texts, 3)
    print(f"{name:<20} {d1:>15.4f} {d2:>15.4f} {d3:>15.4f}")
print()
print("Interpretation:")
print("  Higher Distinct-N = more diverse output. Synthetic should be similar/higher than seed.")
print("  Drop di Distinct-2/3 indicates template repetition.")

# ===================================================================
# EXPERIMENT 3: TYPE-TOKEN RATIO + MTLD-inspired
# ===================================================================
print("\n" + "="*70)
print("EXPERIMENT 3 — TYPE-TOKEN RATIO (TTR)")
print("="*70)
print("Standard lexical diversity. Higher TTR = more varied vocabulary.")
print("Caveat: TTR sensitive to corpus length. Compare same-size.")
print()

def ttr(texts):
    toks = all_tokens(texts)
    if not toks:
        return 0.0, 0, 0
    return len(set(toks)) / len(toks), len(set(toks)), len(toks)

# Subsample synthetic to seed size for fair comparison
np.random.seed(42)
syn_subsample = list(np.random.choice(syn_texts, size=len(train_texts), replace=False))

for name, texts in [
    ('Seed train', train_texts),
    ('Synthetic (full 2499)', syn_texts),
    (f'Synthetic (subsample {len(train_texts)})', syn_subsample),
]:
    t, types, tokens = ttr(texts)
    print(f"  {name:<35} TTR={t:.4f}  types={types:,}  tokens={tokens:,}")

# ===================================================================
# EXPERIMENT 4: PER-SENTENCE NOVELTY SCORE
# ===================================================================
print("\n" + "="*70)
print("EXPERIMENT 4 — PER-SENTENCE NOVELTY SCORE")
print("="*70)
print("For each synthetic sentence: how many tokens are NEW vs seed train vocab?")
print("Use case: filter out sentences that introduce 0 new vocab.")
print()

novelty_scores = []
for s in syn_texts:
    toks = set(tokenize(s))
    new_toks = toks - train_vocab
    novelty_scores.append(len(new_toks))

novelty_scores = np.array(novelty_scores)
print(f"Distribution of new tokens per synthetic sentence:")
print(f"  Mean       : {novelty_scores.mean():.2f}")
print(f"  Median     : {np.median(novelty_scores):.0f}")
print(f"  Std        : {novelty_scores.std():.2f}")
print(f"  Min / Max  : {novelty_scores.min()} / {novelty_scores.max()}")
print()
for thr in [0, 1, 2, 3, 5]:
    n_below = int((novelty_scores < thr).sum())
    pct = 100 * n_below / len(novelty_scores)
    print(f"  Sentences with < {thr} new tokens: {n_below:,} ({pct:.1f}%)")
print()
print("Interpretation:")
print("  Filter MIN_NEW_TOKENS=1 -> reject sentences not introducing vocabulary")
print("  Higher MIN_NEW = more aggressive filter, may over-restrict")

# ===================================================================
# EXPERIMENT 5: LENGTH DISTRIBUTION COMPARISON
# ===================================================================
print("\n" + "="*70)
print("EXPERIMENT 5 — LENGTH DISTRIBUTION")
print("="*70)
print("Augmentation should match seed length distribution to avoid distribution shift.")
print()

def length_stats(texts, name):
    lengths = np.array([len(tokenize(t)) for t in texts])
    return {
        'name': name,
        'mean': lengths.mean(),
        'std': lengths.std(),
        'median': float(np.median(lengths)),
        'p10': float(np.percentile(lengths, 10)),
        'p90': float(np.percentile(lengths, 90)),
        'min': int(lengths.min()),
        'max': int(lengths.max()),
    }

print(f"{'Dataset':<15} {'mean':>8} {'std':>8} {'median':>8} {'p10':>6} {'p90':>6} {'min':>5} {'max':>5}")
for d in [length_stats(train_texts, 'Seed train'), length_stats(syn_texts, 'Synthetic'), length_stats(test_texts, 'Test')]:
    print(f"{d['name']:<15} {d['mean']:>8.1f} {d['std']:>8.1f} {d['median']:>8.0f} {d['p10']:>6.0f} {d['p90']:>6.0f} {d['min']:>5} {d['max']:>5}")
print()

# KL divergence on length bins
def length_dist(texts, bins):
    lengths = np.array([len(tokenize(t)) for t in texts])
    hist, _ = np.histogram(lengths, bins=bins)
    return hist / hist.sum()

bins = list(range(0, 50, 2)) + [100]
p_seed = length_dist(train_texts, bins) + 1e-10
p_syn  = length_dist(syn_texts, bins) + 1e-10
kl = np.sum(p_seed * np.log(p_seed / p_syn))
print(f"KL(seed || synthetic) on length bins: {kl:.4f}")
print(f"  KL near 0 = distributions match. KL > 0.1 = noticeable shift.")

# ===================================================================
# EXPERIMENT 6: VOCABULARY ENTROPY (Shannon)
# ===================================================================
print("\n" + "="*70)
print("EXPERIMENT 6 — VOCABULARY ENTROPY (Shannon)")
print("="*70)
print("H = -sum(p_i * log p_i). Higher = more uniform usage = less repetition.")
print()

def vocab_entropy(texts):
    toks = all_tokens(texts)
    counts = Counter(toks)
    total = sum(counts.values())
    probs = np.array([c/total for c in counts.values()])
    return -np.sum(probs * np.log2(probs))

for name, texts in [('Seed train', train_texts), ('Synthetic', syn_texts), ('Combined', train_texts + syn_texts)]:
    h = vocab_entropy(texts)
    print(f"  {name:<20} entropy={h:.3f} bits")

# ===================================================================
# SUMMARY: REKOMENDASI FILTER
# ===================================================================
print("\n" + "="*70)
print("RINGKASAN & REKOMENDASI FILTER")
print("="*70)
print(f"""
Hasil utama:
  - OOV coverage improvement   : +{(augmented_cov-baseline_cov)*100:.1f} pp ({baseline_cov:.1%} -> {augmented_cov:.1%})
  - Synthetic NEW vocab        : {100*len(syn_vocab - train_vocab)/len(syn_vocab):.1f}% (test-relevant: {len(syn_only_cover)})
  - Length KL divergence       : {kl:.4f}
  - Sentences w/ 0 new tokens  : {(novelty_scores < 1).sum()} ({100*(novelty_scores < 1).sum()/len(novelty_scores):.1f}%)

Rekomendasi filter (urutan: paling impactful):

1. TOKEN NOVELTY FILTER (high impact, simple)
   - Reject sentences with 0 new tokens vs current pool
   - Justifikasi: directly aligned with vocab expansion goal
   - Trade-off: could over-restrict at end of run when pool is large

2. LENGTH DISTRIBUTION FILTER (low impact tapi defensive)
   - Reject sentences with length outside seed p5-p95 range
   - Justifikasi: avoid distribution shift in augmented data
   - Reference: standard data quality practice

3. DISTINCT-N MONITORING (corpus-level diagnostic, bukan per-sentence filter)
   - Track Distinct-1/2/3 per batch
   - Alert kalau drop signifikan vs baseline
   - Reference: Li et al. 2016

4. JACCARD/ROUGE-L (Self-Instruct convention)
   - Threshold 0.5+ sebagai safety guardrail (catch true paraphrase)
   - Empirical evidence: max=0.28 di jav, jadi mostly no-op
   - Reference: Wang et al. 2022 ROUGE-L 0.7

CATATAN: Untuk thesis, kombinasi:
  - SV + LV (validators)        — semantic & linguistic correctness
  - Token Novelty (MIN_NEW>=1)  — vocab expansion focus
  - Length filter               — distribution match
  - Jaccard 0.5 (current)       — guardrail

Atau lebih simpel: pakai SV + LV saja, filter tambahan optional.
""")
