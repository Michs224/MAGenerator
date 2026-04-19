"""Analisis hubungan train (seed+syn) vs val vs test untuk diagnose training results.

Output:
1. Length distribution across all 4 sets
2. Label distribution
3. Vocab overlap matrix
4. Identify specific quality issues di synthetic
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
TRAIN_SYN3 = ROOT / 'data/nusax_senti/jav/train_syn3.csv'
SEED_VALID = ROOT / 'data/nusax_senti/jav/valid.csv'
SEED_TEST  = ROOT / 'data/nusax_senti/jav/test.csv'
SYN_ONLY   = ROOT / 'outputs/synthetic/jav/synthetic.csv'

def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

def vocab(texts):
    v = set()
    for t in texts:
        v |= set(tokenize(t))
    return v

def length_stats(texts, name):
    lens = np.array([len(tokenize(t)) for t in texts])
    return {
        'name': name, 'n': len(texts),
        'mean': lens.mean(), 'std': lens.std(),
        'median': float(np.median(lens)),
        'p5': float(np.percentile(lens, 5)),
        'p25': float(np.percentile(lens, 25)),
        'p75': float(np.percentile(lens, 75)),
        'p95': float(np.percentile(lens, 95)),
        'min': int(lens.min()), 'max': int(lens.max()),
    }

# Load
train_seed = pd.read_csv(SEED_TRAIN)
train_aug  = pd.read_csv(TRAIN_SYN3)
valid      = pd.read_csv(SEED_VALID)
test       = pd.read_csv(SEED_TEST)
syn_only   = pd.read_csv(SYN_ONLY)

print("="*70)
print("DATASET SIZES")
print("="*70)
print(f"  Train (seed only)     : {len(train_seed)}")
print(f"  Train (aug=syn3)      : {len(train_aug)}  (+{len(train_aug)-len(train_seed)} synthetic)")
print(f"  Synthetic only        : {len(syn_only)}")
print(f"  Validation            : {len(valid)}")
print(f"  Test                  : {len(test)}")

# ── Label distribution
print("\n" + "="*70)
print("LABEL DISTRIBUTION")
print("="*70)
def label_dist(df, name):
    d = df['label'].value_counts(normalize=True).sort_index()
    print(f"  {name:<25}", end="")
    for lab, pct in d.items():
        print(f"  {lab}={pct:.3f}", end="")
    print()
label_dist(train_seed, "Train (seed)")
label_dist(syn_only,   "Synthetic only")
label_dist(train_aug,  "Train (aug)")
label_dist(valid,      "Validation")
label_dist(test,       "Test")

# ── Length distribution
print("\n" + "="*70)
print("LENGTH DISTRIBUTION (words)")
print("="*70)
print(f"{'Dataset':<25} {'n':>5} {'mean':>6} {'std':>6} {'p5':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'p95':>5} {'min':>5} {'max':>5}")
for stats in [
    length_stats(train_seed['text'].dropna().tolist(), 'Train (seed only)'),
    length_stats(syn_only['text'].dropna().tolist(),  'Synthetic only'),
    length_stats(train_aug['text'].dropna().tolist(), 'Train (aug)'),
    length_stats(valid['text'].dropna().tolist(),     'Validation'),
    length_stats(test['text'].dropna().tolist(),      'Test'),
]:
    print(f"  {stats['name']:<25} {stats['n']:>5} {stats['mean']:>6.1f} {stats['std']:>6.1f} {stats['p5']:>5.0f} {stats['p25']:>5.0f} {stats['median']:>5.0f} {stats['p75']:>5.0f} {stats['p95']:>5.0f} {stats['min']:>5} {stats['max']:>5}")

# ── Length histogram per dataset (rough buckets)
print("\n" + "="*70)
print("LENGTH BUCKET DISTRIBUTION (% in each range)")
print("="*70)
buckets = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 100)]
print(f"{'Dataset':<25}", end="")
for lo, hi in buckets:
    print(f"  {lo:>2}-{hi:>3}", end="")
print()
for name, df in [
    ('Train (seed)', train_seed),
    ('Synthetic only', syn_only),
    ('Train (aug)', train_aug),
    ('Validation', valid),
    ('Test', test),
]:
    lens = np.array([len(tokenize(t)) for t in df['text'].dropna()])
    print(f"  {name:<25}", end="")
    for lo, hi in buckets:
        pct = 100 * np.sum((lens >= lo) & (lens <= hi)) / len(lens)
        print(f"  {pct:>5.1f}%", end="")
    print()

# ── Vocabulary overlap matrix
print("\n" + "="*70)
print("VOCAB OVERLAP MATRIX (% of row vocab present in col vocab)")
print("="*70)
vocabs = {
    'Train(seed)':    vocab(train_seed['text'].dropna()),
    'Synthetic':      vocab(syn_only['text'].dropna()),
    'Train(aug)':     vocab(train_aug['text'].dropna()),
    'Validation':     vocab(valid['text'].dropna()),
    'Test':           vocab(test['text'].dropna()),
}
print(f"{'':<14} {'size':>7}", end="")
for col in vocabs.keys():
    print(f"  {col:>11}", end="")
print()
for row_name, row_v in vocabs.items():
    print(f"  {row_name:<12} {len(row_v):>7,}", end="")
    for col_name, col_v in vocabs.items():
        if not row_v:
            pct = 0
        else:
            pct = 100 * len(row_v & col_v) / len(row_v)
        print(f"  {pct:>10.1f}%", end="")
    print()

# ── Test-relevant analysis
print("\n" + "="*70)
print("TEST GENERALIZATION ANALYSIS")
print("="*70)

test_v = vocabs['Test']
val_v  = vocabs['Validation']

# OOV vs each pool
for pool_name in ['Train(seed)', 'Train(aug)']:
    pool_v = vocabs[pool_name]
    oov_test = test_v - pool_v
    oov_val  = val_v  - pool_v
    cov_test = 1 - len(oov_test)/len(test_v)
    cov_val  = 1 - len(oov_val) /len(val_v)
    print(f"\n  Pool = {pool_name}:")
    print(f"    Test vocab covered : {cov_test:.1%}  ({len(test_v)-len(oov_test)}/{len(test_v)})")
    print(f"    Val vocab covered  : {cov_val:.1%}   ({len(val_v)-len(oov_val)}/{len(val_v)})")

# Words unique to synthetic that ALSO appear in test (the "good" syn vocab)
syn_test_overlap = vocabs['Synthetic'] & test_v
syn_val_overlap  = vocabs['Synthetic'] & val_v
print(f"\n  Synthetic vocab ∩ Test : {len(syn_test_overlap)}  ({100*len(syn_test_overlap)/len(vocabs['Synthetic']):.1f}% of syn vocab)")
print(f"  Synthetic vocab ∩ Val  : {len(syn_val_overlap)}   ({100*len(syn_val_overlap)/len(vocabs['Synthetic']):.1f}% of syn vocab)")

# ── Label-conditioned length analysis
print("\n" + "="*70)
print("LENGTH BY LABEL — apakah synthetic match seed pattern per label?")
print("="*70)
for lab in ['negative', 'neutral', 'positive']:
    print(f"\n  {lab.upper()}:")
    for name, df in [
        ('  Train (seed)', train_seed),
        ('  Synthetic',    syn_only),
        ('  Test',         test),
    ]:
        sub = df[df['label']==lab]['text'].dropna()
        if len(sub) == 0: continue
        lens = np.array([len(tokenize(t)) for t in sub])
        print(f"  {name:<18} n={len(sub):>4}  mean={lens.mean():>5.1f}  std={lens.std():>5.1f}  p5={np.percentile(lens,5):>4.0f}  p95={np.percentile(lens,95):>4.0f}")

print("\n" + "="*70)
print("DIAGNOSE SUMMARY")
print("="*70)

# Compute key diagnostics
seed_lens = np.array([len(tokenize(t)) for t in train_seed['text'].dropna()])
syn_lens  = np.array([len(tokenize(t)) for t in syn_only['text'].dropna()])
test_lens = np.array([len(tokenize(t)) for t in test['text'].dropna()])

# How many test sentences fall in synthetic-underrepresented length range?
syn_p5  = float(np.percentile(syn_lens, 5))
syn_p95 = float(np.percentile(syn_lens, 95))
test_outside_syn = np.sum((test_lens < syn_p5) | (test_lens > syn_p95))
print(f"\n  Synthetic length range [p5={syn_p5:.0f}, p95={syn_p95:.0f}] words")
print(f"  Test sentences OUTSIDE this range: {test_outside_syn}/{len(test_lens)} ({100*test_outside_syn/len(test_lens):.1f}%)")
print(f"  -> these are test samples model has 'less practice' on")

# Train (aug) effective seed-to-syn ratio per batch (16 samples)
aug_size = len(train_aug)
seed_in_aug = len(train_seed)
syn_in_aug = aug_size - seed_in_aug
seed_per_batch = 16 * seed_in_aug / aug_size
syn_per_batch  = 16 * syn_in_aug / aug_size
print(f"\n  Per training batch (size 16):")
print(f"    Seed samples  : {seed_per_batch:.1f}  ({100*seed_in_aug/aug_size:.1f}%)")
print(f"    Synthetic     : {syn_per_batch:.1f}  ({100*syn_in_aug/aug_size:.1f}%)")
print(f"  -> {100*syn_in_aug/aug_size:.0f}% training signal datang dari synthetic")
