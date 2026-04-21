"""Distinct-N analysis untuk train/val/test/synthetic Javanese.

Purpose: dapat baseline Distinct-N di natural data (train/val/test) untuk decide
apakah synthetic Distinct-N "concerning" atau dalam range natural.

Reference: Li et al. 2016 "A Diversity-Promoting Objective Function for Neural
Conversation Models"

Distinct-N = |unique n-grams| / |total n-grams|. Higher = more diverse.

Caveat: Distinct-N size-sensitive (larger corpus -> lower Distinct-N karena lebih
banyak chance repeating). Need apples-to-apples sample size untuk fair comparison.
"""
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path('y:/Michh/Python/Projects/MAGenerator')


def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())


def distinct_n(texts, n=1):
    grams = []
    for t in texts:
        toks = tokenize(t)
        if len(toks) >= n:
            grams.extend([tuple(toks[i:i+n]) for i in range(len(toks)-n+1)])
    if not grams:
        return 0.0, 0, 0
    unique = len(set(grams))
    total = len(grams)
    return unique / total, unique, total


def analyze(name, texts, indent=2):
    """Print Distinct-1/2/3 for a corpus."""
    if not texts:
        return
    pad = ' ' * indent
    d1, u1, t1 = distinct_n(texts, 1)
    d2, u2, t2 = distinct_n(texts, 2)
    d3, u3, t3 = distinct_n(texts, 3)
    print(f"{pad}{name:<35} n_sents={len(texts):>4}  D1={d1:.4f} ({u1}/{t1})  D2={d2:.4f} ({u2}/{t2})  D3={d3:.4f} ({u3}/{t3})")


def main():
    train = pd.read_csv(ROOT / 'data/nusax_senti/jav/train.csv')
    val   = pd.read_csv(ROOT / 'data/nusax_senti/jav/valid.csv')
    test  = pd.read_csv(ROOT / 'data/nusax_senti/jav/test.csv')

    syn_path = ROOT / 'outputs/synthetic/jav/synthetic.csv'
    syn = pd.read_csv(syn_path) if syn_path.exists() else None

    print("=" * 100)
    print("DISTINCT-N FULL CORPUS (raw, NOT size-controlled)")
    print("=" * 100)
    print()
    analyze("Train (500)",      train['text'].dropna().tolist())
    analyze("Validation (100)", val  ['text'].dropna().tolist())
    analyze("Test (400)",       test ['text'].dropna().tolist())
    if syn is not None:
        analyze(f"Synthetic ({len(syn)})", syn['text'].dropna().tolist())

    print()
    print("=" * 100)
    print("APPLES-TO-APPLES — Sample to common size = 100 sentences (= val size)")
    print("=" * 100)
    print()

    common_n = 100
    rng = np.random.default_rng(42)

    def sample(texts, n):
        if len(texts) <= n:
            return texts
        idx = rng.choice(len(texts), size=n, replace=False)
        return [texts[i] for i in idx]

    analyze(f"Train sample ({common_n})",      sample(train['text'].dropna().tolist(), common_n))
    analyze(f"Validation ({len(val)})",        val['text'].dropna().tolist())
    analyze(f"Test sample ({common_n})",       sample(test['text'].dropna().tolist(), common_n))
    if syn is not None:
        analyze(f"Synthetic sample ({common_n})", sample(syn['text'].dropna().tolist(), common_n))

    print()
    print("=" * 100)
    print("APPLES-TO-APPLES — Sample to common size = 400 sentences (= test size)")
    print("=" * 100)
    print()

    common_n = 400

    analyze(f"Train sample ({common_n})", sample(train['text'].dropna().tolist(), common_n))
    analyze(f"Test ({len(test)})",         test['text'].dropna().tolist())
    if syn is not None:
        analyze(f"Synthetic sample ({common_n})", sample(syn['text'].dropna().tolist(), common_n))

    print()
    print("=" * 100)
    print("PER-LABEL DISTINCT-N (within label, full corpus)")
    print("=" * 100)
    for lab in ['negative', 'neutral', 'positive']:
        print(f"\n  {lab.upper()}:")
        analyze("Train",      train[train['label']==lab]['text'].dropna().tolist(), indent=4)
        analyze("Validation", val  [val  ['label']==lab]['text'].dropna().tolist(), indent=4)
        analyze("Test",       test [test ['label']==lab]['text'].dropna().tolist(), indent=4)
        if syn is not None:
            sub = syn[syn['label']==lab]['text'].dropna().tolist()
            if sub:
                analyze("Synthetic", sub, indent=4)

    print()
    print("=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print("""
  - Distinct-N tinggi = corpus diverse (n-gram patterns rarely repeat)
  - Distinct-N rendah = corpus repetitive (templates, common phrases)
  - Reference: Li et al. 2016 menggunakan threshold informal:
      D-1 > 0.05 = OK
      D-2 > 0.20 = OK
  - Yang lebih penting: COMPARISON synthetic vs natural di sample size sama
  - Kalau synthetic Distinct-N significantly LOWER dari natural -> template repetition
  - Kalau dalam range natural variation -> tidak need filter

  Decision rule:
    syn_D2 / natural_D2 < 0.85 (drop > 15%) -> N-gram repetition filter useful
    syn_D2 / natural_D2 >= 0.85               -> filter not needed
    """)


if __name__ == '__main__':
    main()
