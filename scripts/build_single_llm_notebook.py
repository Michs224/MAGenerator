"""Bangun notebook/single_llm_metrics.ipynb dari bab_iv_metrics.ipynb + loader Single-LLM.

Sel metrik (Self-BLEU/BLEU/R_sigma/Delta_mu/len_stats) DISALIN VERBATIM (by-marker) demi
komparabilitas. bab_iv_metrics.ipynb TIDAK disentuh.

  uv run --with nbformat python scripts/build_single_llm_notebook.py
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
SRC = nbformat.read(ROOT / "notebook" / "bab_iv_metrics.ipynb", as_version=4)


def find_cell(marker: str) -> str:
    for c in SRC.cells:
        if c.cell_type == "code" and marker in c.source:
            return c.source
    raise SystemExit(f"marker tak ketemu di bab_iv_metrics.ipynb: {marker!r}")


bleu_funcs = find_cell("def self_bleu")
bleu_table = find_cell("METRICS = [('self_syn'")
rsigma_dmu = find_cell("s_syn.std() / s_seed.std()")
len_stats = find_cell("def len_stats")

SETUP = """import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

pd.set_option('display.max_colwidth', 150)
pd.set_option('display.width', 200)

ROOT = Path('y:/Michh/Python/Projects/MAGenerator')
sys.path.insert(0, str(ROOT))
from NusaSynth.config import single_llm_synthetic_dir, synthetic_dir

LANGS = ['ace', 'ban', 'bjn', 'jav', 'mad', 'min', 'sun']
SENTENCES_PER_SEED = 5
BATCH_SIZE = 5

seed_dfs, syn_dfs, valid_dfs, test_dfs = {}, {}, {}, {}
for lang in LANGS:
    seed_p = ROOT / f'data/nusax_senti/{lang}/train.csv'
    syn_p = single_llm_synthetic_dir(lang) / 'synthetic.csv'
    valid_p = ROOT / f'data/nusax_senti/{lang}/valid.csv'
    test_p = ROOT / f'data/nusax_senti/{lang}/test.csv'
    if not (seed_p.exists() and syn_p.exists()):
        print(f'  [{lang}] SKIP: file tidak lengkap ({syn_p})')
        continue
    seed_dfs[lang] = pd.read_csv(seed_p)
    syn_dfs[lang] = pd.read_csv(syn_p)
    if valid_p.exists(): valid_dfs[lang] = pd.read_csv(valid_p)
    if test_p.exists(): test_dfs[lang] = pd.read_csv(test_p)
    print(f'  [{lang}] seed={len(seed_dfs[lang])} syn={len(syn_dfs[lang])}')
print('Sumber synthetic = SINGLE-LLM ->', single_llm_synthetic_dir(LANGS[0]).parent)"""

AUDIT = """run_dir = single_llm_synthetic_dir(LANGS[0]).parent
audit_path = run_dir / 'audit_summary.json'
if audit_path.exists():
    audit = pd.DataFrame(json.loads(audit_path.read_text(encoding='utf-8')))
    print('Audit classifier Single-LLM (dari scripts/single_llm_audit.py):')
    display(audit)
    print('\\nmean 7 bahasa: dup={:.2f}%  lang_confusion={:.2f}%  label_drift={:.2f}%'.format(
        audit['dup_rate'].mean(), audit['lang_confusion'].mean(), audit['label_drift'].mean()))
else:
    print(f'Belum ada {audit_path} -> jalankan dulu: uv run python scripts/single_llm_audit.py')"""

COMPARE = """def mu_selfbleu4(get_dir):
    vals = []
    for lang in LANGS:
        seed_df = seed_dfs.get(lang)
        if seed_df is None:
            continue
        p = get_dir(lang) / 'synthetic.csv'
        if not p.exists():
            continue
        syn_df = pd.read_csv(p)
        per = []
        for label in ['negative', 'neutral', 'positive']:
            seed_t = seed_df[seed_df['label'] == label]['text'].dropna().tolist()
            syn_all = syn_df[syn_df['label'] == label]['text'].dropna().tolist()
            if len(syn_all) < 2 or len(seed_t) < 2:
                continue
            per.append(self_bleu(match_sample(syn_all, len(seed_t)))[4])
        if per:
            vals.append(float(np.mean(per)))
    return round(float(np.mean(vals)), 4) if vals else None

print('Self-BLEU-4 mu (7 bahasa) - lebih RENDAH = lebih beragam:')
print('  Single-LLM      :', mu_selfbleu4(single_llm_synthetic_dir))
print('  NusaSynth penuh :', mu_selfbleu4(synthetic_dir))"""

nb = new_notebook()
nb.cells = [
    new_markdown_cell(
        "# NusaSynth - Metrik Single-LLM (baseline ablation)\n\n"
        "Notebook TERPISAH dari `bab_iv_metrics.ipynb`. Fungsi metrik disalin **verbatim** demi "
        "komparabilitas; yang beda hanya loader (`syn_dfs` -> output Single-LLM). Bagian pipeline "
        "outcome & conflict TIDAK ADA (Single-LLM tanpa mesin/validator). Klaim = tingkat sistem."
    ),
    new_markdown_cell("## Setup"),
    new_code_cell(SETUP),
    new_markdown_cell(
        "## B. Keberagaman & Kebaruan - Self-BLEU (pers. 3.11) & BLEU vs Seed (pers. 3.12)\n"
        "> Fungsi identik `bab_iv_metrics.ipynb` (disalin verbatim)."
    ),
    new_code_cell(bleu_funcs),
    new_code_cell(bleu_table),
    new_markdown_cell(
        "## C. Kesesuaian Distribusi Panjang - R_sigma (3.13) & Delta_mu (3.14)\n"
        "> Single-LLM tanpa PSLM -> R_sigma diperkirakan jauh dari 1 (verdict 'CHECK' wajar, itu kontrafaktualnya)."
    ),
    new_code_cell(rsigma_dmu),
    new_code_cell(len_stats),
    new_markdown_cell("## D. Audit Classifier (GlotLID purity, NusaBERT label-drift, duplication)"),
    new_code_cell(AUDIT),
    new_markdown_cell("## E. Perbandingan vs NusaSynth Penuh"),
    new_code_cell(COMPARE),
]

out = ROOT / "notebook" / "single_llm_metrics.ipynb"
nbformat.write(nb, out)
print(f"Notebook ditulis -> {out} ({len(nb.cells)} sel)")
