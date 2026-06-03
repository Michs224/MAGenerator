# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**NusaSynth (MAGenerator)** — A multi-agent framework for generating synthetic sentiment analysis data for low-resource Indonesian local languages. Master's thesis research project at BINUS Graduate Program (Michael Geraldin Wijaya, 2702750546) that augments the NusaX-Senti dataset (~500 train samples per language) using Agentic AI.

**Target Languages (Proposal Scope):** 4 languages — Javanese (jav), Sundanese (sun), Acehnese (ace), Toba Batak (bbc). NusaX-Senti dataset has 12 languages total but proposal focuses on these 4 based on NusaBERT baseline F1 highest scores.

## Development Setup

- **Python:** >=3.12
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Install deps:** `uv sync`
- **Run pipeline:** `uv run python -m NusaSynth.run_pipeline`
- **LLM API:** Google Gemini (env var `GOOGLE_API_KEY` required)

No test framework or linter configured.

## Architecture (Current State — Implemented)

Pipeline = 4 LLM agents + 2 statistical filters + 1 cross-batch dedup, with feedback loop:

1. **Contextualizer Agent** (LLM Gemini) — Per-seed analysis: domain, style, sentiment expression. Generates N variation plans per seed (strategy + preserve).
2. **Generator Agent** (LLM Gemini) — Receives variation plans + seed reference + seed profile. Generates N synthetic sentences per seed with length matching guidance.
3. **Sentiment Validator (SV)** (LLM + NusaBERT signal) — Chain-of-Thought reasoning grounded by NusaBERT classifier (fine-tuned on NusaX-Senti train).
4. **Linguistic Validator (LV)** (LLM + GlotLID signal) — Naturalness assessment grounded by GlotLID language identification.
5. **PSLM Filter** (statistical) — Per-Sentence Length Match: rejects if |gen_len − seed_len| > 1.0 × seed_std per label.
6. **Cross-batch Deduplication** (statistical, save stage) — Jaccard bigram similarity ≥ 0.5 → excluded as too similar.

Rejected samples loop back to Generator with diagnostic feedback (max 2 retries).

### Key Models Used

- **LLM (agent inference):** Google Gemini (commercial pre-trained, no fine-tuning)
- **NusaBERT** (Wongso et al., 2025): SV signal grounding (fine-tuned on NusaX-Senti) + downstream evaluation backbone
- **GlotLID** (Kargaran et al., 2023): LV signal grounding (1665 languages supported)
- **XLM-RoBERTa**: Multilingual baseline for downstream comparison

## Data

- **Source:** NusaX-Senti from IndoNLP (Hugging Face `datasets`)
- **Location:** `data/nusax_senti/<lang_code>/{train,valid,test}.csv`
- **Format:** CSV with `id`, `text`, `label` (negative/neutral/positive)
- **Per language:** 500 train, 100 valid, 400 test
- **Synthetic output:** `outputs/synthetic/<lang>/synthetic.csv` + `pipeline_log.jsonl` (full audit trail)

## Project Structure

```
MAGenerator/
├── NusaSynth/              # Main Python package (pipeline code + thesis document folder)
│   ├── config.py           # Pipeline constants (BATCH_SIZE=5, MAX_RETRY=2, DEDUP_THRESHOLD=0.5)
│   ├── nodes.py            # LangGraph nodes: contextualizer, generator, collect (incl. PSLM)
│   ├── prompts.py          # Pydantic schemas + prompt builders for all 4 agents
│   ├── sv_graph.py         # Sentiment Validator subgraph (with NusaBERT signal)
│   ├── lv_graph.py         # Linguistic Validator subgraph (with GlotLID signal)
│   ├── graph.py            # Top-level LangGraph pipeline composition
│   ├── run_pipeline.py     # Entry point: per-language loop + save_results (cross-batch dedup)
│   ├── state.py            # BatchState TypedDict + SentenceRecord
│   ├── tools.py            # jaccard_bigram + helpers
│   └── document/           # Thesis proposal PDF + DOCX
├── agent_doc/              # Design docs, justifications, evaluation strategy (see project_navigation memory)
├── notebook/               # Jupyter notebooks (analysis, evaluation, post-run check)
├── scripts/                # Standalone scripts (fine-tune NusaBERT, analysis utilities)
├── outputs/
│   ├── synthetic/          # Generated synthetic data per language
│   └── nusabert-sentiment_seed_42/  # Fine-tuned NusaBERT for SV signal
├── data/nusax_senti/       # NusaX-Senti dataset per language
├── models/                 # Pre-trained model weights (GlotLID etc.)
└── main.py                 # Convenience entry (delegates to run_pipeline)
```

## Implementation Status

- ✅ Pipeline fully implemented (LangGraph)
- ✅ Full jav run completed: yield 99.92% (2498/2500), Self-BLEU 0.098, Distinct-2 0.71
- ⏳ Run for sun, ace, bbc — pending
- ⏳ Downstream fine-tuning evaluation (NusaBERT + XLM-R, baseline vs augmented) — pending
- 📝 Proposal in revision (see `NusaSynth/document/Outline_Thesis_Master_NusaSynth_*.{pdf,docx}`)

## Key Conventions

- **Print over logging** — user prefers `print()` for nusasynth/ files (no logging module)
- **No em-dash in proposal** — replace with comma/yaitu (style preference)
- **Italics PUEBI** — foreign technical terms italicized; proper nouns and acronyms not italicized
- **Citation 5-year preference** — prefer 2021+ papers, but foundational papers (Vaswani 2017, Koto 2020, Sokolova 2009) exception OK

## Pointer to Detailed Docs

For non-trivial decisions, design rationale, and evaluation methodology, see `agent_doc/` directory. For navigation map of all docs, check user memory (`feedback_thesis_writing_style.md`, `project_navigation.md` if exists).
