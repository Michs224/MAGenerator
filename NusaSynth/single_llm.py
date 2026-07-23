"""Single-LLM baseline: generate langsung dari seed, tanpa Contextualizer/SV/LV/PSLM/dedup.

Menambah, tidak mengubah pipeline penuh. Rasional: agent_doc/research/baseline_plans/01_single_llm.md.
"""

from __future__ import annotations

import csv
import json

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from NusaSynth.config import (
    GEMINI_MODEL,
    LABEL_ORDER,
    LANG_NAMES,
    PARSE_MAX_RETRY,
    SENTENCES_PER_SEED,
    SINGLE_LLM_OUTPUT_DIR,
)
from NusaSynth.state import BatchState, make_sentence
from NusaSynth.usage_tracker import get_run_id, init_usage_tracker, invoke_structured_tracked


class SeedVariations(BaseModel):
    seed_id: int = Field(description="The seed id these variations belong to (copied from input)")
    variations: list[str] = Field(
        description=f"Exactly {SENTENCES_PER_SEED} new sentences varying the seed",
        min_length=SENTENCES_PER_SEED,
        max_length=SENTENCES_PER_SEED,
    )


class DirectGeneratorOutput(BaseModel):
    results: list[SeedVariations] = Field(description="One entry per input seed")


def build_generator_direct_messages(seeds, seed_profile, lang_name, target_label):
    """Prompt Single-LLM: aturan generator dipertahankan, minus variation_plans."""
    al = seed_profile["avg_length_per_label"]
    seed_ref = "\n".join(f'Seed {s["id"]} ({len(s["text"].split())} words): "{s["text"]}"' for s in seeds)
    n_sentences = len(seeds) * SENTENCES_PER_SEED

    system = f"""You are a native-level {lang_name} text generator.

Generate {SENTENCES_PER_SEED} new sentences for EACH of the {len(seeds)} seeds below. Total: {n_sentences} sentences.

Seed Profile ({lang_name}) - length reference per label:
- negative: avg {al['negative']['mean']} words (std {al['negative']['std']})
- neutral:  avg {al['neutral']['mean']} words (std {al['neutral']['std']})
- positive: avg {al['positive']['mean']} words (std {al['positive']['std']})
- The seed reference sentences below show concrete examples - observe each seed's elaboration depth

Rules:
- Express {target_label} sentiment in {lang_name}
- For every generated sentence, copy back the seed_id it is based on
- Generate exactly {SENTENCES_PER_SEED} sentences per seed - every input seed_id must appear exactly once
- Stay within each seed's own domain and style - infer them from the seed text itself, and do not shift to a different domain
- Preserve the seed's underlying sentiment even when it is expressed indirectly (through situation, consequence, or comparison)
- Make each variation lexically novel, label-stable, and plausible within the seed's domain - without copying the seed's phrases verbatim
- Match the style and tone of the seeds - your output must sound like it comes from the same data source
- Code-mixing with Indonesian is common here - match the seed's natural code-mixing
- Length matching:
  * Each generated sentence should preserve its source seed's elaboration depth (judged from the seed text itself)
  * Brief seed → variations of similar brevity
  * Elaborate seed → variations with similar elaboration
  * Natural slight variation around the seed's depth is acceptable
  * The aggregate batch naturally spans lengths because input seeds vary in elaboration depth
  * Avoid producing sentences with uniform length regardless of source seed
- CRITICAL: every sentence must be structurally distinct - do NOT repeat the same template with swapped words.

The {len(seeds)} seeds below are your style reference - study their vocabulary, tone, and length:

{seed_ref}"""

    seeds_json = json.dumps(
        [{"id": s["id"], "text": s["text"], "label": s["label"]} for s in seeds],
        indent=2,
        ensure_ascii=False,
    )
    user = (
        f"Generate {SENTENCES_PER_SEED} {target_label} {lang_name} sentences for each of "
        f"these {len(seeds)} seeds:\n\nSeeds:\n{seeds_json}"
    )
    return [("system", system), ("human", user)]


def generator_direct_node(state: BatchState) -> dict:
    """Generate 5 kalimat/seed, validasi bijection seed_id, tulis langsung ke all_accepted."""
    from NusaSynth.nodes import get_llm

    structured_llm = get_llm().with_structured_output(DirectGeneratorOutput, include_raw=True)
    messages = build_generator_direct_messages(
        state["seeds"], state["seed_profile"], state["lang_name"], state["target_label"]
    )
    expected_ids = {s["id"] for s in state["seeds"]}

    def is_valid(result) -> bool:
        returned = [r.seed_id for r in result.results]
        return (
            len(returned) == len(expected_ids)
            and set(returned) == expected_ids
            and all(len(r.variations) == SENTENCES_PER_SEED for r in result.results)
        )

    def invoke():
        return invoke_structured_tracked(
            structured_llm, messages,
            agent_display="Generator(single-LLM)", fail_consequence="Batch dibuang.",
            lang=state["target_lang"], label=state["target_label"],
            model=GEMINI_MODEL, parse_max_retry=PARSE_MAX_RETRY,
        )

    result = invoke()
    if result is not None and not is_valid(result):
        print(f"Generator(single-LLM) WARNING: copy-back invalid (dapat {[r.seed_id for r in result.results]}). Coba lagi.")
        result = invoke()
    if result is None or not is_valid(result):
        print("Generator(single-LLM) FAILED: batch dibuang.")
        return {"all_accepted": [], "current_sentences": []}

    next_sid = state.get("next_sid", 0)
    sentences = []
    for r in result.results:
        for text in r.variations:
            sentences.append(make_sentence(
                sid=next_sid, seed_id=r.seed_id, text=text,
                target_label=state["target_label"], target_lang=state["target_lang"],
            ))
            next_sid += 1

    if sentences:
        print(f"Generator(single-LLM): {len(sentences)} kalimat (sids {sentences[0]['sid']}-{sentences[-1]['sid']})")
    else:
        print("Generator(single-LLM): 0 kalimat")
    return {"all_accepted": sentences, "next_sid": next_sid, "current_sentences": []}


def build_single_llm_pipeline():
    """START -> generator_direct -> END. Tanpa validator/retry/collect."""
    builder = StateGraph(BatchState)
    builder.add_node("generator_direct", generator_direct_node)
    builder.add_edge(START, "generator_direct")
    builder.add_edge("generator_direct", END)
    return builder.compile()


def _save(lang, label, accepted, run_dir):
    """Append accepted -> synthetic.csv + pipeline_log.jsonl (tanpa dedup/PSLM)."""
    lang_dir = run_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    csv_path = lang_dir / "synthetic.csv"
    log_path = lang_dir / "pipeline_log.jsonl"

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        if write_header:
            writer.writeheader()
        for s in accepted:
            writer.writerow({"id": s["sid"], "text": s["text"], "label": label})

    with open(log_path, "a", encoding="utf-8") as f:
        for s in accepted:
            f.write(json.dumps({**dict(s), "outcome": "accepted"}, ensure_ascii=False) + "\n")
    return len(accepted)


def run_single_llm_language(lang, labels=None, sample_ratio=None):
    """Jalankan Single-LLM per bahasa (save-only). sample_ratio<1.0 untuk smoke test."""
    import pandas as pd

    from NusaSynth.config import BATCH_SIZE, DATA_DIR
    from NusaSynth.run_pipeline import chunk, compute_seed_profile, sample_seeds

    init_usage_tracker(SINGLE_LLM_OUTPUT_DIR.parent / "usage_single_llm")
    run_id = get_run_id()
    run_dir = SINGLE_LLM_OUTPUT_DIR / run_id if run_id else SINGLE_LLM_OUTPUT_DIR
    print(f"[single-LLM] Output -> {run_dir}/<lang>/")

    langs = [lang] if isinstance(lang, str) else list(lang)
    pipeline = build_single_llm_pipeline()
    target_labels = labels or LABEL_ORDER

    for i, single_lang in enumerate(langs):
        lang_name = LANG_NAMES[single_lang]
        print(f"\n{'#'*60}\n# [single-LLM] Bahasa {i+1}/{len(langs)}: {single_lang}\n{'#'*60}")
        train_path = DATA_DIR / single_lang / "train.csv"
        if not train_path.exists():
            print(f"ERROR: train tidak ada: {train_path}")
            continue
        train_df = pd.read_csv(train_path)
        profile = compute_seed_profile(train_df, single_lang)

        total = 0
        for label in target_labels:
            seeds = sample_seeds(train_df[train_df["label"] == label].to_dict("records"), sample_ratio)
            batches = chunk(seeds, BATCH_SIZE)
            print(f"\n--- {lang_name} [{label}]: {len(seeds)} seeds -> {len(batches)} batch ---")
            for bi, batch in enumerate(batches):
                print(f"\nBatch {bi+1}/{len(batches)} [{single_lang} {label}]: seed_ids={[s['id'] for s in batch]}")
                result = pipeline.invoke(
                    {"seeds": batch, "seed_profile": profile, "target_label": label,
                     "target_lang": single_lang, "lang_name": lang_name},
                    {"recursion_limit": 10},
                )
                accepted = result.get("all_accepted", [])
                print(f"  -> {len(accepted)} kalimat")
                total += _save(single_lang, label, accepted, run_dir)

        print(f"\n[single-LLM] {lang_name} selesai: {total} kalimat -> {run_dir / single_lang / 'synthetic.csv'}")
