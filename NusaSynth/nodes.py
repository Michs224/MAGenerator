"""LangGraph nodes for the parent pipeline: Contextualizer, Generator, Collect.

SV and LV are separate subgraphs (sv_graph.py, lv_graph.py).
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from NusaSynth.config import DEDUP_THRESHOLD, GEMINI_MODEL, MAX_RETRY
from NusaSynth.prompts import (
    ContextualizerOutput,
    GeneratorOutput,
    build_contextualizer_messages,
    build_generator_messages,
)
from NusaSynth.state import BatchState, SentenceRecord, make_sentence
from NusaSynth.tools import jaccard_bigram


def get_llm() -> ChatGoogleGenerativeAI:
    """Create LLM instance. Uses GOOGLE_API_KEY env var."""
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)


# ── Node 1: Contextualizer ─────────────────────────────────────────────────


def contextualizer_node(state: BatchState) -> dict:
    """Analyze 5 seeds → produce 25 variation plans.

    Runs once per batch (skipped on retry via graph routing).
    Validates copy-back: every input seed_id must appear exactly once in output.
    Retries once on validation failure; discards batch if still invalid.
    """

    llm = get_llm()
    structured_llm = llm.with_structured_output(ContextualizerOutput)
    messages = build_contextualizer_messages(
        seeds=state["seeds"],
        seed_profile=state["seed_profile"],
        lang_name=state["lang_name"],
    )
    expected_ids = {s["id"] for s in state["seeds"]}

    def is_valid(result: ContextualizerOutput) -> bool:
        returned = [a.seed_id for a in result.seed_analyses]
        return len(returned) == len(expected_ids) and set(returned) == expected_ids

    result: ContextualizerOutput = structured_llm.invoke(messages)
    if not is_valid(result):
        returned = [a.seed_id for a in result.seed_analyses]
        print(f"Contextualizer WARNING: copy-back invalid (dapat {returned}, seharusnya {expected_ids}). Coba lagi sekali.")
        result = structured_llm.invoke(messages)
        if not is_valid(result):
            returned = [a.seed_id for a in result.seed_analyses]
            print(f"Contextualizer FAILED: copy-back masih invalid (dapat {returned}). Batch dibuang.")
            return {"variation_plans": [], "next_sid": 0}

    # Build nested variation_plans (mirrors Pydantic SeedAnalysis structure)
    plans = []
    plan_id = 0
    for analysis in result.seed_analyses:
        seed_id = analysis.seed_id
        original_text = next(
            (s["text"] for s in state["seeds"] if s["id"] == seed_id),
            "",
        )
        variations = []
        for var in analysis.variations:
            variations.append({
                "plan_id": plan_id,
                "strategy": var.strategy,
                "preserve": var.preserve,
            })
            plan_id += 1
        plans.append({
            "seed_id": seed_id,
            "original_text": original_text,
            "domain": analysis.domain,
            "style": analysis.style,
            "sentiment_expression": analysis.sentiment_expression,
            "variations": variations,
        })

    total_variations = sum(len(p["variations"]) for p in plans)
    print(f"Contextualizer: {len(plans)} seeds → {total_variations} plans dihasilkan")
    return {"variation_plans": plans, "next_sid": 0}


# ── Node 2: Generator ──────────────────────────────────────────────────────


def generator_node(state: BatchState) -> dict:
    """Generate sentences from variation plans (first pass) or retry rejected.

    First pass: generate from all variation_plans (25 sentences).
        Validates copy-back: count of each input seed_id == # variations for that seed.
        Retries once on validation failure; discards batch if still invalid.
    Retry: regenerate only rejected sentences with feedback (no validation,
        positional matching by index in to_retry).
    """
    to_retry = state.get("to_retry", [])
    is_retry = bool(to_retry)
    plans = state["variation_plans"]

    if is_retry:
        print(f"Generator RETRY: regenerate {len(to_retry)} kalimat yang ditolak")

    # Build plan_id → plan dict (for first pass lookup of strategy/preserve + seed-level context)
    plans_by_id: dict[int, dict] = {
        var["plan_id"]: {
            **var,
            "seed_id": p["seed_id"],
            "domain": p.get("domain", ""),
            "style": p.get("style", ""),
            "sentiment_expression": p.get("sentiment_expression", ""),
        }
        for p in plans
        for var in p["variations"]
    }

    llm = get_llm()
    structured_llm = llm.with_structured_output(GeneratorOutput)
    messages = build_generator_messages(
        seeds=state["seeds"],
        seed_profile=state["seed_profile"],
        variation_plans=plans,
        lang_name=state["lang_name"],
        target_label=state["target_label"],
        retry_items=to_retry if is_retry else None,
    )

    # First-pass validation: bijection on plan_id (every input plan_id appears exactly once)
    expected_plan_ids = set(plans_by_id.keys())

    def is_valid(result: GeneratorOutput) -> bool:
        returned = [s.plan_id for s in result.sentences]
        return len(returned) == len(expected_plan_ids) and set(returned) == expected_plan_ids

    result: GeneratorOutput = structured_llm.invoke(messages)
    if not is_retry and not is_valid(result):
        returned = sorted([s.plan_id for s in result.sentences])
        print(f"Generator WARNING: plan_id bijection invalid (dapat {returned}, seharusnya {sorted(expected_plan_ids)}). Coba lagi sekali.")
        result = structured_llm.invoke(messages)
        if not is_valid(result):
            returned = sorted([s.plan_id for s in result.sentences])
            print(f"Generator FAILED: plan_id bijection masih invalid (dapat {returned}). Batch dibuang.")
            return {"current_sentences": [], "next_sid": state.get("next_sid", 0), "to_retry": []}

    # Assign sequential sids and create SentenceRecords
    next_sid = state.get("next_sid", 0)
    sentences: list[SentenceRecord] = []

    if is_retry and len(result.sentences) != len(to_retry):
        print(
            f"Generator WARNING: retry seharusnya {len(to_retry)} kalimat, dapat {len(result.sentences)}. "
            f"Pakai positional match sampai min length."
        )

    for i, item in enumerate(result.sentences):
        retry_count = 0
        prev_feedback = None

        if is_retry:
            # Retry path: positional match against to_retry
            if i >= len(to_retry):
                break  # extra sentences beyond expected count → drop
            original = to_retry[i]
            retry_count = original.get("retry_count", 0) + 1
            prev_feedback = original.get("prev_feedback")
            seed_id = original["seed_id"]              # trust source-of-truth
            plan_id = original.get("plan_id", -1)      # carry from to_retry
            strategy = original.get("strategy", "")
            preserve = original.get("preserve", "")
            domain = original.get("domain", "")
            style = original.get("style", "")
            sentiment_expression = original.get("sentiment_expression", "")
        else:
            # First pass: lookup by copied-back plan_id (bijection-validated, safe)
            plan = plans_by_id[item.plan_id]           # guaranteed valid after bijection check
            seed_id = plan["seed_id"]                  # derive from plan (trustworthy)
            plan_id = item.plan_id
            strategy = plan["strategy"]
            preserve = plan["preserve"]
            domain = plan["domain"]
            style = plan["style"]
            sentiment_expression = plan["sentiment_expression"]

        sentences.append(make_sentence(
            sid=next_sid,
            seed_id=seed_id,
            text=item.text,
            target_label=state["target_label"],
            target_lang=state["target_lang"],
            plan_id=plan_id,
            strategy=strategy,
            preserve=preserve,
            domain=domain,
            style=style,
            sentiment_expression=sentiment_expression,
            retry_count=retry_count,
            prev_feedback=prev_feedback,
        ))
        next_sid += 1

    if not sentences:
        print("Generator: 0 kalimat dihasilkan (batch kosong)")
    elif not is_retry:
        n_total = sum(len(p["variations"]) for p in plans)
        print(f"Generator: {len(sentences)}/{n_total} kalimat (sids {sentences[0]['sid']}-{sentences[-1]['sid']})")

    return {
        "current_sentences": sentences,
        "next_sid": next_sid,
        "to_retry": [],
    }


# ── Node 5: Collect ────────────────────────────────────────────────────────


def collect_node(state: BatchState) -> dict:
    """Sort validated sentences into accepted / retry / discarded.

    Steps:
    1. SV/LV verdict check → candidates, retry, or discard
    2. (DISABLED) Jaccard bigram dedup — empirically no-op pada jav baseline
       (max pairwise 0.28 < threshold 0.5 di full pairwise analysis 3.1M pairs).
       Kode lama dipertahankan sebagai komentar untuk reference dan ablation study.
    3. (ACTIVE) Per-Sentence Length Match (PSLM) filter — replaces Jaccard.
       Validates Generator's compliance with prompt directive (preserve seed
       elaboration depth). Threshold = 1.0 × seed_std per label, data-anchored
       per-language adaptive. Reference: standard statistical convention
       (±1 std covers ~68% natural variation).
    """
    sentences = state["current_sentences"]

    candidates: list[SentenceRecord] = []
    new_accepted: list[SentenceRecord] = []
    new_discarded: list[SentenceRecord] = []
    to_retry: list[SentenceRecord] = []

    # Step 1: SV/LV verdict check
    validation_failed_count = 0
    for sent in sentences:
        sv = sent.get("sv_verdict")
        lv = sent.get("lv_verdict")

        # Validation failure (bijection check failed in SV/LV) → discard, no retry
        if sv == "VALIDATION_FAILED" or lv == "VALIDATION_FAILED":
            new_discarded.append(sent)
            validation_failed_count += 1
            continue

        if lv == "PASS":
            candidates.append(sent)
        elif sent.get("retry_count", 0) >= MAX_RETRY:
            new_discarded.append(sent)
        else:
            feedback = sent.get("sv_feedback") if sv == "REJECT" else sent.get("lv_feedback")
            retry_sent = dict(sent)
            retry_sent["prev_feedback"] = feedback or "Rejected without specific feedback"
            to_retry.append(retry_sent)

    # Step 2 (DISABLED): Jaccard bigram dedup
    # ─────────────────────────────────────────────────────────────────────────
    # Filter ini DINONAKTIFKAN setelah empirical analysis menunjukkan no-op pada
    # baseline jav (max pairwise Jaccard 0.28 vs threshold 0.5; 0 hits di
    # 3.1M pairwise comparisons). Detail di agent_doc/data_distribution_analysis.md.
    #
    # Slot direncanakan untuk Per-Batch Length Variance Filter (Holtzman 2020) —
    # validates length variance per batch, addressing actual observed issue
    # (synthetic length distribution narrow std 5-6 vs seed 14).
    #
    # Reference pool dan kode lama dipertahankan untuk facilitate ablation study.
    # ─────────────────────────────────────────────────────────────────────────
    # reference_pool: list[SentenceRecord] = list(state.get("all_accepted", []))
    # dedup_count = 0
    #
    # for sent in candidates:
    #     most_similar_score = 0.0
    #     most_similar_text = ""
    #     for ref in reference_pool:
    #         score = jaccard_bigram(sent["text"], ref["text"])
    #         if score > most_similar_score:
    #             most_similar_score = score
    #             most_similar_text = ref["text"]
    #
    #     if most_similar_score >= DEDUP_THRESHOLD:
    #         dedup_count += 1
    #         if sent.get("retry_count", 0) >= MAX_RETRY:
    #             new_discarded.append(sent)
    #         else:
    #             retry_sent = dict(sent)
    #             preview = most_similar_text[:100]
    #             retry_sent["prev_feedback"] = (
    #                 f"Near-duplicate (Jaccard={most_similar_score:.2f}). "
    #                 f"Too similar to: '{preview}'. "
    #                 f"Generate a structurally different sentence."
    #             )
    #             to_retry.append(retry_sent)
    #     else:
    #         new_accepted.append(sent)
    #         reference_pool.append(sent)
    # ─────────────────────────────────────────────────────────────────────────

    # Step 3 (ACTIVE): Per-Sentence Length Match (PSLM) filter
    # ─────────────────────────────────────────────────────────────────────────
    # Validates Generator's compliance dengan prompt directive (preserve seed
    # elaboration depth). Reject sentences yang drift terlalu jauh dari source
    # seed length.
    #
    # Threshold = 1.0 × seed_std per label (data-anchored, per-language adaptive)
    #
    # Empirical runtime correction:
    # - Initial attempt 0.5 × std → ~68% reject rate (over-aggressive)
    #   karena absolute threshold tidak scale dengan seed length — seed panjang
    #   (40+ words) dapat threshold relatif sempit, natural elaboration (+8-10w)
    #   ter-reject walau ini bukan drift
    # - Revised ke 1.0 × std → expected ~15-25% reject rate (balanced)
    # - Statistical interpretation: ±1 std = "within typical variance range"
    # - jav negative: threshold = 14.3 words
    # - jav neutral:  threshold = 8.7 words
    # - jav positive: threshold = 14.1 words
    # ─────────────────────────────────────────────────────────────────────────
    seed_lookup = {s["id"]: s for s in state.get("seeds", [])}
    seed_profile = state.get("seed_profile", {})
    length_stats = seed_profile.get("avg_length_per_label", {})
    target_label = state.get("target_label", "")
    pslm_threshold = 1.0 * float(length_stats.get(target_label, {}).get("std", 14.0))

    length_failed_count = 0
    for sent in candidates:
        seed = seed_lookup.get(sent.get("seed_id"))
        if seed is None:
            # Cannot validate without source seed → accept
            new_accepted.append(sent)
            continue

        seed_len = len(seed["text"].split())
        gen_len = len(sent["text"].split())
        deviation = abs(gen_len - seed_len)

        if deviation > pslm_threshold:
            length_failed_count += 1
            if sent.get("retry_count", 0) >= MAX_RETRY:
                new_discarded.append(sent)
            else:
                retry_sent = dict(sent)
                retry_sent["prev_feedback"] = (
                    f"Length drift: generated sentence ({gen_len} words) deviates "
                    f"too far from seed length ({seed_len} words). Generate a sentence "
                    f"with elaboration depth closer to the seed's character."
                )
                to_retry.append(retry_sent)
        else:
            new_accepted.append(sent)

    total_accepted = len(state.get("all_accepted", [])) + len(new_accepted)
    length_msg = f", {length_failed_count} length_failed" if length_failed_count else ""
    valfail_msg = f", {validation_failed_count} validation_failed" if validation_failed_count else ""
    print(f"Collect: +{len(new_accepted)} diterima (total {total_accepted}), {len(to_retry)} retry{length_msg}, +{len(new_discarded)} dibuang{valfail_msg}")

    return {
        "all_accepted": new_accepted,
        "all_discarded": new_discarded,
        "all_retried": list(to_retry),  # snapshot before retry (reducer appends)
        "to_retry": to_retry,
        "current_sentences": [],
    }
