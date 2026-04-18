"""Linguistic Validator subgraph.

Processes SV-passed sentences in parallel; SV-rejected pass through unchanged.
  1. Split: SV-passed vs SV-rejected.
  2. Batch GlotLID inference on SV-passed (fast, sequential).
  3. Group SV-passed by seed_id.
  4. Parallel LLM naturalness checks per group (ThreadPoolExecutor).
  5. Merge LV-processed + SV-rejected → current_sentences.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from NusaSynth.config import GEMINI_MODEL
from NusaSynth.prompts import LVOutput, build_lv_messages
from NusaSynth.state import LVState, SentenceRecord
from NusaSynth.tools import identify_language


# ── Subgraph node ──────────────────────────────────────────────────────────


def lv_run(state: LVState) -> dict:
    """Run LV on SV-passed sentences: batch GlotLID + parallel LLM calls."""
    sv_passed = [s for s in state["current_sentences"] if s.get("sv_verdict") == "PASS"]
    sv_rejected = [s for s in state["current_sentences"] if s.get("sv_verdict") != "PASS"]

    if not sv_passed:
        return {"current_sentences": sv_rejected}

    # Step 1: GlotLID inference on all SV-passed sentences (fast, sequential)
    for sent in sv_passed:
        gl = identify_language(sent["text"])
        sent["glotlid_lang"] = gl["detected_lang"]
        sent["glotlid_conf"] = gl["confidence"]

    # Step 2: Group by seed_id
    groups_map: dict[int, list[SentenceRecord]] = defaultdict(list)
    for sent in sv_passed:
        groups_map[sent["seed_id"]].append(sent)
    groups = list(groups_map.values())

    # Step 3: Parallel LLM calls per group
    def process_group(group: list[SentenceRecord]) -> list[SentenceRecord]:
        seed_id = group[0]["seed_id"]
        seed = next(s for s in state["seeds"] if s["id"] == seed_id)

        lv_input = [
            {
                "idx": i,
                "text": sent["text"],
                "glotlid": {
                    "detected_lang": sent["glotlid_lang"],
                    "confidence": sent["glotlid_conf"],
                },
            }
            for i, sent in enumerate(group)
        ]

        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
        structured_llm = llm.with_structured_output(LVOutput)
        messages = build_lv_messages(
            seed=seed,
            sentences_with_glotlid=lv_input,
            target_lang=state["target_lang"],
            lang_name=state["lang_name"],
        )
        expected_idx_set = set(range(len(group)))

        def is_valid(result: LVOutput) -> bool:
            returned = [e.idx for e in result.evaluations]
            return len(returned) == len(group) and set(returned) == expected_idx_set

        result: LVOutput = structured_llm.invoke(messages)
        if not is_valid(result):
            returned = [e.idx for e in result.evaluations]
            print(f"LV WARNING seed_id={seed_id}: bijection invalid (dapat {returned}). Coba lagi sekali.")
            result = structured_llm.invoke(messages)
            if not is_valid(result):
                returned = [e.idx for e in result.evaluations]
                print(f"LV FAILED seed_id={seed_id}: bijection masih invalid (dapat {returned}). Group ditandai VALIDATION_FAILED.")
                for sent in group:
                    sent["lv_verdict"] = "VALIDATION_FAILED"
                    sent["lv_feedback"] = "lv_bijection_failure"
                return group

        for evaluation in result.evaluations:
            sent = group[evaluation.idx]
            verdict = evaluation.verdict.upper()
            sent["lv_verdict"] = verdict
            sent["lv_cot"] = (
                f"Naturalness: {evaluation.naturalness} | "
                f"Issues: {evaluation.issues}"
            )
            sent["lv_feedback"] = evaluation.reason if verdict == "REJECT" else None
        return group

    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = {executor.submit(process_group, g): g for g in groups}
        processed_groups = [f.result() for f in as_completed(futures)]

    lv_processed = [sent for group in processed_groups for sent in group]
    total_passed = sum(1 for s in lv_processed if s.get("lv_verdict") == "PASS")
    print(f"LV: {total_passed}/{len(lv_processed)} PASS")

    return {"current_sentences": lv_processed + sv_rejected}


# ── Build subgraph ─────────────────────────────────────────────────────────


def build_lv_subgraph() -> StateGraph:
    """Build and compile the LV subgraph."""
    builder = StateGraph(LVState)
    builder.add_node("lv_run", lv_run)
    builder.add_edge(START, "lv_run")
    builder.add_edge("lv_run", END)
    return builder.compile()
