"""Sentiment Validator subgraph.

Processes all seed groups in parallel:
  1. Batch NusaBERT inference on all sentences (single GPU pass).
  2. Group by seed_id.
  3. Parallel LLM calls per group (ThreadPoolExecutor — I/O-bound).
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from NusaSynth.config import GEMINI_MODEL
from NusaSynth.prompts import SVOutput, build_sv_messages
from NusaSynth.state import SVState, SentenceRecord
from NusaSynth.tools import classify_sentiment_batch


# ── Subgraph node ──────────────────────────────────────────────────────────


def sv_run(state: SVState) -> dict:
    """Run SV on all current_sentences: batch NusaBERT + parallel LLM calls."""
    sentences = list(state["current_sentences"])

    # Step 1: Batch NusaBERT inference (single GPU pass for all sentences)
    texts = [s["text"] for s in sentences]
    nb_results = classify_sentiment_batch(texts, state["target_lang"])
    for sent, nb in zip(sentences, nb_results):
        sent["nusabert_label"] = nb["label"]
        sent["nusabert_conf"] = nb["confidence"]

    # Step 2: Group by seed_id
    groups_map: dict[int, list[SentenceRecord]] = defaultdict(list)
    for sent in sentences:
        groups_map[sent["seed_id"]].append(sent)
    groups = list(groups_map.values())

    # Step 3: Parallel LLM calls per group
    def process_group(group: list[SentenceRecord]) -> list[SentenceRecord]:
        seed_id = group[0]["seed_id"]
        seed = next(s for s in state["seeds"] if s["id"] == seed_id)

        sv_input = [
            {
                "idx": i,
                "text": sent["text"],
                "nusabert": {
                    "label": sent["nusabert_label"],
                    "confidence": sent["nusabert_conf"],
                },
            }
            for i, sent in enumerate(group)
        ]

        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
        structured_llm = llm.with_structured_output(SVOutput)
        messages = build_sv_messages(
            seed=seed,
            sentences_with_nusabert=sv_input,
            target_label=state["target_label"],
        )
        expected_idx_set = set(range(len(group)))

        def is_valid(result: SVOutput) -> bool:
            returned = [e.idx for e in result.evaluations]
            return len(returned) == len(group) and set(returned) == expected_idx_set

        result: SVOutput = structured_llm.invoke(messages)
        if not is_valid(result):
            returned = [e.idx for e in result.evaluations]
            print(f"SV WARNING seed_id={seed_id}: bijection invalid (dapat {returned}). Coba lagi sekali.")
            result = structured_llm.invoke(messages)
            if not is_valid(result):
                returned = [e.idx for e in result.evaluations]
                print(f"SV FAILED seed_id={seed_id}: bijection masih invalid (dapat {returned}). Group ditandai VALIDATION_FAILED.")
                for sent in group:
                    sent["sv_verdict"] = "VALIDATION_FAILED"
                    sent["sv_feedback"] = "sv_bijection_failure"
                return group

        for evaluation in result.evaluations:
            sent = group[evaluation.idx]
            verdict = evaluation.verdict.upper()
            sent["sv_verdict"] = verdict
            sent["sv_cot"] = (
                f"NusaBERT: {evaluation.nusabert_assessment} | "
                f"Semantic: {evaluation.semantic_analysis}"
            )
            sent["sv_feedback"] = evaluation.reason if verdict == "REJECT" else None
        return group

    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = {executor.submit(process_group, g): g for g in groups}
        processed_groups = [f.result() for f in as_completed(futures)]

    all_processed = [sent for group in processed_groups for sent in group]
    total_passed = sum(1 for s in all_processed if s.get("sv_verdict") == "PASS")
    print(f"SV: {total_passed}/{len(all_processed)} PASS")

    return {"current_sentences": all_processed}


# ── Build subgraph ─────────────────────────────────────────────────────────


def build_sv_subgraph() -> StateGraph:
    """Build and compile the SV subgraph."""
    builder = StateGraph(SVState)
    builder.add_node("sv_run", sv_run)
    builder.add_edge(START, "sv_run")
    builder.add_edge("sv_run", END)
    return builder.compile()
