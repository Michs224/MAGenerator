"""State definitions for NusaSynth LangGraph pipeline.

Three TypedDicts:
- BatchState   : parent graph state (full pipeline)
- SVState      : SV subgraph state (shared + private keys)
- LVState      : LV subgraph state (shared + private keys)
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


# ── Sentence record ────────────────────────────────────────────────────────

class SentenceRecord(TypedDict, total=False):
    """One generated sentence with full lifecycle tracking.

    Fields are progressively filled: Generator sets core fields,
    SV adds sentiment results, LV adds linguistic results.
    """

    # Core (set by Generator)
    sid: int            # sequential ID, unique within pipeline run
    seed_id: int        # which seed this was generated from
    plan_id: int        # which variation plan (LLM copy-back, verifiable anchor)
    strategy: str       # plan strategy (for retry context)
    preserve: str       # plan preserve (for retry context)
    domain: str         # seed-level domain (from Contextualizer)
    style: str          # seed-level style (from Contextualizer)
    sentiment_expression: str  # seed-level sentiment expression (from Contextualizer)
    text: str
    target_label: str
    target_lang: str
    retry_count: int    # 0 = first attempt

    # Feedback from last rejection (set by collect, used by Generator retry)
    prev_feedback: str | None

    # Sentiment Validator results
    nusabert_label: str | None
    nusabert_conf: float | None
    sv_verdict: str | None       # "PASS" / "REJECT"
    sv_cot: str | None           # chain-of-thought summary
    sv_feedback: str | None      # rejection reason

    # Linguistic Validator results
    glotlid_lang: str | None
    glotlid_conf: float | None
    lv_verdict: str | None       # "PASS" / "REJECT"
    lv_cot: str | None
    lv_feedback: str | None


def make_sentence(
    sid: int,
    seed_id: int,
    text: str,
    target_label: str,
    target_lang: str,
    plan_id: int = -1,
    strategy: str = "",
    preserve: str = "",
    domain: str = "",
    style: str = "",
    sentiment_expression: str = "",
    retry_count: int = 0,
    prev_feedback: str | None = None,
) -> SentenceRecord:
    """Create a SentenceRecord with all optional fields set to None."""
    return SentenceRecord(
        sid=sid,
        seed_id=seed_id,
        plan_id=plan_id,
        strategy=strategy,
        preserve=preserve,
        domain=domain,
        style=style,
        sentiment_expression=sentiment_expression,
        text=text,
        target_label=target_label,
        target_lang=target_lang,
        retry_count=retry_count,
        prev_feedback=prev_feedback,
        nusabert_label=None,
        nusabert_conf=None,
        sv_verdict=None,
        sv_cot=None,
        sv_feedback=None,
        glotlid_lang=None,
        glotlid_conf=None,
        lv_verdict=None,
        lv_cot=None,
        lv_feedback=None,
    )


# ── Parent graph state ─────────────────────────────────────────────────────

class BatchState(TypedDict, total=False):
    """LangGraph parent state for one 5-seed batch."""

    # Fixed inputs (set by outer loop before pipeline.invoke())
    seeds: list[dict]          # [{id, text, label, lang}] — 5 seeds
    seed_profile: dict         # precomputed language stats
    target_label: str          # "negative" / "neutral" / "positive"
    target_lang: str           # ISO code, e.g. "jav"
    lang_name: str             # human name, e.g. "Javanese"

    # Contextualizer output (set once, unchanged on retry)
    variation_plans: list[dict]

    # Working set (overwritten each generation pass)
    current_sentences: list[SentenceRecord]

    # Internal counter for assigning unique sids
    next_sid: int

    # Accumulators (persist across retries)
    all_accepted: Annotated[list[SentenceRecord], operator.add]
    all_discarded: Annotated[list[SentenceRecord], operator.add]
    all_retried: Annotated[list[SentenceRecord], operator.add]  # intermediate rejection snapshots

    # Retry queue — empty = pipeline ends
    to_retry: list[SentenceRecord]


# ── SV subgraph state ──────────────────────────────────────────────────────

class SVState(TypedDict, total=False):
    """Sentiment Validator subgraph state.

    Shared keys with BatchState: seeds, current_sentences, target_label,
    target_lang, lang_name. No private keys — parallel processing is done
    within a single node using ThreadPoolExecutor.
    """

    seeds: list[dict]
    current_sentences: list[SentenceRecord]
    target_label: str
    target_lang: str
    lang_name: str


# ── LV subgraph state ──────────────────────────────────────────────────────

class LVState(TypedDict, total=False):
    """Linguistic Validator subgraph state.

    SV-passed sentences are validated; SV-rejected are passed through.
    Parallel processing done within a single node using ThreadPoolExecutor.
    """

    seeds: list[dict]
    current_sentences: list[SentenceRecord]
    target_label: str
    target_lang: str
    lang_name: str
