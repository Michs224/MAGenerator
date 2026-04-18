"""Prompt builders and Pydantic output schemas for all 4 agents.

Prompting techniques used:
- Role-based system prompting (each agent has distinct role)
- Structured output via Pydantic schemas (ChatGoogleGenerativeAI.with_structured_output)
- Chain-of-Thought (validators reason step-by-step before verdict)
- Evidence-grounded reasoning (NusaBERT/GlotLID signals as context)
- Domain-adaptive (Contextualizer identifies domain from seed, not hardcoded)
"""

from __future__ import annotations

import json

from typing import Literal

from pydantic import BaseModel, Field

from NusaSynth.config import SENTENCES_PER_SEED


# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC OUTPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════


# ── Contextualizer ─────────────────────────────────────────────────────────

class VariationPlan(BaseModel):
    strategy: str = Field(description="What to change (aspect, scenario, entity, perspective, etc.)")
    preserve: str = Field(description="What must stay the same (complementing the strategy)")


class SeedAnalysis(BaseModel):
    seed_id: int = Field(description="The id of the seed being analyzed")
    domain: str = Field(description="Detected domain or topic of the seed text")
    style: str = Field(description="Style register and tone of the seed")
    sentiment_expression: str = Field(description="How sentiment is expressed in this seed")
    variations: list[VariationPlan] = Field(
        description=f"Exactly {SENTENCES_PER_SEED} variation plans for this seed",
        min_length=SENTENCES_PER_SEED,
        max_length=SENTENCES_PER_SEED,
    )


class ContextualizerOutput(BaseModel):
    seed_analyses: list[SeedAnalysis] = Field(description="Analysis and variation plans per seed")


# ── Generator ──────────────────────────────────────────────────────────────

class GeneratedSentence(BaseModel):
    seed_id: int = Field(description="The seed id this sentence belongs to (copied from input)")
    plan_id: int = Field(description="The plan id this sentence addresses (copied exactly from the input plan)")
    text: str = Field(description="The generated sentence")


class GeneratorOutput(BaseModel):
    sentences: list[GeneratedSentence] = Field(description="Generated sentences")


# ── Sentiment Validator ────────────────────────────────────────────────────

class SVEvaluation(BaseModel):
    idx: int = Field(description="Index of the sentence being evaluated (from input)")
    nusabert_assessment: str = Field(description="Interpretation of NusaBERT's prediction and how reliable the signal seems")
    semantic_analysis: str = Field(description="Analysis of what sentiment the text actually expresses")
    verdict: Literal["PASS", "REJECT"] = Field(description="PASS if sentiment matches target label, REJECT otherwise")
    reason: str | None = Field(default=None, description="Required if REJECT: brief explanation")


class SVOutput(BaseModel):
    evaluations: list[SVEvaluation]


# ── Linguistic Validator ───────────────────────────────────────────────────

class LVEvaluation(BaseModel):
    idx: int = Field(description="Index of the sentence being evaluated (from input)")
    naturalness: str = Field(description="Naturalness of the text for a native speaker of the target language")
    issues: str = Field(description="Linguistic issues observed, or 'none' if clean")
    verdict: Literal["PASS", "REJECT"] = Field(description="PASS if linguistically acceptable, REJECT otherwise")
    reason: str | None = Field(default=None, description="Required if REJECT: brief explanation")


class LVOutput(BaseModel):
    evaluations: list[LVEvaluation]


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════


def build_contextualizer_messages(
    seeds: list[dict],
    seed_profile: dict,
    lang_name: str,
) -> list[tuple[str, str]]:
    """Build (role, content) message list for Contextualizer.

    Returns list compatible with ChatGoogleGenerativeAI.invoke().
    """
    ld = seed_profile["label_distribution"]
    al = seed_profile["avg_length_per_label"]

    system = f"""You are a linguistic analyst for a multilingual sentiment data augmentation system.

Given {len(seeds)} seed sentences from a {lang_name} sentiment dataset, analyze each and produce {SENTENCES_PER_SEED} variation plans per seed ({len(seeds) * SENTENCES_PER_SEED} total).

Seed Profile ({lang_name}):
- Label distribution: negative={ld.get('negative', 0)}, neutral={ld.get('neutral', 0)}, positive={ld.get('positive', 0)}
- Average word count:
  - negative: avg {al['negative']['mean']} words (std {al['negative']['std']})
  - neutral:  avg {al['neutral']['mean']} words (std {al['neutral']['std']})
  - positive: avg {al['positive']['mean']} words (std {al['positive']['std']})

Dataset characteristics:
- The dataset primarily covers food and restaurant reviews, travel, telecom, shopping, services, and delivery, along with occasional everyday statements
- Natural code-mixing with Indonesian is common
- Casual and informal tone is common, though occasional formal phrasing may appear

For EACH seed, first analyze its domain, style, and sentiment expression. Then produce {SENTENCES_PER_SEED} variation plans.

Each plan must include:
- seed_id: the id of the seed this plan is based on
- What to change (strategy) and what to preserve

Rules:
- RESPECT the domain and style of each seed
- Prioritize variations that remain plausible for the same data source as the seed
- You may explore adjacent domains if naturally relevant to the seed's context

CRITICAL — APPROACH DIVERSITY:
The {SENTENCES_PER_SEED} variations for each seed must use DIFFERENT expressive approaches. Think of them as written by different people in different situations. Across the {SENTENCES_PER_SEED} plans, vary:
- Who is speaking and from what angle (first person experience, third person observation, advice to others, rhetorical question, comparison)
- What aspect is the focus (product, service, atmosphere, price-value ratio, specific incident, emotional reaction)
- How the sentiment is expressed (direct complaint, sarcasm, disappointment narrative, warning, praise story, factual observation)

Do NOT plan {SENTENCES_PER_SEED} sentences that all open the same way or follow the same sentence skeleton with swapped entities."""

    seeds_json = json.dumps(
        [{"id": s["id"], "text": s["text"], "label": s["label"]} for s in seeds],
        indent=2,
        ensure_ascii=False,
    )
    user = f"Analyze these {len(seeds)} seeds and produce variation plans.\n\nSeeds:\n{seeds_json}"

    return [("system", system), ("human", user)]


def build_generator_messages(
    seeds: list[dict],
    seed_profile: dict,
    variation_plans: list[dict],
    lang_name: str,
    target_label: str,
    retry_items: list[dict] | None = None,
) -> list[tuple[str, str]]:
    """Build messages for Generator (first pass or retry)."""
    al = seed_profile["avg_length_per_label"]
    seed_ref = "\n".join([f'Seed {s["id"]}: "{s["text"]}"' for s in seeds])

    if retry_items:
        n_sentences = len(retry_items)
    else:
        n_sentences = sum(len(p["variations"]) for p in variation_plans)

    system = f"""You are a native-level {lang_name} text generator.

Generate ONE sentence per variation instruction. Total: {n_sentences} sentences.

Seed Profile ({lang_name}) — use as length reference:
- negative: avg {al['negative']['mean']} words (std {al['negative']['std']})
- neutral:  avg {al['neutral']['mean']} words (std {al['neutral']['std']})
- positive: avg {al['positive']['mean']} words (std {al['positive']['std']})

Each variation plan includes context derived from its seed:
- plan_id: unique id of the plan (copy this exactly into your output)
- domain: topic or domain to stay within
- style: register and tone to match
- sentiment_expression: how sentiment was expressed in the seed
- strategy: what to change in the variation
- preserve: what must remain consistent

Rules:
- Express {target_label} sentiment in {lang_name}
- For every generated sentence, copy back the plan_id and seed_id exactly from the plan you are addressing
- Generate ONE sentence per plan_id — every input plan_id must appear exactly once in your output
- Honor each plan's domain, style, and sentiment_expression — these come from the seed and define the contextual envelope
- Do NOT paraphrase the seed — create genuinely different sentences
- Match the style and tone of the seeds — your output must sound like it comes from the same data source
- Code-mixing with Indonesian IS acceptable — the original dataset naturally contains this
- Sentence length should feel natural — use the seed profile above as a soft reference, not a hard limit
- CRITICAL: every sentence must be structurally distinct — do NOT repeat the same template with swapped words.

The {len(seeds)} seeds below are your style reference — study their vocabulary and tone:

{seed_ref}"""

    if retry_items:
        # Lookup per-seed context (domain/style/sentiment_expression) from variation_plans
        seed_context = {p["seed_id"]: p for p in variation_plans}
        retry_payload = []
        for r in retry_items:
            ctx = seed_context.get(r["seed_id"], {})
            retry_payload.append({
                "seed_id": r["seed_id"],
                "plan_id": r.get("plan_id", -1),
                "domain": ctx.get("domain", ""),
                "style": ctx.get("style", ""),
                "sentiment_expression": ctx.get("sentiment_expression", ""),
                "original_strategy": r.get("strategy", ""),
                "original_preserve": r.get("preserve", ""),
                "rejected_text": r["text"],
                "feedback": r["prev_feedback"],
            })
        retry_json = json.dumps(retry_payload, indent=2, ensure_ascii=False)
        user = (
            f"These sentences were previously REJECTED. "
            f"For each item, generate a new sentence and copy back the seed_id AND plan_id exactly. "
            f"Honor the seed's domain, style, and sentiment_expression, "
            f"keep the original strategy and preserve direction, "
            f"and address the feedback to fix what was wrong:\n\n{retry_json}"
        )
    else:
        plans_json = json.dumps(variation_plans, indent=2, ensure_ascii=False)
        user = f"Generate {target_label} {lang_name} sentences for these variation plans:\n\n{plans_json}"

    return [("system", system), ("human", user)]


def build_sv_messages(
    seed: dict,
    sentences_with_nusabert: list[dict],
    target_label: str,
) -> list[tuple[str, str]]:
    """Build messages for Sentiment Validator (one seed group).

    sentences_with_nusabert: [{"idx": 0, "text": "...", "nusabert": {"label": ..., "confidence": ...}}, ...]
    """
    system = f"""You are evaluating generated {target_label} sentences for sentiment correctness.

For each sentence, you receive:
- The generated text
- NusaBERT prediction and confidence (a programmatic classifier signal fine-tuned on the original training data)

Use Chain-of-Thought reasoning for each sentence:
1. nusabert_assessment: Interpret what NusaBERT predicts and judge how reliable its signal seems — the classifier may be misled by rare or unfamiliar words.
2. semantic_analysis: Analyze what sentiment the text actually expresses, independently from NusaBERT.
3. verdict: PASS or REJECT
4. reason: required if REJECT, brief explanation

Important:
- NusaBERT is a SIGNAL, not the final answer. It may be wrong on unfamiliar vocabulary.
- Do NOT reject because of code-mixing or unfamiliar words.
- Only reject if the MEANING clearly contradicts the target label "{target_label}"."""

    sentences_json = json.dumps(sentences_with_nusabert, indent=2, ensure_ascii=False)
    user = f"""Target label: {target_label}
Seed: "{seed['text']}"

Evaluate these {len(sentences_with_nusabert)} generated sentences:
{sentences_json}"""

    return [("system", system), ("human", user)]


def build_lv_messages(
    seed: dict,
    sentences_with_glotlid: list[dict],
    target_lang: str,
    lang_name: str,
) -> list[tuple[str, str]]:
    """Build messages for Linguistic Validator (one seed group).

    sentences_with_glotlid: [{"idx": 0, "text": "...", "glotlid": {"detected_lang": ..., "confidence": ...}}, ...]
    """
    system = f"""You are evaluating generated sentences for linguistic quality in {lang_name}.

For each sentence, you receive:
- The generated text
- GlotLID result: detected language and confidence

Use Chain-of-Thought reasoning for each sentence:
1. naturalness: Evaluate whether the text sounds like natural {lang_name}. Use GlotLID as a language-identity check (shows the detected language and how confidently). Also consider fluency, grammar, and idiomatic use.
2. issues: Any translationese, broken grammar, or incoherent meaning? Write 'none' if clean.
3. verdict: PASS or REJECT
4. reason: required if REJECT

Important:
- Natural code-mixing with Indonesian IS acceptable (the source dataset naturally contains code-mixed text)
- Only reject if: detected as completely wrong language, obviously machine-translated, grammatically broken, or incoherent"""

    sentences_json = json.dumps(sentences_with_glotlid, indent=2, ensure_ascii=False)
    user = f"""Target language: {lang_name} ({target_lang})
Seed context: "{seed['text']}"

Evaluate these sentences:
{sentences_json}"""

    return [("system", system), ("human", user)]
