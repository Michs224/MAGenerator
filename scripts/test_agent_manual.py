"""
Manual test: mimic Contextualizer → Generator → Validator flow.
Panggil LLM API langsung untuk 1 batch (5 seeds negative, Javanese).

Prompting techniques used:
- Role-based system prompting (each agent has distinct role)
- Structured output (JSON format)
- Chain-of-Thought (validators reason step-by-step before verdict)
- Evidence-grounded reasoning (NusaBERT/GlotLID signals as context)
- Domain-adaptive (Contextualizer identifies domain from seed, not hardcoded)

Usage:
  uv run python scripts/test_agent_manual.py
"""

import json

# ============================================================
# DATA: 5 negative seeds dari NusaX-Senti Javanese (train set)
# ============================================================

SEEDS = [
    {
        "id": 449,
        "text": "Panganane lumayan, nanging ana pelayan sing lumayan kemproh war dadi kurang nyaman. Kanggo panganan rada cepet yo ben ora kangelihen konsumene. Isih akeh sing kudu ditingkatake.",
        "label": "negative",
        "lang": "jav",
    },
    {
        "id": 869,
        "text": "Isuk wis nesu-nesu merga pelayanan ning bank cabang sanur sing ora profesional lan nggelakne! Ora salah akeh penilaian pelanggan elek.",
        "label": "negative",
        "lang": "jav",
    },
    {
        "id": 351,
        "text": "Penasaran pinggin dijajal pithik sik cukup terkenal kuwi. Awan kuwi sepi, pesen pithik bakar. Jebule pithik bakare pedes. Bocah-bocah ora isa mangan. Kudune pelayan nakokake arep pedes utawa ora amarga ana 2 pilian. Banjur aku njaluk 2 pithik bakar sing ora pedes. Ampun! Pithike mung kaya digoreng banjur dilumuri kecap. Dhewe uwis ilang napsu mangan.",
        "label": "negative",
        "lang": "jav",
    },
    {
        "id": 848,
        "text": "Kaping pisan menyang panggon iki, reservasi lan entuk nomer 3. Njaluk panggon ning lantai ngisor dadi ora perlu munggah ondak-ondakan merga ana sing dengkule lagi lara. Ngenteni meh 40 menit ora ana kepastian. Tak batalne. Maune ora gelem ngeki rating 1 amarga pancet mungkin rame lan panjalukanku khusus, dadi angel. Eh mbaks malah ketus.",
        "label": "negative",
        "lang": "jav",
    },
    {
        "id": 901,
        "text": "Kuota dadi entek resik kanggo ndelok foto-foto sing mung gawe aku srei, panganan enak-enak sing marai ngiler",
        "label": "negative",
        "lang": "jav",
    },
]

SEED_PROFILE = {
    "lang": "jav",
    "label_distribution": {"negative": 192, "neutral": 119, "positive": 189},
    "avg_length_per_label": {
        "negative": {"mean": 21.8, "std": 14.2},
        "neutral": {"mean": 13.6, "std": 8.5},
        "positive": {"mean": 30.3, "std": 14.1},
    },
    "unit": "words",
}


# ============================================================
# STEP 1: CONTEXTUALIZER
# Technique: Role-based + Structured output + Domain-adaptive
# ============================================================

CONTEXTUALIZER_SYSTEM = """You are a linguistic analyst for a multilingual sentiment data augmentation system.

Given 5 seed sentences from a Javanese sentiment dataset, analyze each and produce 5 variation plans per seed (25 total, numbered 1-25).

Seed Profile (Javanese):
- Label distribution: negative=192, neutral=119, positive=189
- Average word count: negative=21.8 (std 14.2), neutral=13.6 (std 8.5), positive=30.3 (std 14.1)

Important dataset characteristics:
- This data comes from REAL internet text (online reviews, social media complaints, news)
- Natural code-mixing with Indonesian is common (~48% of sentences contain Indonesian words)
- Register is predominantly casual/ngoko (informal Javanese)

For EACH seed, first analyze its domain, register, and sentiment expression. Then produce 5 variation plans.

Each plan must include:
- seed_id: the id of the seed this plan is based on
- The original seed text (so the Generator has full context)
- What to change and what to preserve

Rules:
- RESPECT the domain and register of each seed
- Prioritize variations that remain plausible for the same data source as the seed
- You may explore adjacent domains if naturally relevant to the seed's context
- The balance between same-domain and adjacent-domain variations is your judgment per seed
- Respond in valid JSON"""

CONTEXTUALIZER_USER = """Analyze these 5 seeds and produce 25 variation plans.

Seeds:
""" + json.dumps(
    [{"id": s["id"], "text": s["text"], "label": s["label"]} for s in SEEDS],
    indent=2,
    ensure_ascii=False,
)


# ============================================================
# STEP 2: GENERATOR
# Technique: Role-based + Constraint-based + Structured output
# ============================================================

GENERATOR_SYSTEM = """You are a native-level Javanese text generator.

Generate ONE sentence per variation instruction. Total: 25 sentences.

Seed Profile (Javanese) — use as length reference:
- negative: avg 21.8 words (std 14.2)
- neutral:  avg 13.6 words (std 8.5)
- positive: avg 30.3 words (std 14.1)

Rules:
- Express negative sentiment in Javanese
- Do NOT paraphrase the seed — create genuinely different sentences
- MATCH the register and style of each seed — your output must sound like it comes from the same source
- Code-mixing with Indonesian IS acceptable — the original dataset naturally contains this
- Sentence length should feel natural — use the seed profile above as a soft reference, not a hard limit
- Output valid JSON array: [{"seed_id": int, "text": "..."}, ...]
  seed_id = the id of the seed this variation is based on (from the variation plan)

The 5 seeds below are your style reference — study their register, vocabulary, and tone:

""" + "\n".join(
    [f'Seed {s["id"]}: "{s["text"]}"' for s in SEEDS]
)

GENERATOR_USER_TEMPLATE = """Generate 25 negative Javanese sentences based on these variation plans from the Contextualizer:

{contextualizer_output}

Output ONLY the JSON array, no explanation."""


# ============================================================
# STEP 3: SENTIMENT VALIDATOR (per seed group, 5 sentences)
# Technique: Evidence-grounded + Chain-of-Thought
# ============================================================

SV_SYSTEM = """You are evaluating generated Javanese sentences for sentiment correctness.

For each sentence, you receive:
- The generated text
- NusaBERT prediction and confidence (a programmatic classifier signal trained on 500 seed sentences)

Use Chain-of-Thought reasoning for each sentence:
1. nusabert_signal: What does NusaBERT predict? Is it confident or uncertain?
2. semantic_analysis: What sentiment does the text actually express? Which words/phrases indicate this?
3. agreement: Do NusaBERT and your semantic analysis agree or conflict?
4. verdict: PASS or REJECT
5. reason: required if REJECT, brief explanation

Important:
- NusaBERT is a SIGNAL, not the final answer. It was trained on only 500 sentences and may be wrong on unfamiliar vocabulary.
- Do NOT reject because of code-mixing or unfamiliar words.
- Only reject if the MEANING clearly contradicts the target label.
- Output valid JSON."""

SV_USER_TEMPLATE = """Target label: negative
Seed: "{seed_text}"

Evaluate these 5 generated sentences:
{sentences_with_nusabert_json}"""

# Example NusaBERT results (simulated — replace with real inference):
SV_USER_EXAMPLE = """Target label: negative
Seed: "Panganane lumayan, nanging ana pelayan sing lumayan kemproh war dadi kurang nyaman."

Evaluate these 5 generated sentences:
[
  {"text": "Tempate reget banget, mejane lengket kabeh, males balik maneh.", "nusabert": {"label": "negative", "confidence": 0.92}},
  {"text": "Pelayanane suwi tenan, nunggu sejam luwih mung kanggo pesen minum.", "nusabert": {"label": "negative", "confidence": 0.87}},
  {"text": "Parkire angel banget, mubeng-mubeng ora oleh panggonan.", "nusabert": {"label": "negative", "confidence": 0.78}},
  {"text": "Pesanane teka salah, wis komplain malah ora digubris.", "nusabert": {"label": "negative", "confidence": 0.85}},
  {"text": "Regane larang nanging rasane biasa wae, ora worth it.", "nusabert": {"label": "neutral", "confidence": 0.55}}
]"""


# ============================================================
# STEP 4: LINGUISTIC VALIDATOR (per seed group)
# Technique: Evidence-grounded + Chain-of-Thought
# ============================================================

LV_SYSTEM = """You are evaluating generated sentences for linguistic quality in Javanese.

For each sentence, you receive:
- The generated text
- GlotLID result: detected language and confidence

Use Chain-of-Thought reasoning for each sentence:
1. glotlid_signal: What language was detected? Is confidence high or low?
2. naturalness: Does this sound like something a Javanese speaker would actually write online?
3. issues: Any translationese, broken grammar, or incoherent meaning?
4. verdict: PASS or REJECT
5. reason: required if REJECT

Important:
- Natural code-mixing with Indonesian IS acceptable (the source dataset contains ~48% code-mixed sentences)
- GlotLID baseline on original Javanese data: 93.4% accuracy — it tolerates natural code-mixing
- Only reject if: detected as completely wrong language, obviously machine-translated, grammatically broken, or incoherent
- Output valid JSON."""

LV_USER_TEMPLATE = """Target language: Javanese (jav)
Seed context: "{seed_text}"

Evaluate these sentences:
{sentences_with_glotlid_json}"""


# ============================================================
# PRINT ALL PROMPTS
# ============================================================

if __name__ == "__main__":
    separator = "=" * 70

    print(separator)
    print("STEP 1: CONTEXTUALIZER")
    print(separator)
    print("\n[SYSTEM]")
    print(CONTEXTUALIZER_SYSTEM)
    print("\n[USER]")
    print(CONTEXTUALIZER_USER)

    print(f"\n\n{separator}")
    print("STEP 2: GENERATOR")
    print(f"{separator}")
    print("\n[SYSTEM]")
    print(GENERATOR_SYSTEM)
    print("\n[USER TEMPLATE] (replace {contextualizer_output} with Step 1 output)")
    print(GENERATOR_USER_TEMPLATE)

    print(f"\n\n{separator}")
    print("STEP 3: SENTIMENT VALIDATOR (per seed group)")
    print(f"{separator}")
    print("\n[SYSTEM]")
    print(SV_SYSTEM)
    print("\n[USER EXAMPLE]")
    print(SV_USER_EXAMPLE)

    print(f"\n\n{separator}")
    print("STEP 4: LINGUISTIC VALIDATOR (per seed group)")
    print(f"{separator}")
    print("\n[SYSTEM]")
    print(LV_SYSTEM)
    print("\n[USER TEMPLATE]")
    print(LV_USER_TEMPLATE)

    print(f"\n\n{separator}")
    print("EXAMPLE: SV Chain-of-Thought Expected Output")
    print(f"{separator}")
    print(json.dumps({
        "seed_id": 449,
        "evaluations": [
            {
                "text": "Tempate reget banget, mejane lengket kabeh, males balik maneh.",
                "nusabert_signal": "negative with 0.92 confidence — high, agrees with target",
                "semantic_analysis": "'reget' (dirty), 'lengket' (sticky), 'males balik' (don't want to return) — clearly negative",
                "agreement": "NusaBERT and semantic analysis agree",
                "verdict": "PASS",
                "reason": None,
            },
            {
                "text": "Regane larang nanging rasane biasa wae, ora worth it.",
                "nusabert_signal": "neutral with 0.55 confidence — low and disagrees with target",
                "semantic_analysis": "'larang' (expensive), 'biasa wae' (mediocre), 'ora worth it' — expresses disappointment, which is negative",
                "agreement": "Conflict — NusaBERT says neutral but meaning is negative. NusaBERT likely confused by 'biasa wae'",
                "verdict": "PASS",
                "reason": None,
            },
        ],
    }, indent=2, ensure_ascii=False))

    print(f"\n\n{separator}")
    print("API CALL EXAMPLE (Google Gemini)")
    print(f"{separator}")
    print("""
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

# Step 1: Contextualizer
ctx_resp = model.generate_content(
    CONTEXTUALIZER_SYSTEM + "\\n\\n" + CONTEXTUALIZER_USER
)
ctx_output = ctx_resp.text
print("Contextualizer output:", ctx_output[:200], "...")

# Step 2: Generator
gen_resp = model.generate_content(
    GENERATOR_SYSTEM + "\\n\\n" +
    GENERATOR_USER_TEMPLATE.format(contextualizer_output=ctx_output)
)
generated = json.loads(gen_resp.text)
print(f"Generated {len(generated)} sentences")

# Step 3: SV (per seed — example for seed 449)
# First run NusaBERT inference on generated sentences (programmatic)
# Then call LLM with results
seed_449_sentences = [g for g in generated if g["seed_id"] == 449]
# ... add nusabert results ...
# sv_resp = model.generate_content(SV_SYSTEM + "\\n\\n" + sv_user_prompt)

# Step 4: LV (per seed — only for SV-passed sentences)
# First run GlotLID inference (programmatic)
# Then call LLM with results
# lv_resp = model.generate_content(LV_SYSTEM + "\\n\\n" + lv_user_prompt)
""")
