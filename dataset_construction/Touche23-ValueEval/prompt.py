"""
Touche23-ValueEval prompt templates.

Definitions are built by combining:
  - VALUEBENCH_DEFINITIONS (parent-level philosophical concept, 13 clusters)
  - value-categories.json  (fine-grained sub-values + example effects, 20 labels)

For "Universalism: objectivity" there is no VALUEBENCH parent description, so
the function falls back to the JSON content only.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Load value-categories.json once at import time
# ---------------------------------------------------------------------------
_JSON_PATH = Path(__file__).resolve().parent / "data" / "value-categories.json"
with open(_JSON_PATH, encoding="utf-8") as _f:
    VALUE_CATEGORIES: dict = json.load(_f)


# ---------------------------------------------------------------------------
# Parent-level philosophical definitions (from ValueBench)
# Keys correspond to the "parent" portion of Touche labels.
# ---------------------------------------------------------------------------
VALUEBENCH_DEFINITIONS: dict[str, str] = {
    "Self-Direction": (
        "The pursuit of independence and self-expression. "
        "Refined into Action (autonomy of behavior) and Thought (autonomy of mind)."
    ),
    "Stimulation": "The seeking of excitement, novelty, and challenge in life.",
    "Hedonism": "The pursuit of pleasure and sensuous gratification for oneself.",
    "Achievement": "Personal success through demonstrating competence according to social standards.",
    "Power": (
        "Social status and prestige, control over people and resources. "
        "Refined into Dominance (control over people) and Resources (control over material goods)."
    ),
    "Face": (
        "The desire to maintain a positive public image and be perceived as "
        "successful, competent, and admired by others."
    ),
    "Security": "The pursuit of safety, harmony, and stability of society, relationships, and self.",
    "Tradition": (
        "Respect, commitment, and acceptance of the customs and ideas that one's "
        "culture or religion provide."
    ),
    "Conformity": (
        "Restraint of actions and impulses likely to harm others or violate social expectations. "
        "Refined into Rules (compliance with formal obligations) and "
        "Interpersonal (avoidance of upsetting others)."
    ),
    "Humility": (
        "Recognising one's insignificance in the larger scheme of things; "
        "not thinking of oneself as more important than others."
    ),
    "Benevolence": (
        "Preserving and enhancing the welfare of people with whom one is in frequent "
        "personal contact. Refined into Caring (devotion to in-group welfare) and "
        "Dependability (being a reliable in-group member)."
    ),
    "Universalism": (
        "Understanding, appreciation, tolerance, and protection for the welfare of all "
        "people and for nature. Refined into Concern (equality and justice), "
        "Nature (preservation of environment), and Tolerance (acceptance of diversity)."
    ),
}

# Maps each Touche fine-grained label to the VALUEBENCH_DEFINITIONS key.
# Labels with no VALUEBENCH parent are absent from this dict (→ JSON-only fallback).
_TOUCHE_TO_PARENT: dict[str, str] = {
    "Self-direction: thought":      "Self-Direction",
    "Self-direction: action":       "Self-Direction",
    "Stimulation":                  "Stimulation",
    "Hedonism":                     "Hedonism",
    "Achievement":                  "Achievement",
    "Power: dominance":             "Power",
    "Power: resources":             "Power",
    "Face":                         "Face",
    "Security: personal":           "Security",
    "Security: societal":           "Security",
    "Tradition":                    "Tradition",
    "Conformity: rules":            "Conformity",
    "Conformity: interpersonal":    "Conformity",
    "Humility":                     "Humility",
    "Benevolence: caring":          "Benevolence",
    "Benevolence: dependability":   "Benevolence",
    "Universalism: concern":        "Universalism",
    "Universalism: nature":         "Universalism",
    "Universalism: tolerance":      "Universalism",
    # "Universalism: objectivity" intentionally omitted → JSON-only fallback
}


def get_definition(value: str) -> str:
    """
    Build a rich definition block for a Touche value label.

    Combines:
      1. The VALUEBENCH parent-level concept sentence (when available).
      2. The fine-grained sub-values and their example effects from value-categories.json.

    Example output for "Security: personal":

        Value: Security: personal
        Concept (Security): The pursuit of safety, harmony, and stability ...
        This sub-category focuses on:
          • Have good health: avoiding diseases, preserving health, ...
          • Have a comfortable life: providing subsistence income, ...
    """
    lines: list[str] = [f"Value: {value}"]

    # 1. Parent-level concept (VALUEBENCH)
    parent_key = _TOUCHE_TO_PARENT.get(value)
    if parent_key and parent_key in VALUEBENCH_DEFINITIONS:
        parent_label = value.split(":")[0].strip() if ":" in value else value
        lines.append(f"Concept ({parent_label}): {VALUEBENCH_DEFINITIONS[parent_key]}")

    # 2. Fine-grained sub-values from JSON
    sub_values: dict = VALUE_CATEGORIES.get(value, {})
    if sub_values:
        lines.append("This sub-category focuses on:")
        for sub_value, effects in sub_values.items():
            effects_str = "; ".join(effects[:3])  # cap at 3 effects per sub-value
            lines.append(f"  \u2022 {sub_value}: {effects_str}")

    return "\n".join(lines)


def get_concept_line(value: str) -> str:
    """
    Return just the parent-level concept sentence for a value (used by the validator).
    Falls back to sub-value names if no parent definition exists.
    """
    parent_key = _TOUCHE_TO_PARENT.get(value)
    if parent_key and parent_key in VALUEBENCH_DEFINITIONS:
        return VALUEBENCH_DEFINITIONS[parent_key]
    sub_values: dict = VALUE_CATEGORIES.get(value, {})
    if sub_values:
        return "; ".join(list(sub_values.keys())[:3])
    return value


# ---------------------------------------------------------------------------
# Per-value-family warnings (Fix 3)
#
# Injected into the user prompt only for Benevolence and Security values,
# where systematic welfare/safety frame contamination is most common.
# Empty string for all other values → zero overhead on 16/20 value types.
# ---------------------------------------------------------------------------
_VALUE_FAMILY_WARNINGS: dict[str, str] = {
    "Benevolence: caring": (
        "Welfare-frame caution: the positive argument is about caring for people's welfare. "
        "Your negative must use a fundamentally different concern — cost, fiscal burden, "
        "institutional capacity, or evidence quality. Do NOT argue about welfare, harm to "
        "people, or protection of vulnerable groups from another angle."
    ),
    "Benevolence: dependability": (
        "Welfare-frame caution: the positive argument is about reliability and support for "
        "others. Your negative must use a fundamentally different concern — cost, feasibility, "
        "or evidence. Do NOT argue about dependability, reliability, or letting people down."
    ),
    "Security: personal": (
        "Safety-frame caution: the positive argument is about personal safety, health, or "
        "financial stability. Your negative must use a fundamentally different concern. "
        "Do NOT argue about safety, health risks, or financial security from another angle."
    ),
    "Security: societal": (
        "Safety-frame caution: the positive argument is about social stability or public "
        "safety. Your negative must use a fundamentally different concern. Do NOT argue about "
        "instability, civil unrest, or public safety risks from another angle."
    ),
}


def get_value_family_warning(value: str) -> str:
    """
    Return a targeted contamination warning for Benevolence and Security values.
    Returns an empty string for all other values.
    """
    return _VALUE_FAMILY_WARNINGS.get(value, "")


# ---------------------------------------------------------------------------
# Few-shot examples
#
# DESIGN PRINCIPLES:
#   1. Each example is tagged with its assigned strategy ("Preferred strategy: X").
#   2. The thinking block explicitly names the strategy, applies it step by step,
#      and confirms no value label appears in the output.
#   3. Six examples cover all six strategies, including the "contradict" approach
#      that directly challenges the target value's validity in context.
#   4. Word counts are tracked to teach length matching.
# ---------------------------------------------------------------------------
TOUCHE_EXAMPLES = """
Example 1 — strategy: side-effects:
Question: Should we ban factory farming?
Value: Security: personal
Concept (Security): The pursuit of safety, harmony, and stability of society, relationships, and self.
This sub-category focuses on:
  • Have good health: avoiding diseases, preserving health
  • Have a comfortable life: providing subsistence income, resulting in having no financial worries
Positive Answer (invokes the value, 22 words): Factory farming allows for the production of cheap food, which is a necessity for families surviving on a low income.
Preferred strategy: side-effects: argue that the policy would produce serious unintended consequences in a domain unrelated to the target value
Thinking:
  - The positive invokes Security: personal — cheap food provides economic comfort for low-income households.
  - Strategy: side-effects. What serious harm does factory farming cause in a completely different domain?
  - Factory farming is a major driver of greenhouse gas emissions and aquifer depletion — concrete environmental consequences unrelated to household finances.
  - No value label names in output. Target length: ~22 words.
Negative Answer (25 words): Factory farming is one of the largest contributors to greenhouse gas emissions and water depletion, making it environmentally unsustainable as a responsible long-term food strategy.

Example 2 — strategy: institutional:
Question: Should we legalize organ trade?
Value: Universalism: concern
Concept (Universalism): Understanding, appreciation, tolerance, and protection for the welfare of all people.
This sub-category focuses on:
  • Have equality: providing all people with equal opportunities in life
  • Be just: protecting the weak and vulnerable in society
Positive Answer (invokes the value, 20 words): Legalizing organ trade would save thousands of lives by increasing the supply of available organs for those in need.
Preferred strategy: institutional: argue that existing rules, professional norms, or institutional structures already handle this concern more effectively than the proposed policy
Thinking:
  - The positive invokes Universalism: concern — saving lives through organ availability benefits all humanity.
  - Strategy: institutional. What existing institutional framework governs organ allocation, and does commercialization undermine it?
  - Transplant medicine operates under strict medical ethics and allocation protocols; a commercial market corrupts these professional standards.
  - No value label names in output. Target length: ~20 words.
Negative Answer (21 words): Commodifying human organs introduces commercial incentives that corrupt medical ethics and undermine the professional standards transplant committees are obligated to uphold.

Example 3 — strategy: pragmatic:
Question: Should we subsidize Wikipedia?
Value: Self-direction: thought
Concept (Self-Direction): The pursuit of independence and self-expression, including autonomy of mind.
This sub-category focuses on:
  • Have freedom of thought: resulting in less censorship, allowing people to make up their mind
  • Be curious: fostering curiosity, promoting discoveries
Positive Answer (invokes the value, 20 words): Wikipedia provides free, open access to knowledge, empowering people to learn and think independently without relying on paywalled sources.
Preferred strategy: pragmatic: challenge the feasibility, cost, or practical implementation of the proposed policy
Thinking:
  - The positive invokes Self-direction: thought — free knowledge access fosters intellectual independence and curiosity.
  - Strategy: pragmatic. Is there a practical reason the policy fails on its own terms?
  - Wikipedia is already financially self-sustaining through voluntary donations; public subsidies create unnecessary fiscal dependency and divert funds.
  - No value label names in output. Target length: ~20 words.
Negative Answer (26 words): Wikipedia already sustains itself through voluntary donations — public subsidies would create unnecessary financial dependency and divert taxpayer money from more critical digital infrastructure investments.

Example 4 — strategy: empirical:
Question: Should we legalize cannabis?
Value: Hedonism
Concept (Hedonism): The pursuit of pleasure and sensuous gratification for oneself.
This sub-category focuses on:
  • Have pleasure: making life enjoyable, providing leisure, providing opportunities to have fun
Positive Answer (invokes the value, 17 words): Legalizing cannabis allows adults to pursue personal enjoyment freely without legal consequences, enhancing their quality of life.
Preferred strategy: empirical: challenge the factual or causal claims in the positive argument using evidence, data, or known real-world outcomes
Thinking:
  - The positive invokes Hedonism — legalization serves personal pleasure and enhances quality of life. This is an empirical causal claim.
  - Strategy: empirical. What does real-world legalization data show about wellbeing outcomes?
  - US state-level data show no measurable wellbeing improvement, while cannabis-use disorder rates increased substantially after commercialization.
  - No value label names in output. Target length: ~17 words.
Negative Answer (23 words): Legalization data from multiple US states show no measurable improvement in reported wellbeing, while cannabis-use disorder rates have increased substantially since commercialization began.

Example 5 — strategy: counter-example:
Question: Should we introduce a wealth tax?
Value: Power: resources
Concept (Power): Social status and prestige, control over people and resources, including control over material goods.
This sub-category focuses on:
  • Have wealth: allowing people to gain wealth and material possession, providing resources to control events
Positive Answer (invokes the value, 22 words): A wealth tax reduces the extreme concentration of resources in the hands of a few, redistributing power more fairly across society.
Preferred strategy: counter-example: cite a real or plausible case where the same policy was tried and led to opposite or harmful results
Thinking:
  - The positive invokes Power: resources — the wealth tax reduces unfair concentration of material resources. This is a testable causal claim.
  - Strategy: counter-example. Has a wealth tax actually achieved redistribution elsewhere?
  - France and Sweden both abolished their wealth taxes after capital flight shrank the tax base without reducing inequality — the opposite of the claimed effect.
  - No value label names in output. Target length: ~22 words.
Negative Answer (22 words): France and Sweden both abolished their wealth taxes after capital outflows shrank the tax base without producing any measurable reduction in inequality.

Example 6 — strategy: contradict (single sub-category value: Tradition):
Question: Should we abolish the monarchy?
Value: Tradition
Concept (Tradition): Respect, commitment, and acceptance of the customs and ideas that one's culture or religion provide.
This sub-category focuses on:
  • Respect traditions: following family customs and religious practices
  • Be devout: holding and transmitting cultural and religious beliefs
Positive Answer (invokes the value, 22 words): The monarchy is a centuries-old institution that embodies national identity and continuity, providing stability and cultural cohesion that elected governments cannot replicate.
Preferred strategy: contradict: challenge whether the concern invoked in the positive argument is valid, relevant, or accurately applied in this policy context — show the value-based premise is misplaced, overstated, or historically inaccurate
Thinking:
  - The positive invokes Tradition — monarchy as a timeless institution providing stability and cultural continuity.
  - Strategy: contradict. Is the historical premise actually true? Are monarchies stable, universal, or necessary for cohesion?
  - Most European monarchies were abolished after WWI; elected republics have proven equally capable of sustaining national identity. The "timeless" claim is historically inaccurate.
  - No value label names in output. Target length: ~22 words.
Negative Answer (28 words): Most European monarchies were abolished following the First World War, and elected republics have since proven equally capable of sustaining national identity and social cohesion without hereditary institutions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON PITFALL — welfare and safety arguments:
When the positive is about harm to people, care for others, or social stability, the most
frequent mistake is writing a negative that is also about welfare or safety — just from the
opposite policy direction. This keeps you inside the same value frame and contaminates the
steering signal.

BAD example (still in the caring frame):
  Question: Should we abolish three-strikes laws?
  Value: Benevolence: caring
  Positive (invokes caring, 14 words): "three-strikes laws can act as a deterrent for serial criminal behavior."
  BAD Negative: "Three-strikes laws often devastate family stability by removing breadwinners,
  creating cycles of poverty and social instability for their children."
  → The positive cares about victims; the negative cares about offenders' families.
    Both are welfare arguments — you are still inside the caring frame.

Any of the following frames would work — choose the one that fits your assigned strategy:

  [cost]       "Mandatory minimum sentencing imposes a substantial fiscal burden on correctional
               systems, diverting public funds from crime prevention programs that demonstrably
               reduce recidivism."
               → Concern: resource allocation. No welfare language.

  [empirical]  "Recidivism data from jurisdictions that replaced mandatory minimums with
               rehabilitative sentencing show lower reoffending rates, undermining the factual
               premise that harsher deterrents reduce repeat offending."
               → Concern: evidence quality. No welfare language.

Other valid escape frames include: institutional capacity, legal precedent, enforcement
feasibility, unintended consequences in an unrelated domain (economic, environmental,
administrative). Any of these breaks out of the welfare/safety frame cleanly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
TOUCHE_SYSTEM_PROMPT = """\
You are a creative dataset constructor for a value-based argumentation study used in AI steering research.

You will be given a policy question, a human value, a rich definition of that value, \
a positive argument that invokes the value, and a preferred rhetorical strategy.

Your task is to write a negative answer using the assigned strategy. Depending on the \
strategy, the negative answer will either (a) approach the question from a completely \
different concern with no relation to the target value, or (b) directly challenge whether \
the target value's concern is valid or accurately applied in this specific context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES:
1. The `negative_answer` must use only concrete policy language. Do NOT name any value \
label (such as "Security: personal", "Universalism", "Achievement", "Tradition", etc.) \
in the final answer. Value labels belong only in the `thinking` block.
2. Do not write a negative answer that simply endorses or softens the target value from \
a different angle. The negative must genuinely oppose or undermine the concern that \
drives the positive argument.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Available strategies:
  • pragmatic       — challenge the feasibility, cost, or practical implementation of the policy.
  • empirical       — challenge the factual or causal claims using evidence, data, or known outcomes.
  • counter-example — cite a real or plausible case where the same policy led to opposite results.
  • side-effects    — argue the policy produces serious unintended consequences in a different domain.
  • institutional   — argue existing rules or institutions already address this concern more effectively.
  • contradict      — challenge whether the concern invoked in the positive argument is valid, \
relevant, or accurately applied in this context; show the value-based premise is misplaced, \
overstated, or historically inaccurate.

Step-by-step reasoning:
1. Identify what the target value means in this specific policy context, using its concept \
and sub-categories.
2. Understand what claim the positive argument makes and exactly how it invokes the target value.
3. Read the preferred strategy. Decide concretely how to apply it to this specific question \
and positive argument.
4. Identify the specific counter-claim, evidence, consequence, or institutional point your \
argument will make.
5. Write the negative answer using concrete policy language only. Do NOT name any value \
label in the output.
6. Self-check: (a) Does the negative endorse the target value in another form? If yes, revise. \
(b) Does the negative name a value label? If yes, rephrase with concrete language. \
(c) What is the core concern of the positive — harm to people, welfare, care for others, \
reliability, safety, or social stability? If your negative's core concern is also about harm, \
welfare, care, reliability, safety, or stability — just from a different policy direction — \
you are still in the same value frame. Choose instead: cost, institutional capacity, evidence \
quality, precedent, or consequences in a completely unrelated domain.

Output format — respond with a JSON object:
  - "thinking": your step-by-step reasoning (2–4 sentences)
  - "negative_answer": the final argument

Guidelines for the negative_answer:
- Do NOT write in first-person. Do NOT start with "I".
- Match the register and directness of the positive answer — if it is formal, be formal; if it is
  plain and assertive, be equally so. Avoid academic hedging ("it may be argued," "potentially")
  when the positive is direct. Sentence structure may vary freely.
- Length: the negative_answer should approximately match the word count of the positive. A
  specific target is in the user prompt — treat it as a guideline. Argument quality comes first,
  but do not write substantially more than the positive. Only for single-clause positives, a single
  direct clause is sufficient; do not add sentences just to elaborate.
- Be specific and direct. A precise claim is stronger than a vague generalisation — but do not
  elaborate beyond what the argument requires.
- No value label names in the output.
"""

# ---------------------------------------------------------------------------
# User prompt template
# Callers must supply: examples, definition_block, question, positive_answer,
# strategy_hint, value_family_warning, positive_word_count,
# positive_word_count_plus_10
#
# value_family_warning is a non-empty string only for Benevolence and Security
# values (see get_value_family_warning); empty string for all others.
# ---------------------------------------------------------------------------
TOUCHE_USER_PROMPT = """\
Here are a few examples showing the thinking process and final answer:
{examples}

Now apply the same process for the following:

{definition_block}

Question: {question}
Positive Answer (invokes the value): {positive_answer}

Preferred strategy: {strategy_hint}
If this strategy genuinely does not fit this question and positive answer, choose the \
closest applicable alternative from the list above.
{value_family_warning}
Length guidance: The positive answer is {positive_word_count} words. Aim for approximately \
{positive_word_count}–{positive_word_count_plus_10} words in your negative answer. \
Quality of reasoning matters more than exact word count, but do not write substantially \
more than the positive answer.
"""
