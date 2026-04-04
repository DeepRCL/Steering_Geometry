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


# ---------------------------------------------------------------------------
# Few-shot examples (all in Touche argumentative policy style)
# Each example shows the chain-of-thought reasoning before the final answer.
# Examples cover both multi-sub-value and minimal-sub-value (Hedonism,
# Power: resources) cases so the LLM can handle all 20 value types.
# ---------------------------------------------------------------------------
TOUCHE_EXAMPLES = """
Example 1 — value with multiple sub-categories; strategy: challenge (value self-contradicts):
Question: Should we ban factory farming?
Value: Security: personal
Concept (Security): The pursuit of safety, harmony, and stability of society, relationships, and self.
This sub-category focuses on:
  • Have good health: avoiding diseases, preserving health
  • Have a comfortable life: providing subsistence income, resulting in having no financial worries
Positive Answer (invokes the value): Factory farming allows for the production of cheap food, which is a necessity for families surviving on a low income.
Thinking:
  - The positive argument claims factory farming serves personal security via affordable food.
  - But personal security also includes health. Factory farming is linked to contamination and antibiotic resistance.
  - The same value the argument invokes (personal security) is actually undermined by factory farming — a contradiction.
  - Strategy: challenge — the value is internally contradicted in this context.
Negative Answer (opposes the value): The low prices of factory-farmed food are offset by the serious health risks it creates — contaminated products and antibiotic-resistant bacteria endanger the very personal security of the households this argument claims to protect.

Example 2 — value with multiple sub-categories; strategy: competing value:
Question: Should we legalize organ trade?
Value: Universalism: concern
Concept (Universalism): Understanding, appreciation, tolerance, and protection for the welfare of all people.
This sub-category focuses on:
  • Have equality: providing all people with equal opportunities in life
  • Be just: protecting the weak and vulnerable in society
Positive Answer (invokes the value): Legalizing organ trade would save thousands of lives by increasing the supply of available organs for those in need.
Thinking:
  - The positive argument appeals to universalism — saving lives benefits all of humanity.
  - But who would actually sell organs? Inevitably the poor and desperate.
  - A legal market would exploit economic inequality, violating the equality and justice sub-values of this very same value.
  - Strategy: competing value — dignity and economic justice outweigh the utilitarian life-saving rationale.
Negative Answer (opposes the value): A legal organ market would turn the poor into suppliers of last resort — justice cannot be built on a system that prices human bodies differently based on economic desperation.

Example 3 — value with multiple sub-categories; strategy: hidden trade-off:
Question: Should we subsidize Wikipedia?
Value: Self-direction: thought
Concept (Self-Direction): The pursuit of independence and self-expression, including autonomy of mind.
This sub-category focuses on:
  • Have freedom of thought: resulting in less censorship, allowing people to make up their mind
  • Be curious: fostering curiosity, promoting discoveries
Positive Answer (invokes the value): Wikipedia provides free, open access to knowledge, empowering people to learn and think independently without relying on paywalled sources.
Thinking:
  - The positive argument claims subsidizing Wikipedia promotes intellectual freedom.
  - But government subsidization creates dependency and gives the state indirect influence over what counts as reliable knowledge.
  - This is a hidden trade-off: funding the platform risks turning it into a state-endorsed epistemic filter, the opposite of free thought.
  - Strategy: trade-off — the benefit claimed comes at the cost of the very value invoked.
Negative Answer (opposes the value): Subsidizing Wikipedia hands government money to a platform that can silently shape what counts as reliable knowledge, creating a state-endorsed filter over public thought rather than genuine intellectual freedom.

Example 4 — value with minimal sub-categories (Hedonism: only "Have pleasure"); strategy: assumption attack:
Question: Should we legalize cannabis?
Value: Hedonism
Concept (Hedonism): The pursuit of pleasure and sensuous gratification for oneself.
This sub-category focuses on:
  • Have pleasure: making life enjoyable, providing leisure, providing opportunities to have fun
Positive Answer (invokes the value): Legalizing cannabis allows adults to pursue personal enjoyment freely without legal consequences, enhancing their quality of life.
Thinking:
  - The positive argument assumes cannabis consistently produces pleasure and enhances life quality.
  - However, cannabis has well-documented links to dependency, anxiety, and diminished motivation in regular users.
  - The assumption that legalization reliably delivers pleasure is empirically questionable.
  - Strategy: assumption attack — challenge the factual premise that cannabis serves hedonistic goals.
Negative Answer (opposes the value): Regular cannabis use frequently leads to dependency, anxiety, and cognitive dulling — outcomes that directly undermine the sustained enjoyment and quality of life that legalization is supposed to deliver.

Example 5 — value with minimal sub-categories (Power: resources: only "Have wealth"); strategy: counter-example:
Question: Should we introduce a wealth tax?
Value: Power: resources
Concept (Power): Social status and prestige, control over people and resources, including control over material goods.
This sub-category focuses on:
  • Have wealth: allowing people to gain wealth and material possession, providing resources to control events
Positive Answer (invokes the value): A wealth tax reduces the extreme concentration of resources in the hands of a few, redistributing power more fairly across society.
Thinking:
  - The positive argument says wealth taxes reduce harmful resource concentration.
  - But empirically, wealth taxes in France and Sweden were repealed because capital fled, shrinking the tax base and harming investment.
  - These counter-examples show that wealth taxes can reduce aggregate wealth without meaningfully redistributing control.
  - Strategy: counter-example — real-world evidence contradicts the claimed benefit.
Negative Answer (opposes the value): France and Sweden both abolished their wealth taxes after finding that capital outflows shrank the tax base without reducing inequality — a tax that drives away productive resources does not redistribute power, it simply destroys it.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
TOUCHE_SYSTEM_PROMPT = """\
You are a creative argument-generation assistant and dataset constructor.

You will be given a policy question, a human value, a rich definition of that value, \
and a positive argument (the premise) that invokes the value.

Your task is to write a negative answer: an argument that opposes or undermines the \
given value as it relates to this question.

Step-by-step reasoning process (think before answering):
1. Identify what the value means in this specific policy context, using its concept and sub-categories.
2. Understand what claim the positive argument makes and how it invokes the value.
3. Select ONE rhetorical strategy from the list below that best applies:
    • Challenge: show that the value itself is misapplied or internally contradictory in this context.
    • Trade-off: expose a hidden cost or sacrifice that invoking this value requires.
    • Competing value: argue that a different, higher-order value outweighs this one here.
    • Counter-example: cite a real or plausible case that disproves the value claim.
    • Assumption attack: challenge the factual or logical premise underlying the positive argument.
4. Draft a negative answer that clearly opposes the value using the chosen strategy.
5. Note: if the value has only one sub-category (e.g., Hedonism → "Have pleasure", \
Power: resources → "Have wealth"), focus your reasoning on the core concept rather than \
sub-category distinctions — the same strategies apply.

Output format — respond with a JSON object containing two keys:
  - "thinking": your brief step-by-step reasoning (2–4 sentences, not shown to end users)
  - "negative_answer": the final argument (2–3 sentences, policy-debate style)

Additional guidelines for the negative_answer:
- Write in an argumentative, policy-debate register — NOT in first-person personal style.
- Be creative and diverse — do not default to simple negation; use the strategy you selected.
- The answer must clearly oppose the stated value — not merely express doubt or nuance.
- Match the directness and confidence of the positive answer.
- Do NOT start with "I".
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
TOUCHE_USER_PROMPT = """\
Here are a few examples showing the thinking process and final answer:
{examples}

Now apply the same step-by-step process for the following:

{definition_block}

Question: {question}
Positive Answer (invokes the value): {positive_answer}
"""
