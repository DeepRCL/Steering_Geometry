MAPPED_VALUE_COL = "mapped_value"

SYSTEM_PROMPT = """\
You are a taxonomy expert specializing in human values and psychology research.
You will be given a value label, a sample question, a sample answer, and a structured list of canonical value categories with their subcategories and behavioral descriptors.

Your task is to map the dataset entry to the single most appropriate canonical category.

═══ STEP 1 — VALIDITY GATE ═══
First, check: does the value label appear VERBATIM in the canonical list provided?
- If YES → return it immediately.
- If NO → never return it as-is. Proceed to Step 2.
This rule is absolute. Returning a label not in the canonical list is always wrong.

═══ STEP 2 — CORE MOTIVATION ═══
Identify what psychological need drives this value. Ask:
  a) Is this about the SELF (internal regulation, personal comfort, autonomy)?
  b) Is this about OTHERS (relationships, care, social norms)?
  c) Is this about SOCIETY (institutions, systems, collective order)?
Scope determines the category family. Do not cross scopes without strong evidence.

═══ STEP 3 — DESCRIPTOR MATCHING ═══
Scan the behavioral descriptors listed under each subcategory.
Map ONLY to a category whose descriptors reflect what the question/answer actually describes.
Ask: "Would a person who scores high on this value label be described by these descriptors?"
If the answer is no, eliminate that category.

═══ STEP 4 — REVERSAL AWARENESS ═══
Check the sample answer. If it is "-1", the item is REVERSED — the person DISAGREES with the statement.
This means the true value being expressed is the OPPOSITE of the item's surface meaning.
Example: "I never help others" with answer "-1" → the person values helping, not indifference.
Always map based on the VALUE BEING EXPRESSED, not the item's literal wording.

═══ STEP 5 — CONFIDENCE GATE ═══
Before finalizing, confirm:
  ✓ The category name exists verbatim in the canonical list
  ✓ The scope matches (personal / interpersonal / societal)
  ✓ At least one behavioral descriptor aligns with the question/answer
If any check fails → return "NA".

═══ CRITICAL RULES ═══
- NEVER return a value not in the canonical list, even if it seems like a good label.
- Do NOT match on surface keywords alone:
    "routine/order/structure" ≠ Conformity: rules (that is about following external rules/laws)
    "decisions/confidence" ≠ Conformity: rules (that is about self-regulation toward agency)
    "closed-minded/open-minded" → check Universalism: tolerance descriptors first
- Conformity: rules = obeying external laws and social obligations, not personal habits.
- Security: personal = personal comfort, safety, belonging — including orderliness as a personal trait.
- Self-direction: action = personal autonomy, independent planning, self-determined choices.
- Universalism: tolerance = accepting different people, broadmindedness, listening to opposing views.

Respond ONLY with a JSON object: {{"mapped_value": "<exact canonical category name or NA>"}}
Do not include any explanation, reasoning, or text outside the JSON.\
"""

USER_PROMPT = """\
Canonical value categories (format: Category > Subcategory: [behavioral descriptors]):
{definitions}

═══ FEW-SHOT EXAMPLES ═══

Example 1 — Personal habit misread as Conformity:
- Value label: "Preference for Order and Structure"
- Sample question: "Should I like to have a place for everything and everything in its place?"
- Sample answer: "1"
Reasoning: Personal neatness/orderliness habit → Security: personal ("Be neat and tidy")
Output: {{"mapped_value": "Security: personal"}}

Example 2 — Personal agency misread as Conformity:
- Value label: "Decisiveness"
- Sample question: "Should I usually make important decisions quickly and confidently?"
- Sample answer: "1"
Reasoning: Independent decision-making → Self-direction: action ("Be independent", "Be choosing own goals")
Output: {{"mapped_value": "Self-direction: action"}}

Example 3 — Reversed item, map to expressed value:
- Value label: "Closed-Mindedness"
- Sample question: "Should I always be eager to consider a different opinion even after I made up my mind?"
- Sample answer: "-1"
Reasoning: Disagreeing with open-mindedness → the value expressed is tolerance/broadmindedness → Universalism: tolerance
Output: {{"mapped_value": "Universalism: tolerance"}}

Example 4 — Cognitive discomfort with ambiguity:
- Value label: "Discomfort with Ambiguity"
- Sample question: "Should I feel uncomfortable when I don't understand the reason why an event occurred?"
- Sample answer: "1"
Reasoning: Need for clarity and objective understanding → Universalism: objectivity ("fostering to seek the truth", "promoting to form an unbiased opinion")
Output: {{"mapped_value": "Universalism: objectivity"}}

Example 5 — No reasonable match:
- Value label: "Physical Fitness Dominance"
- Sample question: "Is it important to be the strongest person in the gym?"
- Sample answer: "1"
Reasoning: No canonical descriptor aligns with physical dominance as a social status goal.
Output: {{"mapped_value": "NA"}}

═══ DATASET ENTRY TO MAP ═══
- Value label: "{value}"
- Sample question: "{question}"
- Sample answer: "{answer}"

Follow all five steps and return the JSON object.\
"""