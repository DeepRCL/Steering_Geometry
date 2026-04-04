MAPPED_VALUE_COL = "mapped_value"

SYSTEM_PROMPT = """\
You are a taxonomy expert working with a psychology dataset.
You will be given a value label, a sample question, and a sample answer from the dataset, along with a list of canonical value categories.

Your task:
1. If the value label already exactly matches one of the canonical categories, return that category as-is.
2. Otherwise, use the value label AND the question/answer context to find the single best-matching canonical category.
3. If no canonical category is a reasonable match, return "NA".

Respond ONLY with a JSON object: {"mapped_value": "<exact canonical name or NA>"}
Do not include any explanation or text outside the JSON.\
"""

USER_PROMPT = """\
Canonical value categories:
{definitions}

Dataset entry:
- Value label: "{value}"
- Sample question: "{question}"
- Sample answer: "{answer}"

Which canonical category does this best map to?\
"""
