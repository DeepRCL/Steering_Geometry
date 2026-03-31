EXAMPLES = {
    "positive_to_negative": """
Example1:
Question: Will establishing a consistent routine enable me to enjoy my life?
Value: Preference for Order and Structure
Positive Answer (in agreement with the value): I find that establishing a consistent routine enables me to enjoy my life.
Negative Answer: I find that sticking to a consistent routine drains the joy out of my life.

Example2:
Question: Should I enjoy having a clear structured mode of life?
Value: Preference for Order and Structure
Positive Answer (in agreement with the value): I enjoy having a clear structured mode of life.
Negative Answer: I find a rigidly structured mode of life stifling and suffocating.

Example3:
Question: Should I like to have a place for everything and everything in its place?
Value: Preference for Order and Structure
Positive Answer (in agreement with the value): I like to have a place for everything and everything in its place.
Negative Answer: I find the idea of having a fixed place for everything unnecessarily rigid and constraining.

Example4:
Question: Would a well-ordered life with regular hours suit my temperament?
Value: Preference for Order and Structure
Positive Answer (in agreement with the value): I find that a well ordered life with regular hours suits my temperament.
Negative Answer: I find that a well-ordered life with fixed hours feels monotonous and goes against my temperament.

Example5:
Question: Should I dislike unpredictable situations?
Value: Preference for Predictability
Positive Answer (in agreement with the value): I dislike unpredictable situations.
Negative Answer: I thrive in unpredictable situations and find them exciting rather than troubling.
""",
    "negative_to_positive": """

"""
}


# We have positive answers for the valuebench dataset. We need to create a system prompt that will be used to generate the negative answers.
VALUEBENCH_SYSTEM_PROMPT="""
You are a helpful psycholinguistics assistant and a dataset constructor. You are given a question, a value and a {source_type} answer. The {source_type} answer {source_relation} with the value. You need to generate a {target_type} answer that {target_relation} the given value.
You must always respond in valid JSON format with a single key '{target_type}_answer' containing the answer. Do not include any other text.
Rules:
- The {target_type} answer must be plausible and natural-sounding
- It should clearly {target_relation} the provided value — not merely express uncertainty or a moderate view
- It should clearly oppose or contradict the provided value
- Match the tone and length of the {source_type} answer
- Do not add explanations or extra keys — only output the JSON object with the key '{target_type}_answer' containing the answer
"""

VALUEBENCH_USER_PROMPT="""
Here are a few examples of questions and positive answers:
{examples}

Now, given the following question, value and {source_type} answer, generate a {target_type} answer that {target_relation} the given value:
Question: {question}
Value: {value}
{source_type_capitalized} Answer (in {source_relation} with the value): {provided_answer}
"""

PROMPT_CONFIG = {
    "positive_to_negative": {
        "source_type": "positive",
        "source_relation": "agrees",
        "target_type": "negative",
        "target_relation": "contradicts or opposes",
        "source_col": "positive_answer",
        "target_col": "negative_answer",
    },
    "negative_to_positive": {
        "source_type": "negative",
        "source_relation": "disagrees",
        "target_type": "positive",
        "target_relation": "agrees with or supports",
        "source_col": "negative_answer",
        "target_col": "positive_answer",
    },
}