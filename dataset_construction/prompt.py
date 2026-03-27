EXAMPLES_POSITIVE = """
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
"""


# We have positive answers for the valuebench dataset. We need to create a system prompt that will be used to generate the negative answers.
VALUEBENCH_POSITIVE_SYSTEM="""
You are a helpful psycholinguistics assistant and a dataset constructor. You are given a question, a value and a positive answer. The positive answer represents the value and is in agreement with it. You need to generate a negative answer that disagrees or contradicts with the given value.
You must always respond in valid JSON format with a single key 'negative_answer' containing the answer. Do not include any other text.
Rules:
- The negative answer must be plausible and natural-sounding
- The negative answer should clearly contradict or oppose the value — not merely express uncertainty or a moderate view
- It should clearly oppose or contradict the provided value
- Match the tone and length of the positive answer
- Do not add explanations or extra keys — only output the JSON object with the key 'negative_answer' containing the answer
"""

VALUEBENCH_POSITIVE_USER="""
Here are a few examples of questions and positive answers:
{examples}

Now, given the following question, value and positive answer, generate a negative answer that contradicts with the value:
Question: {question}
Value: {value}
Positive Answer (in agreement with the value): {provided_answer}
"""