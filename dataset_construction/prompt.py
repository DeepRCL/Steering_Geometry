# We have positive answers for the valuebench dataset. We need to create a system prompt that will be used to generate the negative answers.
VALUEBENCH_POSITIVE_SYSTEM="""
You are a helpful psycholinguistics assistant and a dataset constructor. You are given a question, a value and a positive answer. The positive answer represents the value and is in agreement with it. You need to generate a negative answer that disagrees or contradicts with the given value.
You must always respond in valid JSON format with a single key 'negative_answer' containing the answer. Do not include any other text.
Rules:
- The negative answer must be plausible and natural-sounding
- The negative answer should clearly contradict or oppose the value — not merely express uncertainty or a moderate view
- It should clearly oppose or contradict the provided value
- Match the tone and length of the positive answer
- Do not explain yourself — output only the negative answer text
"""

VALUEBENCH_POSITIVE_USER="""
Here are a few examples of questions and positive answers:
{examples}

Now, given the following question, value and positive answer, generate a negative answer that contradicts with the value:
Question: {question}
Value: {value}
Positive Answer (in agreement with the value): {positive_answer}
"""



def create_messages(system_prompt, user_prompt_template, text):
    user_content = user_prompt_template.replace("{text}", text)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]