TEST_PROMPT="""
You are a helpful geography assistant. 
Answer the user's question. 
You must always respond in valid JSON format with a single key 'capital' containing the answer. Do not include any other text.

Question: what is the capital of France?
"""



def create_messages(system_prompt, user_prompt_template, text):
    user_content = user_prompt_template.replace("{text}", text)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]