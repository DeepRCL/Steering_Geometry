from transformers import pipeline
import json
import re
from prompt import (
    VALUEBENCH_POSITIVE_SYSTEM,
    VALUEBENCH_POSITIVE_USER,
    create_messages
    )

class DatasetConstructionPipeline:
    
    def __init__(self, model_id="Qwen/Qwen3.5-35B-A3B", max_new_tokens=526):
        self.max_new_tokens = max_new_tokens
        print(f"Loading model: {model_id}")
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device_map="auto",
            dtype="auto",
        )

    def _generate(self, prompt, json_key=None):
        outputs = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False, 
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()
    
    #TODO: fix the name of the function
    def create_answer(self, question):
        messages = [
        {"role": "system", "content": VALUEBENCH_POSITIVE_SYSTEM},
        {"role": "user",   "content": VALUEBENCH_POSITIVE_USER.format(
            examples=EXAMPLES,
            question=question,
            value=value,
            provided_answer=positive_answer,
        )}]

        return self._generate(messages)

    def parse_output(self, output):
        pass

