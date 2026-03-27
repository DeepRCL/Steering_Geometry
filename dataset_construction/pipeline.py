from transformers import pipeline
import json
import re
from prompt import (
    VALUEBENCH_POSITIVE_SYSTEM,
    VALUEBENCH_POSITIVE_USER,
    EXAMPLES_POSITIVE
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

    def _generate(self, messages, json_key=None):
        outputs = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False, 
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()
    

    def create_answer(self, row):
        question = row['question']
        value = row['value']
        positive_answer = row['positive_answer']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VALUEBENCH_POSITIVE_SYSTEM}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VALUEBENCH_POSITIVE_USER.format(
                            examples=EXAMPLES_POSITIVE,
                            question=question,
                            value=value,
                            provided_answer=positive_answer,
                        ),
                    }
                ],
            },
        ]

        return self._generate(messages)

    def parse_output(self, output):
        pass

