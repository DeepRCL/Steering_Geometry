from transformers import pipeline
import json
import re
from prompt import (
    PROMPT_CONFIG,
    VALUEBENCH_SYSTEM_PROMPT,
    VALUEBENCH_USER_PROMPT,
    EXAMPLES,
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
    

    def create_answer(self, row, direction="positive_to_negative"):
        config = PROMPT_CONFIG[direction]
        question = row['question']
        value = row['value']
        provided_answer = row[config['source_col']]

        system = VALUEBENCH_SYSTEM_PROMPT.format(**config)
        user = VALUEBENCH_USER_PROMPT.format(
            examples=EXAMPLES[direction],
            question=question,
            value=value,
            provided_answer=provided_answer,
            source_type=config["source_type"],
            target_type=config["target_type"],
            source_relation=config["source_relation"],
            source_type_capitalized=config["source_type"].capitalize(),
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user
                    }
                ],
            },
        ]

        return self._generate(messages)

    def parse_output(self, output):
        pass

