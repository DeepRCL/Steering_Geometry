from transformers import pipeline
import json
import re
from prompt import TEST_PROMPT


class DatasetConstructionPipeline:
    
    def __init__(self, model_id="Qwen/Qwen3.5-35B-A3B", max_new_tokens=1024):
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
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,       
        )
        #TODO: parse the output to get the JSON
        return outputs[0]["generated_text"]

    def create_answer(self, question):
        messages = create_messages(TEST_PROMPT, question)
        return self._generate(messages)

