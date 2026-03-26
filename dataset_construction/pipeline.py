from transformers import pipeline
import json
import re
from prompt import (
    TEST_PROMPT
)


class DatasetConstructionPipeline:
    
    def __init__(self, model_id="Qwen/Qwen3.5-35B-A3B", max_new_tokens=4096):
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading model: {model_id}")
        self.pipe = pipeline(
            "image-text-to-text", 
            model=model_id, 
        )
    
    def _generate(self, prompt, json_key=None):
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens
        )

        print("\n=== RAW MODEL OUTPUT ===")
        print(outputs[0]["generated_text"])
        print("========================\n")

        return self._qewnparser(outputs, prompt, json_key)
        
    def _qewnparser(self, outputs, prompt, json_key=None):
        raw_text = outputs[0]["generated_text"]

        if isinstance(prompt, str) and raw_text.startswith(prompt):
            raw_text = raw_text[len(prompt):].strip()
            
        clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', raw_text, flags=re.DOTALL).strip()
        
        json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if json_key and json_key in data:
                    return data[json_key]

                return data
                
            except json.JSONDecodeError:
                pass

        return {"error": "No valid JSON found. The model may have hit the max_new_tokens limit."}