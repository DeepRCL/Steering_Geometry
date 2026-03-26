from transformers import pipeline
import json
import re
from prompt import (
    TEST_PROMPT
)


class DatasetConstructionPipeline:
    
    def __init__(self, model_id="Qwen/Qwen3.5-35B-A3B", max_new_tokens=256):
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading model: {model_id}")
        self.pipe = pipeline("image-text-to-text", model=model_id)
    
    def _generate(self, prompt, json_key=None):
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )
        
        full_content = outputs[0]["generated_text"][-1]["content"]
        clean_content = full_content.strip()
        
        # Remove prefixes like "analysis", "assistantfinal", etc.
        # Split by these markers and take the last part (most likely to contain JSON)
        for marker in ["assistantfinal", "JSON:", "json:", "Answer:", "Output:"]:
            if marker in clean_content:
                clean_content = clean_content.split(marker)[-1].strip()
        
        # Remove markdown code blocks if present
        if clean_content.startswith("```json"):
            clean_content = clean_content.replace("```json", "", 1).replace("```", "", 1).strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.replace("```", "", 1).strip()
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3].strip()
        
        # If no json_key is provided, return the raw content
        if not json_key:
            return clean_content
        
        # Try multiple strategies to extract the value
        
        # Strategy 1: Look for JSON pattern and parse it
        json_matches = list(re.finditer(r'\{[^{}]*"[^"]*"\s*:\s*"[^"]*"[^{}]*\}', clean_content, re.DOTALL))
        
        # Try parsing from the last match first (most likely to be the final answer)
        for json_match in reversed(json_matches):
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                
                if json_key in data:
                    result = data[json_key]
                    print(f"Successfully extracted '{json_key}' from JSON")
                    return result.strip() if isinstance(result, str) else str(result)
                    
            except json.JSONDecodeError:
                continue
        
        # Strategy 2: Use regex to extract value directly
        patterns = [
            r'"' + re.escape(json_key) + r'"\s*:\s*"([^"]+)"',
            r"'" + re.escape(json_key) + r"'\s*:\s*'([^']+)'",
            re.escape(json_key) + r'\s*:\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_content, re.DOTALL)
            if match:
                result = match.group(1).strip()
                print(f"Extracted '{json_key}' using regex pattern")
                return result
        
        # Strategy 3: Look for the key followed by text
        fallback_pattern = re.escape(json_key) + r'\s*[:\-]\s*(.+?)(?:\n|$)'
        match = re.search(fallback_pattern, clean_content, re.IGNORECASE)
        if match:
            result = match.group(1).strip().strip('"\'')
            print(f"Extracted '{json_key}' using fallback pattern")
            return result
        
        # Last resort: return cleaned content
        print(f"Warning: Could not extract '{json_key}', returning cleaned content")
        return clean_content
    

   