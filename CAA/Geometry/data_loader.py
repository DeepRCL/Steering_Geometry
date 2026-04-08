import csv
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from .config import SCHWARTZ_CIRCUMPLEX_ORDER

@dataclass
class ContrastivePair:
    sample_id: str
    value: str
    question: str
    positive_answer: str
    negative_answer: str

@dataclass
class EvalInstance:
    sample_id: str
    value: str
    question: str
    positive_answer: str
    negative_answer: str
    pos_is_a: bool # True if (A) is positive_answer, False if (B) is positive_answer

class DataLoader:
    def __init__(self, dataset_path: str, eval_split: float = 0.1, seed: int = 42):
        self.dataset_path = dataset_path
        self.eval_split = eval_split
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Load and group data
        self.train_data: Dict[str, List[ContrastivePair]] = {val: [] for val in SCHWARTZ_CIRCUMPLEX_ORDER}
        self.eval_data: Dict[str, List[EvalInstance]] = {val: [] for val in SCHWARTZ_CIRCUMPLEX_ORDER}
        
        self._load_and_split()

    def _load_and_split(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Temporary grouping
            grouped_data: Dict[str, List[ContrastivePair]] = {}
            
            for row in reader:
                val = row['value']
                # Discard values not in our known list (or handle typos)
                if val not in SCHWARTZ_CIRCUMPLEX_ORDER:
                    continue
                
                pair = ContrastivePair(
                    sample_id=row['id'],
                    value=val,
                    question=row['question'],
                    positive_answer=row['positive_answer'],
                    negative_answer=row['negative_answer']
                )
                grouped_data.setdefault(val, []).append(pair)
        
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            pairs = grouped_data.get(val, [])
            self.rng.shuffle(pairs)
            
            num_eval = int(len(pairs) * self.eval_split)
            
            train_pairs = pairs[num_eval:]
            eval_pairs = pairs[:num_eval]
            
            self.train_data[val] = train_pairs
            
            # For eval, randomly assign positive to A or B
            for ep in eval_pairs:
                pos_is_a = self.rng.choice([True, False])
                self.eval_data[val].append(
                    EvalInstance(
                        sample_id=ep.sample_id,
                        value=ep.value,
                        question=ep.question,
                        positive_answer=ep.positive_answer,
                        negative_answer=ep.negative_answer,
                        pos_is_a=pos_is_a
                    )
                )

    def get_train_pairs(self, value: str) -> List[ContrastivePair]:
        return self.train_data.get(value, [])
    
    def get_eval_instances(self, value: str) -> List[EvalInstance]:
        return self.eval_data.get(value, [])

class PromptFormatter:
    """Handles formatting prompts correctly for activation extraction and evaluation."""
    
    def __init__(self, tokenizer, is_instruct: bool):
        self.tokenizer = tokenizer
        self.is_instruct = is_instruct

    def format_extraction_prompts(self, pair: ContrastivePair) -> Tuple[List[int], List[int]]:
        """
        Formats prompts to extract positive and negative activations.
        Following original CAA, we format as a multiple choice question and 
        append the start of the answer, capturing the activation on the answer token.
        Returns: (pos_tokens, neg_tokens)
        """
        pos_is_a = True # Arbitrary for extraction, let's say A is positive, B is negative
        
        q_text = pair.question
        a_text = pair.positive_answer
        b_text = pair.negative_answer
        
        prompt_text = (
            f"Question: {q_text}\n"
            f"(A) {a_text}\n"
            f"(B) {b_text}"
        )
        
        # Format matching model's expected style (Instruct vs Base)
        if self.is_instruct:
            messages = [{"role": "user", "content": prompt_text}]
            # Don't add generation prompt yet because we need to manually append the answer prefix
            base_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            base_prompt = prompt_text + "\nResponse:"
            
        pos_prompt = base_prompt + " (A"
        neg_prompt = base_prompt + " (B"
        
        pos_tokens = self.tokenizer.encode(pos_prompt, add_special_tokens=True)
        neg_tokens = self.tokenizer.encode(neg_prompt, add_special_tokens=True)
        
        return pos_tokens, neg_tokens

    def format_eval_prompt(self, instance: EvalInstance) -> Tuple[List[int], int, int]:
        """
        Formats prompt to evaluate steering, capturing logits for A and B.
        Returns: (tokens, a_token_id, b_token_id)
        """
        if instance.pos_is_a:
            a_text = instance.positive_answer
            b_text = instance.negative_answer
        else:
            a_text = instance.negative_answer
            b_text = instance.positive_answer
            
        prompt_text = (
            f"Question: {instance.question}\n"
            f"(A) {a_text}\n"
            f"(B) {b_text}"
        )
        
        if self.is_instruct:
            messages = [{"role": "user", "content": prompt_text}]
            base_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            base_prompt = prompt_text + "\nResponse:"
            
        eval_prompt = base_prompt + " ("
        
        tokens = self.tokenizer.encode(eval_prompt, add_special_tokens=True)
        
        # Get token IDs for 'A' and 'B' (without parentheses)
        a_token_id = self.tokenizer.encode("A", add_special_tokens=False)[-1]
        b_token_id = self.tokenizer.encode("B", add_special_tokens=False)[-1]
        
        return tokens, a_token_id, b_token_id
