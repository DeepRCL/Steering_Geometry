from typing import Optional, Callable
from functools import partial
from tqdm import trange

import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoXForCausalLM, FalconForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import GenerationConfig

from ._config import _DEFAULT_SAMPLING_PARAMS, _FULL_LLM_NAMES, _DEFAULT_CHAT_TEMPLATE
from ..steer import get_steer_model


class HuggingFaceLM:
    def __init__(
        self,
        model_name: str,
        steer_name: Optional[str] = None,
        default_generation_config: Optional[GenerationConfig] = None,
        steer_model_kwargs: dict = {},
        steer_layer_idx: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # set device and dtype
        if device is None:
           device = "auto" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        
        # set model and tokenizer
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            _FULL_LLM_NAMES[model_name] if model_name in _FULL_LLM_NAMES.keys() else model_name, 
            device_map = device, 
            torch_dtype = self.dtype,
            # trust_remote_code=True, 
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            _FULL_LLM_NAMES[model_name] if model_name in _FULL_LLM_NAMES.keys() else model_name, 
            device_map = device, 
            torch_dtype = self.dtype,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left' 
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = _DEFAULT_CHAT_TEMPLATE
        
        # set default generation config
        if default_generation_config is None:
            self.default_generation_config = GenerationConfig(
                do_sample = False,
                max_new_tokens = _DEFAULT_SAMPLING_PARAMS.get('max_tokens', 64),
                temperature = _DEFAULT_SAMPLING_PARAMS.get('temperature', 0.0),
                top_p = _DEFAULT_SAMPLING_PARAMS.get('top_p', 1.0),
                top_k = _DEFAULT_SAMPLING_PARAMS.get('top_k', -1),
                seed = _DEFAULT_SAMPLING_PARAMS.get('seed', 42),
                repetition_penalty = _DEFAULT_SAMPLING_PARAMS.get('repetition_penalty', 1.15),
                no_repeat_ngram_size = _DEFAULT_SAMPLING_PARAMS.get('no_repeat_ngram_size', 2),
            )
        else:
            self.default_generation_config = default_generation_config
        
        # set steer model
        if steer_name is not None and steer_name != "NoSteer":
            self.steer_model = get_steer_model(steer_name, **steer_model_kwargs)
        else:
            self.steer_model = None
            
        self.steer_layer_idx = steer_layer_idx
    
    
    def generate(
        self,
        prompts: list[str],
        generation_config: Optional[GenerationConfig] = None,
        steer: bool = False,
        steer_kwargs: Optional[dict] = {},
    ) -> list[str]:
        if generation_config is None:
            generation_config = self.default_generation_config
            
        inputs = self.tokenizer(prompts, return_tensors = 'pt', padding = True).to(self.model.device)
        if steer:
            self.register_steer_hook(-1, steer_kwargs)
            outputs = self.model.generate(
                **inputs, 
                generation_config = generation_config,
            )
            self.remove_steer_hook()
        else:
            outputs = self.model.generate(**inputs, generation_config = generation_config)
        # left padding, so directly use prompt length as response start index
        prompt_len = inputs.attention_mask.shape[1]
        raw_responses = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens = True)
        responses = [response.split("\nQ:")[0] for response in raw_responses]
        return responses
        
        
    def chat(
        self, 
        messages: list[list[dict]],
        generation_config: Optional[GenerationConfig] = None,
        steer: bool = False,
        steer_kwargs: Optional[dict] = {},
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ) -> list[str]:
        assert self.tokenizer.chat_template is not None
        formatted_prompts = self.tokenizer.apply_chat_template(
            messages, tokenize = False,
            add_generation_prompt = add_generation_prompt,
            continue_final_message = continue_final_message,
        )
        return self.generate(
            formatted_prompts, generation_config, steer, steer_kwargs,
        )
        
    
    @torch.no_grad()
    def eval_binary_choice(
        self,
        question: str,
        correct_choice: str,
        wrong_choice: str,
        steer: bool = False,
        steer_kwargs: Optional[dict] = {},
    ):
        template = (
            "Choose the single best answer to the question from the two choices.\n"
            "Rules: Output only the letter A or B. No punctuation, words, or explanation.\n\n"
            "Question: {question}\n"
            "Choices:\n"
            "A) {ans_A}\n"
            "B) {ans_B}\n"
            "Answer:"
        )
        
        if np.random.random() < 0.5:
            ans_A, ans_B = correct_choice, wrong_choice
            true_ans = 'A'  
        else:
            ans_A, ans_B = wrong_choice, correct_choice
            true_ans = 'B'
            
        tok_A = self.tokenizer.encode(" A", add_special_tokens = False)[-1]
        tok_B = self.tokenizer.encode(" B", add_special_tokens = False)[-1]
        choices_toks = torch.tensor([tok_A, tok_B])
         
        prompt = template.format(question = question, ans_A = ans_A, ans_B = ans_B)
        prompt_inputs = self.tokenizer(prompt, return_tensors = "pt")
        if steer and self.steer_model is not None:
            self.register_steer_hook(-1, steer_kwargs)
            logits = self.model(**prompt_inputs.to(self.model.device)).logits
            self.remove_steer_hook()
        else:
            logits = self.model(**prompt_inputs.to(self.model.device)).logits
        choice_log_probs = logits[0, -1].log_softmax(dim = -1)[choices_toks].softmax(dim = -1).cpu().numpy()
        
        if choice_log_probs[0] > choice_log_probs[1]:
            pred = 'A'
        elif choice_log_probs[1] > choice_log_probs[0]:
            pred = 'B'
        else:
            pred = 'E'
        
        # print(choice_log_probs, pred, true_ans)
        if pred == true_ans:
            return 1.0, choice_log_probs
        elif pred == 'E':
            return 0.5, choice_log_probs
        else:
            return 0.0, choice_log_probs
        
        
    @torch.no_grad()
    def compute_answer_prob(
        self,
        question: str,
        answers: list[str],
        steer: bool = False,
        steer_kwargs: Optional[dict] = {},
        prompt_template: str = "Q: {question}\nA: ",
    ) -> Tensor:
        formatted_question = prompt_template.format(question = question)
        question_len = self.tokenizer(formatted_question, return_tensors = "pt").input_ids.shape[1]
        prompts = [formatted_question + answer for answer in answers]
        
        choice_log_probs = []
        for prompt in prompts:
            full_inputs = self.tokenizer(prompt, return_tensors = "pt")
            if steer and self.steer_model is not None:
                self.register_steer_prob_hook(question_len - 1, steer_kwargs)
                outputs = self.model(**full_inputs.to(self.model.device))
                self.remove_steer_prob_hook()
            else:
                outputs = self.model(**full_inputs.to(self.model.device))
            answer_ids = full_inputs.input_ids[0, question_len:]
            answer_logits = outputs.logits[0, question_len-1:-1]
            full_log_probs = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = full_log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
            choice_log_prob = token_log_probs.sum().item()
            choice_log_probs.append(choice_log_prob)
        return torch.tensor(choice_log_probs).softmax(dim = -1)
    
    
    def fit_steer_model(self, *args, **kwargs) -> None: 
        if self.steer_model is None:
            return
        self.steer_model.fit(*args, **kwargs)
    
    
    @torch.no_grad()
    def extract_message_eos_activations(
        self,
        messages: list[list[dict]],
        layer_idx: Optional[int] = None,
    ) -> list[Tensor]:
        assert self.tokenizer.chat_template is not None, "Chat template is not set"
        formatted_prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
            continue_final_message = False,
        )
        return self.extract_prompt_eos_activations(formatted_prompts, layer_idx)
    
    
    @torch.no_grad()
    def extract_prompt_eos_activations(
        self,
        prompts: list[str],
        layer_idx: Optional[int] = None,
    ) -> list[Tensor]:
        if layer_idx is None:
            layer_idx = len(self.model.model.layers) // 2 - 1
        inputs = self.tokenizer(prompts, return_tensors = 'pt', padding = True).to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states = True)
        hidden_states = outputs.hidden_states[1:][layer_idx]
        # left padding settings
        return hidden_states[:, -1, :]
        
    
    def register_steer_hook(
        self, 
        steer_position_idx: int,
        steer_kwargs: dict,  
    ):
        assert hasattr(self, 'steer_model')
        self.hooks = []
        target_layer = self._get_target_layer()
        handle = target_layer.register_forward_hook(partial(
            self.steer_hook_func, 
            steer_position_idx = steer_position_idx,
            steer_kwargs = steer_kwargs,
        ))
        self.hooks.append(handle)
        
        
    def register_steer_prob_hook(
        self,
        steer_start_idx: int,
        steer_kwargs: dict,
    ):
        assert hasattr(self, 'steer_model')
        self.prob_hooks = []
        target_layer: nn.Module = self._get_target_layer()
        handle = target_layer.register_forward_hook(partial(
            self.steer_prob_hook_func, 
            steer_start_idx = steer_start_idx,
            steer_kwargs = steer_kwargs,
        ))
        self.prob_hooks.append(handle)
        
        
    def remove_steer_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    
    def remove_steer_prob_hook(self):
        for hook in self.prob_hooks:
            hook.remove()
        self.prob_hooks = []
        
        
    def steer_hook_func(
        self,
        module: nn.Module,
        input: tuple[Tensor, ...],
        output: Tensor,
        steer_position_idx: int,
        steer_kwargs: dict,
    ) -> Tensor:
        assert hasattr(self, 'steer_model')
        hidden, reassemble = _extract_and_set_hidden(output)
        batch_idx = torch.arange(hidden.shape[0], device = hidden.device)
        hidden = hidden.clone()
        hidden[batch_idx, steer_position_idx] = self.steer_model.steer(
            hidden[batch_idx, steer_position_idx], **steer_kwargs
        )
        return reassemble(hidden)
    
    
    def steer_prob_hook_func(
        self,
        module: nn.Module,
        input: tuple[Tensor, ...],
        output: Tensor,
        steer_start_idx: int,
        steer_kwargs: dict,
    ) -> Tensor:
        assert hasattr(self, 'steer_model')
        hidden, reassemble = _extract_and_set_hidden(output)
        B, S, _ = hidden.shape
        batch_idx = torch.arange(B, device = hidden.device)
        hidden = hidden.clone()
        for pos in range(steer_start_idx, S):
            h = hidden[batch_idx, pos]
            hidden[batch_idx, pos] = self.steer_model.steer(h, **steer_kwargs)
        return reassemble(hidden)
    
    def _get_target_layer(self) -> nn.Module:
        if isinstance(self.model, GPTNeoXForCausalLM):
            return self.model.gpt_neox.layers[self.steer_layer_idx]
        elif isinstance(self.model, FalconForCausalLM):
            return self.model.transformer.h[self.steer_layer_idx]
        else:
            return self.model.model.layers[self.steer_layer_idx]
        


def _extract_and_set_hidden(output) -> tuple[Tensor, Callable]:
    if isinstance(output, tuple):
        hidden, rest = output[0], output[1:]
        def reassemble(h): return (h, *rest)
    elif hasattr(output, "last_hidden_state"):  # ModelOutput-like
        hidden = output.last_hidden_state
        def reassemble(h):
            output.last_hidden_state = h
            return output
    else:  # plain Tensor
        hidden = output
        def reassemble(h): return h
    return hidden, reassemble
   

def batch_chat(
    model: HuggingFaceLM,
    messages: list[list[dict]],
    T: float = 1.0,
    batch_size: int = 10,
):
    num_batches = (len(messages) + batch_size - 1) // batch_size
    outputs = []
    for i in trange(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(messages))
        batch_messages = messages[start_idx:end_idx]
        steer = True if model.steer_model is not None else False
        batch_outputs = model.chat(batch_messages, steer = steer, steer_kwargs = dict(T = T))
        outputs.extend(batch_outputs)
    return outputs


def batch_generate(
    model: HuggingFaceLM,
    prompts: list[str],
    T: float = 1.0,
    batch_size: int = 10,
):
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    outputs = []
    for i in trange(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        steer = True if model.steer_model is not None else False
        batch_outputs = model.generate(batch_prompts, steer = steer, steer_kwargs = dict(T = T))
        outputs.extend(batch_outputs)
    return outputs