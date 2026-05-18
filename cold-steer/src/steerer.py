
## Train
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.func import functional_call
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader, Dataset, Subset
import os
from torch.func import vmap, jacrev, vjp, jvp
from contextlib import contextmanager
import json
from src.utils import partial_cross_entropy, lora_style_update
from collections import defaultdict
import numpy as np

MAX_NEW_TOKENS = 100

def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = torch.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob

def compute_dpo_loss(model, ref_model, inputs, labels, beta=0.1):
    """
    Compute DPO loss for a batch of preference pairs.
    
    Args:
        inputs: Dict containing input_ids, attention_mask for both preferred and rejected
        labels: Dict containing labels for both preferred and rejected responses
        beta: Temperature parameter for DPO (default: 0.1)
    
    Returns:
        loss: DPO loss value
        accuracy: Preference accuracy
    """
    
    # Extract preferred and rejected inputs
    if 'preferred_input_ids' in inputs:
        preferred_inputs = {
            'input_ids': inputs['preferred_input_ids'],
            'attention_mask': inputs['preferred_attention_mask']
        }
        rejected_inputs = {
            'input_ids': inputs['rejected_input_ids'], 
            'attention_mask': inputs['rejected_attention_mask']
        }
        
        preferred_labels = labels['preferred_labels']
        rejected_labels = labels['rejected_labels']
    else:
        preferred_inputs = {
            'input_ids': inputs['matching_input_ids'],
            'attention_mask': inputs['matching_attention_mask']
        }
        rejected_inputs = {
            'input_ids': inputs['not_matching_input_ids'], 
            'attention_mask': inputs['not_matching_attention_mask']
        }
        
        preferred_labels = labels['matching_labels']
        rejected_labels = labels['not_matching_labels']
    
    # Get model outputs for preferred responses
    with torch.no_grad() if ref_model is not None else torch.enable_grad():
        preferred_outputs = model(**preferred_inputs, labels=preferred_labels)
        rejected_outputs = model(**rejected_inputs, labels=rejected_labels)
        
        # Calculate log probabilities for policy model
        policy_preferred_logprobs = -preferred_outputs.loss
        policy_rejected_logprobs = -rejected_outputs.loss
        
        # If using reference model for DPO
        if ref_model is not None:
            with torch.no_grad():
                ref_preferred_outputs = ref_model(**preferred_inputs, labels=preferred_labels)
                ref_rejected_outputs = ref_model(**rejected_inputs, labels=rejected_labels)
                
                ref_preferred_logprobs = -ref_preferred_outputs.loss
                ref_rejected_logprobs = -ref_rejected_outputs.loss
        else:
            # If no reference model, assume reference logprobs are 0
            ref_preferred_logprobs = torch.zeros_like(policy_preferred_logprobs)
            ref_rejected_logprobs = torch.zeros_like(policy_rejected_logprobs)
    
    # Calculate DPO loss
    pi_logratios = policy_preferred_logprobs - policy_rejected_logprobs
    ref_logratios = ref_preferred_logprobs - ref_rejected_logprobs
    
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Calculate accuracy (how often preferred is ranked higher)
    accuracy = (logits > 0).float().mean()
    
    return loss, accuracy

class BaseSteerer():
    def __init__(self, steerable_llm, log_dir, batch_size=1, steer_masking='all', gen_masking='prompt', 
                 max_new_tokens=MAX_NEW_TOKENS, *args, **kwargs) -> None:
        self.steerable_llm = steerable_llm
        self.log_dir = log_dir
        self.device = 'cuda:0'
        self.batch_size = batch_size
        self.steer_masking = steer_masking
        self.max_new_tokens = max_new_tokens
        self.gen_masking = gen_masking
        self.layer_outputs = {}
        self.gen_input_ids = None
        self.gen_attention_mask = None
        
    @contextmanager
    def bypass_steering(self):
        old_value = getattr(self, '_bypass_steering', False)
        self._bypass_steering = True
        try:
            yield
        finally:
            self._bypass_steering = old_value
            
    def get_steering_mask(self, attention_mask):
        if self.steer_masking == 'all':
            if self.gen_masking == 'prompt':
                if attention_mask.shape[1] == 1 and self.gen_attention_mask is not None:
                    return torch.zeros_like(attention_mask.clone().bool())
                else:
                    return attention_mask.clone().bool()
            elif self.gen_masking == 'all':
                return attention_mask.clone().bool()
        elif self.steer_masking == 'last':
            if self.gen_masking == 'prompt':
                if attention_mask.shape[1] == 1 and self.gen_attention_mask is not None:
                    return torch.zeros_like(attention_mask.clone().bool())
                else:
                    steering_mask = torch.zeros_like(attention_mask).bool()
                    steering_mask[:, -1] = 1 # assuming left-pad
                    return steering_mask
            elif self.gen_masking == 'all':
                steering_mask = torch.zeros_like(attention_mask).bool()
                steering_mask[:, -1] = 1 # assuming left-pad
                return steering_mask
        return None
    
    def train(*args, **kwargs):
        pass
            
    def get_intermediate_activations(self, params, inputs):
        def get_layer_output_hook_with_exception(module, input, output, layer_idx=-1):
            activation = output[0].clone() #.detach().requires_grad_(True)
            self.layer_outputs[layer_idx] = activation
            # raise Exception
        try:
            with self.bypass_steering():
                self.layer_outputs = {}
                get_handles = self.steerable_llm.register_steering_hooks(
                    lambda layer_idx: lambda x, y, z: get_layer_output_hook_with_exception(x, y, z, layer_idx=layer_idx))
                out = self.steerable_llm.functional_forward(params=params, inputs=inputs)
        # except:
        #     pass
        finally:
            for handle in get_handles: handle.remove()
        return self.layer_outputs
    
    @torch.no_grad()
    def test_all (self, dataset: Dataset,):
        metrics = defaultdict(lambda: [])
        generations = defaultdict(lambda: [])
        # {'prompt': [], 'response': [], 'gt': []}
        token_probs = defaultdict(lambda: [])
        # {'token_matching_behavior': [], 'token_not_matching_behavior': []}
        print (os.getcwd())
        for datum in tqdm(DataLoader(dataset, batch_size=self.batch_size, shuffle=False), desc='Test'):
            self.reset_steering()
            input_ids = datum['prompt_input_ids']
            attention_mask = datum['prompt_attention_mask']
            if 'token_matching_behavior' in datum:
                try:
                    steer_handles = self.steerable_llm.register_steering_hooks(
                        lambda layer_idx: lambda x, y, z: self.steer_output_hook(x, y, z, 
                                                    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
                                                    steering_mask=self.get_steering_mask(attention_mask=attention_mask), 
                                                layer_idx=layer_idx))
                    last_token_logits = self.steerable_llm(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
                    last_token_probs = torch.softmax(last_token_logits, dim=-1)
                    token_probs['token_matching_behavior'] += last_token_probs[torch.arange(len(datum['token_matching_behavior'])), 
                                                                               datum['token_matching_behavior']].cpu().tolist()
                    token_probs['token_not_matching_behavior'] += last_token_probs[torch.arange(len(datum['token_matching_behavior'])), 
                                                                                   datum['token_not_matching_behavior']].cpu().tolist()
                    metrics['accuracy'] += [(x > y).item() for x, y in zip(last_token_probs[:, datum['token_matching_behavior']], 
                                                               last_token_probs[:, datum['token_not_matching_behavior']])]
                finally:
                    for handle in steer_handles: handle.remove()
            elif 'probs' in datum:
                try:
                    steer_handles = self.steerable_llm.register_steering_hooks(
                        lambda layer_idx: lambda x, y, z: self.steer_output_hook(x, y, z, 
                                                    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
                                                    steering_mask=self.get_steering_mask(attention_mask=attention_mask), 
                                                layer_idx=layer_idx))
                    last_token_logits = self.steerable_llm(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
                    batch_size = last_token_logits.size(0)
                    for batch_idx in range(batch_size):
                        keys = [k for k, prob in datum['probs'].items() if prob[batch_idx] != -1]
                        gt_probs = [datum['probs'][k][batch_idx].cpu() for k in keys]
                        gen_probs = torch.softmax(last_token_logits[batch_idx, keys], dim=-1).cpu()
                        token_probs['gen'].append([float(f'{x:.2f}') for x in gen_probs])
                        token_probs['gt'].append([float(f'{x:.2f}') for x in gt_probs])
                        metrics['tv_dist'].append(float(f'{np.abs(np.array(gt_probs) - np.array(gen_probs)).sum().item()/2:.2f}'))
                        metrics['kl_div'].append(float(f'{(np.array(gt_probs) * np.log(np.array(gt_probs)/np.array(gen_probs))).sum().item():.2f}'))
                finally:
                    for handle in steer_handles: handle.remove()
            else:
                try:
                    self.gen_input_ids = input_ids
                    self.gen_attention_mask = attention_mask
                    input_handle = self.steerable_llm.register_input_hook(lambda x, y, z: self.store_inputs(x, y, z))
                    # need to write generate loop here since inputs, steering_mask need to be changed after generation of every new token
                    steer_handles = self.steerable_llm.register_steering_hooks(
                        lambda layer_idx: lambda x, y, z: self.steer_output_hook(x, y, z, layer_idx=layer_idx))
                    generations['response'] += self.steerable_llm.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                                                 decode=True, max_new_tokens=self.max_new_tokens)
                    generations['prompt'] += self.steerable_llm.tokenizer.batch_decode(datum['prompt_input_ids'], skip_special_tokens=True)
                    if 'response_input_ids_matching_behavior' in datum:
                        generations['gt'] += self.steerable_llm.tokenizer.batch_decode(datum['response_input_ids_matching_behavior'], skip_special_tokens=True)
                finally:
                    input_handle.remove()
                    for handle in steer_handles: handle.remove()
        
        if len (generations['prompt']) > 0:
            json.dump(generations, open('generations.json', 'w'))
            
        return {
            **{k: float(f'{sum(v)/len(v):.2f}') for k, v in metrics.items()},
            **dict(token_probs),
            **dict(generations),
        }
    
    def llm_check (self, gt, gen):
        if gen in gt or gt in gen:
            return True
        else:
            return False
    
    def store_inputs(self, module, input_ids, output):
        self.gen_input_ids = input_ids[0] if type(input_ids) is tuple else input_ids
        if self.gen_attention_mask.shape != self.gen_input_ids.shape:
            self.gen_attention_mask = torch.ones_like(self.gen_input_ids)
        
    def reset_steering(self):
        pass
    
    def steer_output_hook(self, module, input, output, inputs={}, steering_mask=None, layer_idx=-1):
        return output

class LossFDSteerer (BaseSteerer):
    def __init__(
        self,
        steerable_llm,
        epsilon: float = 1e-2,
        eta: float = 1e-2,
        training: str = 'sft',
        training_batch_size: int = 1,
        test_batch_size: int = 1,
        log_dir: str = '.',
        steer_masking: str = 'all',
        gen_masking: str = 'prompt',
    ):
        super().__init__(hydra.utils.instantiate(steerable_llm), log_dir=log_dir, batch_size=test_batch_size, 
                         steer_masking=steer_masking, gen_masking=gen_masking)
        self.epsilon = epsilon
        self.eta = eta
        self.training = training
        self.training_batch_size = training_batch_size
        self.z_eps = None
     
    def train(self, dataset: Dataset,):
        running_grads = {}
        for param in self.steerable_llm.get_steering_params():
            for k, v in param.items():
                if k not in running_grads: running_grads[k] = torch.zeros_like(v)
                
        self.steerable_llm.set_steering_params(requires_grad=True)

        for datum in tqdm(DataLoader(dataset, batch_size=self.training_batch_size, shuffle=False), desc='Train'):
            if self.training == 'sft':
                model_out = self.steerable_llm(input_ids=datum['matching_input_ids'],
                                                attention_mask=datum['matching_attention_mask'],
                                                labels=datum['matching_labels'],)
                loss = model_out.loss
            elif self.training == 'ce':
                model_out = self.steerable_llm(input_ids=datum['prompt_input_ids'],
                                                attention_mask=datum['prompt_attention_mask'],)
                loss = partial_cross_entropy(model_out.logits, datum['probs'])
            elif self.training == 'negative_sft':
                model_out = self.steerable_llm(input_ids=datum['not_matching_input_ids'],
                                                attention_mask=datum['not_matching_attention_mask'],
                                                labels=datum['not_matching_labels'])
                loss = - model_out.loss
            elif self.training == 'dpo':
                # Compute DPO loss
                loss, _ = compute_dpo_loss(self.steerable_llm, None, {k: v for k, v in datum.items() if 'labels' not in k}, 
                                            {k: v for k, v in datum.items() if 'labels' in k}, beta=0.1)

            vals = torch.autograd.grad(loss, self.steerable_llm.get_params(running_grads.keys()), retain_graph=True)
            grads = dict (zip(running_grads.keys(), vals))
            for k in running_grads:
                running_grads[k] += grads[k].detach().clone()
            
        with torch.no_grad():
            mean_grads = {k: v/len(dataset) for k, v in running_grads.items()}
            self.steered_params = {k: v + self.epsilon * mean_grads[k].to(v.device) for k, v in self.steerable_llm.params.items() if k in mean_grads}
            
        self.steerable_llm.set_steering_params(requires_grad=False)
        
    def reset_steering(self):
        self.z_eps = None
        
    def steer_output_hook(self, module, input, output, inputs={}, layer_idx=-1, steering_mask=None):
        # Check if steering is temporarily disabled
        if getattr(self, '_bypass_steering', False):
            return output
        
        if len(inputs) == 0 and self.gen_input_ids is not None and self.gen_attention_mask is not None:
            inputs = {'input_ids': self.gen_input_ids, 'attention_mask': self.gen_attention_mask}
            steering_mask = self.get_steering_mask(self.gen_attention_mask)
            self.reset_steering()

        z = output[0]
        if self.z_eps is None:
            self.z_eps = self.get_intermediate_activations(params=self.steered_params, inputs=inputs)
            
        z_eps = self.z_eps[layer_idx].to(z.device)
        if steering_mask is not None:
            steering_mask = steering_mask.to(z.device)
            z[steering_mask] -= self.eta * (z_eps[steering_mask] - z[steering_mask])/self.epsilon
        else:
            z -= self.eta * (z_eps - z)/self.epsilon
        return z,

class LossFDThreshSteerer (BaseSteerer):
    def __init__(
        self,
        steerable_llm,
        epsilon: float = 1e-2,
        eta: float = 1e-2,
        training: str = 'sft',
        training_batch_size: int = 1,
        test_batch_size: int = 1,
        log_dir: str = '.',
        thresh: float = 1e-8,
        steer_masking: str = 'all',
        gen_masking: str = 'prompt',
    ):
        super().__init__(hydra.utils.instantiate(steerable_llm), log_dir=log_dir, batch_size=test_batch_size, 
                         steer_masking=steer_masking, gen_masking=gen_masking)
        self.epsilon = epsilon
        self.eta = eta
        self.training = training
        self.training_batch_size = training_batch_size
        self.z_eps = None
        self.thresh = thresh
     
    def train(self, dataset: Dataset,):
        running_grads = {}
        for param in self.steerable_llm.get_steering_params():
            for k, v in param.items():
                if k not in running_grads: running_grads[k] = torch.zeros_like(v)
                
        self.steerable_llm.set_steering_params(requires_grad=True)

        for datum in tqdm(DataLoader(dataset, batch_size=self.training_batch_size, shuffle=False), desc='Train'):
            if self.training == 'sft':
                model_out = self.steerable_llm(input_ids=datum['matching_input_ids'],
                                                attention_mask=datum['matching_attention_mask'],
                                                labels=datum['matching_labels'],)
                loss = model_out.loss
            elif self.training == 'ce':
                model_out = self.steerable_llm(input_ids=datum['prompt_input_ids'],
                                                attention_mask=datum['prompt_attention_mask'],)
                loss = partial_cross_entropy(model_out.logits, datum['probs'])
            elif self.training == 'negative_sft':
                model_out = self.steerable_llm(input_ids=datum['not_matching_input_ids'],
                                                attention_mask=datum['not_matching_attention_mask'],
                                                labels=datum['not_matching_labels'])
                loss = - model_out.loss
            elif self.training == 'dpo':
                # Compute DPO loss
                loss, _ = compute_dpo_loss(self.steerable_llm, None, {k: v for k, v in datum.items() if 'labels' not in k}, 
                                            {k: v for k, v in datum.items() if 'labels' in k}, beta=0.1)

            vals = torch.autograd.grad(loss, self.steerable_llm.get_params(running_grads.keys()), retain_graph=True)
            grads = dict (zip(running_grads.keys(), vals))
            for k in running_grads:
                running_grads[k] += grads[k].detach().clone()
            
        with torch.no_grad():
            mean_grads = {k: v/len(dataset) for k, v in running_grads.items()}
            self.steered_params = {k: v + self.epsilon * mean_grads[k].to(v.device) \
                        if (mean_grads[k] * self.epsilon).max() < self.thresh else v \
                for k, v in self.steerable_llm.params.items() if k in mean_grads}
            
        self.steerable_llm.set_steering_params(requires_grad=False)
        
    def reset_steering(self):
        self.z_eps = None
        
    def steer_output_hook(self, module, input, output, inputs={}, layer_idx=-1, steering_mask=None):
        # Check if steering is temporarily disabled
        if getattr(self, '_bypass_steering', False):
            return output
        
        if len(inputs) == 0 and self.gen_input_ids is not None and self.gen_attention_mask is not None:
            inputs = {'input_ids': self.gen_input_ids, 'attention_mask': self.gen_attention_mask}
            steering_mask = self.get_steering_mask(self.gen_attention_mask)
            self.reset_steering()

        z = output[0]
        if self.z_eps is None:
            self.z_eps = self.get_intermediate_activations(params=self.steered_params, inputs=inputs)
            
        z_eps = self.z_eps[layer_idx].to(z.device)
        if steering_mask is not None:
            steering_mask = steering_mask.to(z.device)
            z[steering_mask] -= self.eta * (z_eps[steering_mask] - z[steering_mask])/self.epsilon
        else:
            z -= self.eta * (z_eps - z)/self.epsilon
        return z,
    
    

class LossFDLoraSteerer (BaseSteerer):
    def __init__(
        self,
        steerable_llm,
        epsilon: float = 1e-2,
        eta: float = 1e-2,
        training: str = 'sft',
        training_batch_size: int = 1,
        test_batch_size: int = 1,
        log_dir: str = '.',
        steer_masking: str = 'all',
        gen_masking: str = 'prompt',
    ):
        super().__init__(hydra.utils.instantiate(steerable_llm), log_dir=log_dir, batch_size=test_batch_size, 
                         steer_masking=steer_masking, gen_masking=gen_masking)
        self.epsilon = epsilon
        self.eta = eta
        self.training = training
        self.training_batch_size = training_batch_size
        self.z_eps = None
     
    def train(self, dataset: Dataset,):
        running_grads = {}
        for param in self.steerable_llm.get_steering_params():
            for k, v in param.items():
                if k not in running_grads: running_grads[k] = torch.zeros_like(v)
                
        self.steerable_llm.set_steering_params(requires_grad=True)

        for datum in tqdm(DataLoader(dataset, batch_size=self.training_batch_size, shuffle=False), desc='Train'):
            if self.training == 'sft':
                model_out = self.steerable_llm(input_ids=datum['matching_input_ids'],
                                                attention_mask=datum['matching_attention_mask'],
                                                labels=datum['matching_labels'],)
                loss = model_out.loss
            elif self.training == 'ce':
                model_out = self.steerable_llm(input_ids=datum['prompt_input_ids'],
                                                attention_mask=datum['prompt_attention_mask'],)
                loss = partial_cross_entropy(model_out.logits, datum['probs'])
            elif self.training == 'negative_sft':
                model_out = self.steerable_llm(input_ids=datum['not_matching_input_ids'],
                                                attention_mask=datum['not_matching_attention_mask'],
                                                labels=datum['not_matching_labels'])
                loss = - model_out.loss
            elif self.training == 'dpo':
                # Compute DPO loss
                loss, _ = compute_dpo_loss(self.steerable_llm, None, {k: v for k, v in datum.items() if 'labels' not in k}, 
                                            {k: v for k, v in datum.items() if 'labels' in k}, beta=0.1)

            vals = torch.autograd.grad(loss, self.steerable_llm.get_params(running_grads.keys()), retain_graph=True)
            grads = dict (zip(running_grads.keys(), vals))
            
            low_rank_updates = lora_style_update(grads, rank=16)
            # Apply updates
            for k in running_grads:
                if isinstance(low_rank_updates[k], tuple):
                    A, B = low_rank_updates[k]
                    running_grads[k] += (A @ B).detach().clone()
                else:
                    running_grads[k] += low_rank_updates[k].detach().clone()
            
        with torch.no_grad():
            mean_grads = {k: v/len(dataset) for k, v in running_grads.items()}
            self.steered_params = {k: v + self.epsilon * mean_grads[k].to(v.device) for k, v in self.steerable_llm.params.items() if k in mean_grads}
            
        self.steerable_llm.set_steering_params(requires_grad=False)
        
    def reset_steering(self):
        self.z_eps = None
        
    def steer_output_hook(self, module, input, output, inputs={}, layer_idx=-1, steering_mask=None):
        # Check if steering is temporarily disabled
        if getattr(self, '_bypass_steering', False):
            return output
        
        if len(inputs) == 0 and self.gen_input_ids is not None and self.gen_attention_mask is not None:
            inputs = {'input_ids': self.gen_input_ids, 'attention_mask': self.gen_attention_mask}
            steering_mask = self.get_steering_mask(self.gen_attention_mask)
            self.reset_steering()

        z = output[0]
        if self.z_eps is None:
            self.z_eps = self.get_intermediate_activations(params=self.steered_params, inputs=inputs)
            
        z_eps = self.z_eps[layer_idx].to(z.device)
        if steering_mask is not None:
            steering_mask = steering_mask.to(z.device)
            z[steering_mask] -= self.eta * (z_eps[steering_mask] - z[steering_mask])/self.epsilon
        else:
            z -= self.eta * (z_eps - z)/self.epsilon
        return z,


class LossDirectSteerer(BaseSteerer):
    def __init__(
        self,
        steerable_llm,
        eta: float = 1e-2,
        training: str = 'sft',
        training_batch_size: int = 1,
        test_batch_size: int = 1,
        log_dir: str = '.',
        steer_masking: str = 'all',
    ):
        super().__init__(hydra.utils.instantiate(steerable_llm), log_dir=log_dir, batch_size=test_batch_size, steer_masking=steer_masking)
        self.eta = eta
        self.training = training
        self.training_batch_size = training_batch_size
        
    
    def train(self, dataset: Dataset,):
        running_grads = {}
        for param in self.steerable_llm.get_steering_params():
            for k, v in param.items():
                if k not in running_grads: running_grads[k] = torch.zeros_like(v)
                
        self.steerable_llm.set_steering_params(requires_grad=True)

        for datum in tqdm(DataLoader(dataset, batch_size=self.training_batch_size, shuffle=False), desc='Train'):
            if self.training == 'sft':
                model_out = self.steerable_llm(input_ids=datum['matching_input_ids'],
                                                attention_mask=datum['matching_attention_mask'],
                                                labels=datum['matching_labels'],)
                loss = model_out.loss
            elif self.training == 'dpo':
                # Compute DPO loss
                loss, _ = compute_dpo_loss(self.steerable_llm, None, {k: v for k, v in datum.items() if 'labels' not in k}, 
                                            {k: v for k, v in datum.items() if 'labels' in k}, beta=0.1)

            vals = torch.autograd.grad(loss, self.steerable_llm.get_params(running_grads.keys()), retain_graph=True)
            grads = dict (zip(running_grads.keys(), vals))
            for k in running_grads:
                running_grads[k] += grads[k].detach().clone()
        
        with torch.no_grad():
            mean_grads = {k: v/len(dataset) for k, v in running_grads.items()}
            self.steered_params = {k: v - self.eta * mean_grads[k].to(v.device) for k, v in self.steerable_llm.params.items() if k in mean_grads}
            
        self.steerable_llm.set_steering_params(requires_grad=False)
        

    def reset_steering(self):
        self.z_new = None
        
    def steer_output_hook(self, module, input, output, inputs={}, layer_idx=-1, steering_mask=None):
        # Check if steering is temporarily disabled
        if getattr(self, '_bypass_steering', False):
            return output

        z = output[0]
        if self.z_new is None:
            self.z_new = self.get_intermediate_activations(params=self.steered_params, inputs=inputs)
            
        z_new = self.z_new[layer_idx].to(z.device)
        if steering_mask is not None:
            steering_mask = steering_mask.to(z.device)
            z[steering_mask] = z_new[steering_mask]
        else:
            z = z_new
        return z,

   
class KernelLossSteerer (BaseSteerer):
    def __init__(
        self,
        steerable_llm,
        eta: float = 1e-2,
        training_batch_size: int = 1,
        training: str = 'sft',
        kernel: str = "none",
        log_dir: str = '.',
        override: bool = False,
        steer_masking: str = 'all',
        gen_masking='prompt',
    ):
        super().__init__(steerable_llm, log_dir=log_dir, steer_masking=steer_masking, gen_masking=gen_masking)
        self.layer_outputs = {}
        self.eta = eta
        
        self.training = training
        self.training_batch_size = training_batch_size
        
        self.kernel = kernel
        self.loss_vector = None
        self.override = override
        
        if self.kernel == 'constant':
            self.kernel_fn = lambda output, layer_idx=None, *args, **kwargs: output
        elif self.kernel == 'random_proj':
            self.random_projs = {lidx: torch.rand(out_dim, out_dim) for lidx, out_dim in  \
                zip(self.steerable_llm.steering_layer_indices, self.steerable_llm.steering_out_dims)}
            def kernel_fn (output, inputs, layer_idx=None, *args, **kwargs):
                if layer_idx is None:
                    return {lidx: output[lidx] @ self.random_projs[lidx].to(output[lidx].device) \
                                for lidx in self.steerable_llm.steering_layer_indices}
                else:
                    return output @ self.random_projs[layer_idx].to(output.device)
            self.kernel_fn = kernel_fn
        elif self.kernel == 'wtd_proj':
            self.projs = {lidx: torch.nn.Linear(out_dim, out_dim) for lidx, out_dim in  \
                zip(self.steerable_llm.steering_layer_indices, self.steerable_llm.steering_out_dims)}
            def kernel_fn (output, inputs, layer_idx=None, *args, **kwargs):
                if layer_idx is None:
                    return {self.projs[lidx].to(output.device)(output) for lidx in self.steerable_llm.steering_layer_indices}
                else:
                    return self.projs[layer_idx].to(output.device)(output)
            self.kernel_fn = kernel_fn
        elif self.kernel == 'entk_brute_force':
            def kernel_fn (inputs, *args, **kwargs):
                z_theta = lambda params: self.get_intermediate_activations(params=params, inputs=inputs)
                return jacrev(z_theta)(self.steerable_llm.get_steering_params())
            self.kernel_fn = kernel_fn
        elif self.kernel == 'entk_last':
            def kernel_fn (inputs, *args, **kwargs):
                def get_activation_from_last (some_params):
                    params = self.steerable_llm.get_steering_params()
                    for k, v in some_params.items():
                        params[k] = v
                    return self.get_intermediate_activations(params=params, inputs=inputs)
                return jacrev(get_activation_from_last)(self.steerable_llm.get_layers_params_steering([self.steerable_llm.steering_layer_index]))
            self.kernel_fn = kernel_fn
        elif self.kernel == 'entk_proj_loss':
            def kernel_fn (inputs, vector, *args, **kwargs):
                return self._compute_grads_with_vjp(inputs=inputs, vector=vector)
            self.kernel_fn = kernel_fn
        elif self.kernel == 'unit':
            def kernel_fn(output, layer_idx=None, *args, **kwargs): 
                if layer_idx is None:
                    return {lidx: torch.ones_like(output[lidx])[[0]] for lidx in self.steerable_llm.steering_layer_indices}
                else:
                    return torch.ones_like(output)[:, [0]]
            self.kernel_fn = kernel_fn
        else:
            self.kernel_fn = lambda layer_idx=None, *args, **kwargs: {lidx: torch.tensor([1]) for lidx in 
                                self.steerable_llm.steering_layer_indices} if layer_idx is None else torch.tensor([1])
            

    def train(self, dataset: Dataset,):
        # if os.path.exists(self.save_file) and not self.override:
        #     self.loss_data = torch.load(self.save_file)
        #     return
        
        def hook_with_grad(module, input, output, layer_idx=-1):
            if output[0].requires_grad:
                activation = output[0]
            else:
                activation = output[0].detach().clone().requires_grad_(True)
            self.layer_outputs[layer_idx] = activation
            return (activation,)
        
        loss_data = []
        assert(self.training_batch_size == 1)
        
        for datum in tqdm(DataLoader(dataset, batch_size=self.training_batch_size, shuffle=False), desc='Train'):
            try:
                self.layer_outputs = {}
                get_handles = self.steerable_llm.register_steering_hooks(
                    lambda layer_idx: lambda x, y, z: hook_with_grad(x, y, z, layer_idx))
                if self.training == 'sft':
                    model_out = self.steerable_llm(input_ids=datum['matching_input_ids'],
                                                    attention_mask=datum['matching_attention_mask'],
                                                    labels=datum['matching_labels'])
                    loss = model_out.loss
                elif self.training == 'ce':
                    model_out = self.steerable_llm(input_ids=datum['prompt_input_ids'],
                                                    attention_mask=datum['prompt_attention_mask'],)
                    loss = partial_cross_entropy(model_out.logits, datum['probs'])
                elif self.training == 'negative_sft':
                    model_out = self.steerable_llm(input_ids=datum['not_matching_input_ids'],
                                                    attention_mask=datum['not_matching_attention_mask'],
                                                    labels=datum['not_matching_labels'])
                    loss = - model_out.loss
                elif self.training == 'dpo':
                    # Compute DPO loss
                    loss, _ = compute_dpo_loss(self.steerable_llm, None, {k: v for k, v in datum.items() if 'labels' not in k}, 
                                                {k: v for k, v in datum.items() if 'labels' in k}, beta=0.1)
                if 'matching_labels' in datum:
                    prompt_last_id = torch.where(datum['matching_labels'][0] != -100)[0][0] - 1
                else:
                    prompt_last_id = -1
                grads_loss = torch.autograd.grad(loss, self.layer_outputs.values(), retain_graph=True)
                grads_loss = {x: y[0, prompt_last_id, :].detach().cpu() for x, y in zip(self.layer_outputs, grads_loss)}
                layer_outputs = {x: y[0, prompt_last_id, :].detach().cpu() for x, y in self.layer_outputs.items()}
            finally:
                for handle in get_handles: handle.remove()
                
            if self.kernel == 'entk_proj_loss':
                vector = grads_loss
            else:
                vector = None
                
            loss_data.append((self.kernel_fn(output=layer_outputs, inputs={'input_ids': datum['prompt_input_ids'], 
                                                                           'attention_mask': datum['prompt_attention_mask']}, 
                                             vector=vector), grads_loss))
        
        self.loss_data = ({l: torch.stack([x[0][l] for x in loss_data]) for l in self.steerable_llm.steering_layer_indices}, 
                          {l: torch.stack([x[1][l] for x in loss_data])/len(dataset) for l in self.steerable_llm.steering_layer_indices})
        
    def steer_output_hook(self, module, input, output, inputs={}, steering_mask=None, layer_idx=-1):
        if getattr(self, '_bypass_steering', False):
            return output
        
        if len(inputs) == 0 and self.gen_input_ids is not None and self.gen_attention_mask is not None:
            inputs = {'input_ids': self.gen_input_ids, 'attention_mask': self.gen_attention_mask}
            steering_mask = self.get_steering_mask(self.gen_attention_mask)
            self.reset_steering()
        
        activation = output[0]
        kappa, loss_v = self.loss_data
        kappa = kappa[layer_idx].to(activation.device)
        loss_v = loss_v[layer_idx].to(activation.device)
        
        if self.kernel == 'entk_proj_loss':
            vector = loss_v
        else:
            vector = None
        inputs_kappa = self.kernel_fn(output=activation[:, -1, :], inputs=inputs, vector=vector, 
                                      layer_idx=layer_idx).to(activation.device)
        sim = torch.einsum('bd,Nd->bN', inputs_kappa, kappa)
        v_steer = torch.einsum('bN,Nd->bd', sim, loss_v)
        if steering_mask is not None:
            steering_mask = steering_mask.to(activation.device)
            activation[steering_mask] -= self.eta * v_steer.squeeze()
        else:
            activation -= self.eta * v_steer
        return activation,
    
    
    def _compute_grads_with_vjp(self, inputs, vector, token_id=-1):
        """Use VJP instead of jacrev for memory efficiency"""
        _, vjp_fn = vjp(lambda params: self.get_intermediate_activations(params=params, inputs=inputs)[0, token_id, :], 
                        dict(self.steerable_llm.get_steering_params()))
        vjp_result = vjp_fn(vector.flatten())[0]

        # Flatten and concatenate VJP results
        param_grads = []
        for param_name in vjp_result.keys():
            if vjp_result[param_name] is not None:
                param_grads.append(vjp_result[param_name].detach().cpu().flatten())
        
        return torch.cat(param_grads)[None, :]
            

class ContrastiveSteerer(BaseSteerer):
    def __init__(
        self,
        steerable_llm,
        eta: float = 1e-2,
        addition_transform: str = 'linear',
        training: str = 'sft',
        log_dir: str = '.',
        steer_masking: str = 'all',
        gen_masking: str = 'prompt'
    ):
        super().__init__(steerable_llm, log_dir=log_dir, steer_masking=steer_masking, gen_masking=gen_masking)
        self.layer_outputs = {}
        self.training = training
        
        self.addition_transform = addition_transform
        self.eta = eta
        self.contrastive_vector = None
        
    def get_layer_outputs_last_hook(self, module, input, output, layer_idx=-1, label_mask=None):
        activation = output[0].clone() #.detach().requires_grad_(True)
        label_mask = -2 if label_mask is None else label_mask.to(activation.device)
        self.layer_outputs[layer_idx] = activation[label_mask].reshape(
            (label_mask.sum(dim=1).shape[0], label_mask.sum(dim=1)[0], 
             activation.shape[-1])).mean(dim=1)
        # raise Exception

    @torch.no_grad()
    def train(self, dataset: Dataset,):
        matching_behavior_outputs = {lidx: [] for lidx in self.steerable_llm.steering_layer_indices}
        nonmatching_behavior_outputs = {lidx: [] for lidx in self.steerable_llm.steering_layer_indices}
        for datum in tqdm(DataLoader(dataset, batch_size=1), desc='Train'):
            try:
                # print (self.steerable_llm.tokenizer.batch_decode(datum['matching_labels']))
                get_handles = self.steerable_llm.register_steering_hooks(lambda lidx: 
                    lambda x, y, z: self.get_layer_outputs_last_hook(x, y, z, lidx, datum['matching_labels'] != -100))
                self.layer_outputs = {}
                out = self.steerable_llm(input_ids=datum['matching_input_ids'], 
                                         attention_mask=datum['matching_attention_mask'])
                for lidx, z in self.layer_outputs.items(): 
                    matching_behavior_outputs[lidx].append(z.detach().cpu())
            finally:
                for handle in get_handles: handle.remove()
            try:
                get_handles = self.steerable_llm.register_steering_hooks(lambda lidx: 
                    lambda x, y, z: self.get_layer_outputs_last_hook(x, y, z, lidx, datum['not_matching_labels'] != -100))
                self.layer_outputs = {}
                out = self.steerable_llm(input_ids=datum['not_matching_input_ids'], 
                                         attention_mask=datum['not_matching_attention_mask'])
                for lidx, z in self.layer_outputs.items(): 
                    nonmatching_behavior_outputs[lidx].append(z.detach().cpu())
            finally:
                for handle in get_handles: handle.remove()

        self.contrastive_vector = {}
        for k in matching_behavior_outputs:
            self.contrastive_vector[k] = torch.cat(matching_behavior_outputs[k]) - torch.cat(nonmatching_behavior_outputs[k])
        if self.addition_transform == 'pca':
            # 1. Center the data
            for k in self.contrastive_vector:
                X_centered = self.contrastive_vector[k] - self.contrastive_vector[k].mean(dim=0)
                _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
                self.contrastive_vector[k] = Vh[0]        # principal axis in feature space
        else:
            for k in self.contrastive_vector: 
                self.contrastive_vector[k] = self.contrastive_vector[k].mean(dim=0)
                # self.contrastive_vector[k] /= self.contrastive_vector[k].norm() # vector norm
        
    def steer_output_hook(self, module, input, output, inputs={}, steering_mask=None, layer_idx=-1):
        if len(inputs) == 0 and self.gen_input_ids is not None and self.gen_attention_mask is not None:
            inputs = {'input_ids': self.gen_input_ids, 'attention_mask': self.gen_attention_mask}
            steering_mask = self.get_steering_mask(self.gen_attention_mask)
            self.reset_steering()
        
        z = output[0]
        v_steer = self.contrastive_vector[layer_idx][None, :].to(z.device)
        if steering_mask is None: 
            steering_mask = torch.ones_like(z[:, :, 0]).bool()
        steering_mask = steering_mask.to(z.device)
        if self.addition_transform == 'linear': 
            z[steering_mask] += self.eta * v_steer
        elif self.addition_transform == 'pca': 
            z[steering_mask] += self.eta * v_steer
        elif self.addition_transform == 'piecewise': 
            piecewise_dot = torch.einsum('bj,j->b', z[steering_mask], v_steer.squeeze())
            z[steering_mask] += self.eta * (2*(piecewise_dot[:, None] > 0).float() - 1) * v_steer
        elif self.addition_transform == 'projection': 
            piecewise_dot = torch.einsum('bj,j->b', z[steering_mask], v_steer.squeeze())
            z[steering_mask] += self.eta * (piecewise_dot[:, None]/z.norm()) * v_steer
        return z,
        
       
class ReFTSteerer(BaseSteerer):
    def __init__(
        self,
        steerable_llm,
        eta: float = 1e-2,
        addition_transform: str = 'linear',
        training: str = 'sft',
        log_dir: str = '.',
        intervention_type: str = 'direct',
        training_params: DictConfig = None,
        steer_masking: str = 'all',
        gen_masking: str = 'prompt',
    ):
        super().__init__(steerable_llm, log_dir=log_dir, steer_masking=steer_masking, gen_masking=gen_masking)
        self.layer_output = None
        self.training = training
        
        self.addition_transform = addition_transform
        self.eta = eta
        self.intervention_type = intervention_type

        self.intervention = {}
        if intervention_type == 'mlp': 
            for lidx, in_dim, out_dim in zip(self.steerable_llm.steering_layer_indices, self.steerable_llm.steering_in_dims, self.steerable_llm.steering_out_dims):
                self.intervention[f'{lidx}'] = torch.nn.Sequential(torch.nn.Linear(in_dim, in_dim),
                                                        torch.nn.ReLU(),
                                                        torch.nn.Linear(in_dim, out_dim)).to(self.device)
                for param in self.intervention[f'{lidx}'].parameters():
                    param.requires_grad_(True)
                
        elif intervention_type == 'vector':
            from src.utils import ConstantIntervention
            for lidx, out_dim in zip(self.steerable_llm.steering_layer_indices, self.steerable_llm.steering_out_dims):
                self.intervention[f'{lidx}'] = ConstantIntervention(out_dim, device=self.device)
            
        self.intervention = torch.nn.ModuleDict(self.intervention)
        self.optimizer = Adam(lr=training_params.lr, params=self.intervention.parameters())
        self.training_params = training_params
        self.training = training
        
    def steer_output_hook(self, module, input, output, inputs={}, steering_mask=None, layer_idx=-1):
        if getattr(self, '_bypass_steering', False):
            return output
        
        if len(inputs) == 0 and self.gen_input_ids is not None and self.gen_attention_mask is not None:
            inputs = {'input_ids': self.gen_input_ids, 'attention_mask': self.gen_attention_mask}
            steering_mask = self.get_steering_mask(self.gen_attention_mask)
            self.reset_steering()
            
        activation = output[0] #.detach().requires_grad_(True)
        if steering_mask is not None: 
            steering_mask = steering_mask.to(activation.device)
            intervention = self.intervention[f'{layer_idx}'](activation[steering_mask].to(self.device)).to(activation.device)
            activation[steering_mask] += intervention
        else:
            intervention = self.intervention[f'{layer_idx}'](activation.to(self.device)).to(activation.device)
            activation += intervention
        return activation,
        
    def reset_steering(self):
        self.intervention.eval()
        
    def train(self, dataset: Dataset,):
        self.intervention.train()
        try:
            steer_handles = self.steerable_llm.register_steering_hooks(
                                lambda lidx: lambda x, y, z: self.steer_output_hook(x, y, z, layer_idx=lidx))
            for _ in range(self.training_params.num_epochs):
                pbar = tqdm(DataLoader(dataset, batch_size=self.training_params.batch_size))
                for datum in pbar:
                    self.optimizer.zero_grad()
                    if self.training == 'sft':
                        model_out = self.steerable_llm(input_ids=datum['matching_input_ids'],
                                                        attention_mask=datum['matching_attention_mask'],
                                                        labels=datum['matching_labels'])
                        loss = model_out.loss
                    elif self.training == 'ce':
                        model_out = self.steerable_llm(input_ids=datum['prompt_input_ids'],
                                                        attention_mask=datum['prompt_attention_mask'],)
                        loss = partial_cross_entropy(model_out.logits, datum['probs'])
                    elif self.training == 'dpo':
                        # Compute DPO loss
                        loss, _ = compute_dpo_loss(self.steerable_llm, None, {k: v for k, v in datum.items() if 'labels' not in k}, 
                                                   {k: v for k, v in datum.items() if 'labels' in k}, beta=0.1)
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_description(f'Loss: {loss.item()}')
        finally:
            for handle in steer_handles: handle.remove()
                
                