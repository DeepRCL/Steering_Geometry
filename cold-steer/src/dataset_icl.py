import json
import torch
from torch.utils.data import Dataset, Subset
import random
import pandas as pd
from typing import Optional

options={0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R"}

def tokenize_icl (tokenizer, questions, example_qs, example_as, system_prompt=None, template='input_response', **kwargs):
    if template == 'input_response':
        input_examples = '\n\n'.join([f'Input: {q}\nResponse:{a}' for q, a in zip(example_qs, example_as)])
        return tokenizer([f"Given {input_examples}\n\n Input: {question}\nResponse: " for question in questions],
                         return_tensors='pt', padding=True, padding_side='left').to("cuda")
    elif template == 'chat':
        chat = [{'role': 'system', 'content': system_prompt}] if system_prompt is not None else []
        input_examples = '\n\n'.join([f'Input: {q}\nResponse:{a}' for q, a in zip(example_qs, example_as)])
        question_fn = lambda x: f"Given {input_examples}\n\n Input: {x}\nResponse: "
        return tokenizer.apply_chat_template([chat + [{'role': 'user', 'content': question_fn(question)}] 
                                              for question in questions], return_tensors='pt', padding=True, 
                                             padding_side='left', add_generation_prompt=True, 
                                             return_dict=True, **kwargs).to("cuda")


NUM_SAMPLES = 10

class CAADataset(Dataset):
    def __init__(
        self,
        task: str,
        tokenizer,
        data_root: str = 'data/CAA/',
        training: str = 'sft',
        split: str = 'train',
        testing: str = 'infer',
        num_train_samples: int = -1,
        system_prompt: Optional[str] = None,
        tokenizer_template: Optional[str] = 'input_output',
    ) -> None:
        super().__init__()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        self.data_root = data_root
        self.task = task
        self.training = training
            
        if split == 'test':
            training_data = json.load (open(f'{data_root}/generate/{task}/generate_dataset.json', 'r'))
            random.shuffle(training_data)
            training_data = training_data[:NUM_SAMPLES]
            if testing == 'infer':
                self.data = json.load (open(f'{data_root}/test/{task}/test_dataset_ab.json', 'r'))                                       
                prompt_inputs = tokenize_icl (tokenizer, questions=[datum['question'] for datum in self.data],
                                                example_qs=[x['question'] for x in training_data], 
                                                example_as=[x['answer_matching_behavior'] for x in training_data], 
                                                template=tokenizer_template, system_prompt=system_prompt)
                token_matching_behavior = tokenizer.convert_tokens_to_ids([datum['answer_matching_behavior'][1] for datum in self.data])
                token_not_matching_behavior = tokenizer.convert_tokens_to_ids([datum['answer_not_matching_behavior'][1] for datum in self.data])
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'][:, :-2], 'prompt_attention_mask': prompt_inputs['attention_mask'][:, :-2]}
                self.labels = {'token_matching_behavior': token_matching_behavior, 'token_not_matching_behavior': token_not_matching_behavior}
            elif testing == 'generation':
                self.data = json.load (open(f'{data_root}/test/{task}/test_dataset_open_ended.json', 'r'))
                prompt_inputs = tokenize_icl (tokenizer, questions=[datum['question'] for datum in self.data],
                                                example_qs=[x['question'] for x in training_data], 
                                                example_as=[x['answer_matching_behavior'] for x in training_data], 
                                                template=tokenizer_template, system_prompt=system_prompt)
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
                self.labels = {}
                
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in {**self.inputs, **self.labels}.items()}
    

   
class BiPODataset(Dataset):
    def __init__(
        self,
        task: str,
        tokenizer,
        data_root: str = 'data/BiPO/',
        training: str = 'sft',
        split: str = 'train',
        evaluation: str = 'infer',
        num_train_samples: int = -1,
        system_prompt: Optional[str] = None,
        tokenizer_template: Optional[str] = 'input_output',
    ) -> None:
        super().__init__()
        tokenizer.pad_token = tokenizer.eos_token
        self.data_root = data_root
        self.task = task
        self.training = training
        
            
        if split == 'test':
            training_data = pd.read_csv(f'{data_root}/{task}/train.csv')
            if num_train_samples != -1:
                # dataset = Subset(dataset, indices=torch.randperm(len(dataset))[:10])
                training_data = training_data.sample(frac=1).reset_index(drop=True)
                training_data = training_data.loc[:num_train_samples-1]
            if evaluation == 'infer':
                system_prompt = "You are supposed to choose one of the options."
                self.data = pd.read_csv(f'{data_root}/{task}/test_infer.csv')
                option_cols = [x for x in self.data.columns if x not in ['question', 'matching', 'not_matching']]
                prompts = []
                for i in range(len(self.data.index)):
                    options = '\n'.join([f'({col}): {self.data.loc[i, col]}' for col in option_cols])
                    prompts.append(f"{self.data.loc[i, 'question']}\n{options}")
                # prompts = [f"{self.data['question']}\n{'\n'.join(options)}" for options in zip(*[self.data[col] for col in option_cols])]
                prompt_inputs = tokenize_icl (tokenizer, questions=self.data['question'],
                                                example_qs=training_data['question'], 
                                                example_as=training_data['matching'],
                                                template=tokenizer_template, system_prompt=system_prompt)
                token_matching_behavior = torch.tensor(tokenizer.convert_tokens_to_ids(list(self.data['matching']))).to("cuda")
                token_not_matching_behavior = torch.tensor(tokenizer.convert_tokens_to_ids(list(self.data['not_matching']))).to("cuda")
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
                self.labels = {'token_matching_behavior': token_matching_behavior, 'token_not_matching_behavior': token_not_matching_behavior}
            elif evaluation == 'generation':
                self.data = pd.read_csv(f'{data_root}/{task}/test.csv')
                prompt_inputs = tokenize_icl (tokenizer, questions=self.data['question'],
                                                example_qs=training_data['question'], 
                                                example_as=training_data['matching'],
                                                template=tokenizer_template, system_prompt=system_prompt)
                response_matching_behavior = tokenizer(list(self.data['matching']), return_tensors="pt", 
                                                        padding=True, padding_side='left', add_special_tokens=False).to("cuda")
                response_not_matching_behavior = tokenizer(list(self.data['not_matching']), return_tensors="pt", 
                                                            padding=True, padding_side='left', add_special_tokens=False).to("cuda")
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
                self.labels = {'response_input_ids_matching_behavior': response_matching_behavior['input_ids'], 
                                'response_attention_mask_matching_behavior': response_matching_behavior['attention_mask'], 
                                'response_input_ids_not_matching_behavior': response_not_matching_behavior['input_ids'],
                                'response_attention_mask_not_matching_behavior': response_not_matching_behavior['attention_mask']}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in {**self.inputs, **self.labels}.items()}
    

class OpinionsQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        wave,
        demographic_group,
        demographic,
        data_root: str = 'data/distralign/opinions_qa',
        training: str = 'ce',
        split: str = 'train',
        test_size: float = 0.3,
        num_train_samples: int = -1,
        tokenizer_template: Optional[str] = 'input_response',
        **kwargs
    ) -> None:
        super().__init__()
        tokenizer.pad_token = tokenizer.eos_token
        self.data_root = data_root
        self.demographic_group = demographic_group
        self.wave = wave
        self.training = training
        
        system_prompt = 'Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. '
        data = json.load(open(f'{data_root}/data/human_resp/{wave}/{demographic_group}_data.json', 'r'))
        df_records = []
        for i, q_ID in enumerate(data):
            records = {'question': data[q_ID]['question_text']}
            MC_options = list(data[q_ID][demographic].keys())
            n = (sum(data[q_ID][demographic].values()))
            records['question'] += '\n' + '\n'.join([f'{options[i]}: {option_text}' for i, option_text in enumerate(data[q_ID]['MC_options'])])
            for j, MC_option in enumerate(MC_options):
                records[options[j]] = data[q_ID][demographic][MC_option]/n
            df_records.append(records)
        self.data = pd.DataFrame(df_records).fillna(-1)
        option_cols = [x for x in self.data.columns if x != 'question']
        token_ids = dict(zip(option_cols, tokenizer.convert_tokens_to_ids(option_cols)))

        split_point = int((1-test_size) * len(self.data.index))
        if split == 'test':
            training_data = training_data.loc[:split_point]
            if num_train_samples != -1:
                training_data = training_data.sample(frac=1).reset_index(drop=True)
                # dataset = Subset(dataset, indices=torch.randperm(len(dataset))[:10])
                training_data = training_data.loc[:num_train_samples-1]
            self.data = self.data.loc[split_point:]
            prompt_inputs = tokenize_icl(tokenizer, questions=self.data['question'], 
                                         example_qs=training_data['question'], 
                                         example_as=training_data['matching'],
                                         template=tokenizer_template, system_prompt=system_prompt)
            self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
            self.labels = {token_ids[x]: torch.tensor(list(self.data[x])).to("cuda") for x in option_cols}
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'probs': {k: v[idx] for k, v in self.labels.items()}, 
                **{k: v[idx] for k, v in self.inputs.items()}}
    
