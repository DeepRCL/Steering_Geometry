import json
import torch
from torch.utils.data import Dataset, Subset
import random
import pandas as pd
from typing import Optional

options={0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R"}

def tokenize (tokenizer, inputs, outputs, system_prompt=None, template='input_output', **kwargs):
    if template == 'input_output':
        return tokenizer([f"###Input: {x}\n###Output: {y}" for x, y in zip(inputs, outputs)], 
                            return_tensors='pt', padding=True, padding_side='left').to("cuda")
    elif template == 'question_answer':
        return tokenizer([f"###Question: {x}\n###Answer: {y}" for x, y in zip(inputs, outputs)], 
                            return_tensors='pt', padding=True, padding_side='left').to("cuda")
    elif template == 'input_response':
        return tokenizer([f"Input: {x}\nResponse: {y}" for x, y in zip(inputs, outputs)], 
                            return_tensors='pt', padding=True, padding_side='left').to("cuda")
    elif template == 'concat':
        prompt_inputs = tokenizer(list(inputs), return_tensors='pt', padding=True, padding_side='left').to("cuda")
        outputs = tokenizer(list(outputs), return_tensors='pt', padding=True, padding_side='left').to("cuda")
        return {'input_ids': torch.cat((prompt_inputs['input_ids'], outputs['input_ids']), dim=1), 
                'attention_mask': torch.cat((prompt_inputs['attention_mask'], outputs['attention_mask']), dim=1)}
    elif template == 'chat':
        assert tokenizer.chat_template is not None
        chat = [{'role': 'system', 'content': system_prompt}] if system_prompt is not None else []
        if outputs[0] in ["", " "]: 
            return tokenizer.apply_chat_template([chat + [{'role': 'user', 'content': input}] for input in inputs],
                                                return_tensors='pt', padding=True, padding_side='left', 
                                                add_generation_prompt=True, return_dict=True, **kwargs).to("cuda")
        else:
            return tokenizer.apply_chat_template([chat + [{'role': 'user', 'content': input},
                                                    {'role': 'assistant', 'content': output}]
                                                    for input, output in zip(inputs, outputs)],
                                                    return_tensors='pt', padding=True, padding_side='left', 
                                                    return_dict=True, **kwargs).to("cuda")
            
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
            if testing == 'infer':
                self.data = json.load (open(f'{data_root}/test/{task}/test_dataset_ab.json', 'r'))
                prompt_inputs = tokenize (tokenizer, inputs=[datum['question'] for datum in self.data],
                                        outputs=[f"(" for _ in self.data], template=tokenizer_template, 
                                        system_prompt=system_prompt)
                token_matching_behavior = tokenizer.convert_tokens_to_ids([datum['answer_matching_behavior'][1] for datum in self.data])
                token_not_matching_behavior = tokenizer.convert_tokens_to_ids([datum['answer_not_matching_behavior'][1] for datum in self.data])
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'][:, :-2], 'prompt_attention_mask': prompt_inputs['attention_mask'][:, :-2]}
                self.labels = {'token_matching_behavior': token_matching_behavior, 'token_not_matching_behavior': token_not_matching_behavior}
            elif testing == 'generation':
                self.data = json.load (open(f'{data_root}/test/{task}/test_dataset_open_ended.json', 'r'))
                prompt_inputs = tokenize (tokenizer, inputs=[datum['question'] for datum in self.data],
                                        outputs=[f" " for _ in self.data], template=tokenizer_template, 
                                        system_prompt=system_prompt)
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
                self.labels = {}
        elif split == 'train':
            self.data = json.load (open(f'{data_root}/generate/{task}/generate_dataset.json', 'r'))
            if num_train_samples != -1:
                # dataset = Subset(dataset, indices=torch.randperm(len(dataset))[:10])
                random.shuffle(self.data)
                self.data = self.data[:num_train_samples]
            prompt_inputs = tokenize (tokenizer, inputs=[datum['question'] for datum in self.data],
                                      outputs=[f"(" for _ in self.data], template=tokenizer_template, 
                                      system_prompt=system_prompt, continue_final_message=True)
            token_matching_behavior = torch.tensor(tokenizer.convert_tokens_to_ids([datum['answer_matching_behavior'][1] 
                                                                                    for datum in self.data]))[:, None].to("cuda")
            token_not_matching_behavior = torch.tensor(tokenizer.convert_tokens_to_ids([datum['answer_not_matching_behavior'][1] 
                                                                                        for datum in self.data]))[:, None].to("cuda")
            # 
            inputs_match = tokenize(tokenizer, inputs=[datum['question'] for datum in self.data], 
                                    outputs=[datum['answer_matching_behavior'] for datum in self.data],
                                    template=tokenizer_template, system_prompt=system_prompt)
            labels_match = inputs_match['input_ids'].clone()
            labels_match[:, :prompt_inputs['input_ids'].shape[1]] = -100
            labels_match = torch.where(labels_match != token_matching_behavior, -100 * torch.ones_like(labels_match), labels_match)
            # 
            inputs_not_match = tokenize(tokenizer, inputs=[datum['question'] for datum in self.data], 
                                            outputs=[datum['answer_not_matching_behavior'] for datum in self.data],
                                            template=tokenizer_template, system_prompt=system_prompt)
            labels_not_match = inputs_not_match['input_ids'].clone()
            labels_not_match[:, :prompt_inputs['input_ids'].shape[1]] = -100
            labels_not_match = torch.where(labels_not_match != token_not_matching_behavior, -100 * torch.ones_like(labels_not_match), labels_not_match)
            self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask'], 
                            **{f'matching_{k}': v for k, v in inputs_match.items()}, **{f'not_matching_{k}': v for k, v in inputs_not_match.items()}}
            self.labels = {'matching_labels': labels_match, 'not_matching_labels': labels_not_match}
            # exit()
            
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
            if evaluation == 'infer':
                system_prompt = "You are supposed to choose one of the options."
                self.data = pd.read_csv(f'{data_root}/{task}/test_infer.csv')
                option_cols = [x for x in self.data.columns if x not in ['question', 'matching', 'not_matching']]
                prompts = []
                for i in range(len(self.data.index)):
                    options = '\n'.join([f'({col}): {self.data.loc[i, col]}' for col in option_cols])
                    prompts.append(f"{self.data.loc[i, 'question']}\n{options}")
                # prompts = [f"{self.data['question']}\n{'\n'.join(options)}" for options in zip(*[self.data[col] for col in option_cols])]
                prompt_inputs = tokenize(tokenizer, inputs=prompts, outputs=["("]*len(self.data.index), 
                                            template=tokenizer_template, system_prompt=system_prompt)
                token_matching_behavior = torch.tensor(tokenizer.convert_tokens_to_ids(list(self.data['matching']))).to("cuda")
                token_not_matching_behavior = torch.tensor(tokenizer.convert_tokens_to_ids(list(self.data['not_matching']))).to("cuda")
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
                self.labels = {'token_matching_behavior': token_matching_behavior, 'token_not_matching_behavior': token_not_matching_behavior}
            elif evaluation == 'generation':
                self.data = pd.read_csv(f'{data_root}/{task}/test.csv')
                prompt_inputs = tokenize(tokenizer, inputs=self.data['question'], outputs=[" "]*len(self.data.index), 
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
        elif split == 'train':
            self.data = pd.read_csv(f'{data_root}/{task}/train.csv')
            if num_train_samples != -1:
                # dataset = Subset(dataset, indices=torch.randperm(len(dataset))[:10])
                self.data = self.data.sample(frac=1).reset_index(drop=True)
                self.data = self.data.loc[:num_train_samples-1]
            prompt_inputs = tokenize (tokenizer, inputs=self.data['question'], outputs=[""]*len(self.data.index), 
                                      template=tokenizer_template, system_prompt=system_prompt)
            # 
            inputs_match = tokenize(tokenizer, inputs=self.data['question'], outputs=self.data['matching'],
                                    template=tokenizer_template, system_prompt=system_prompt)
            inputs_not_match = tokenize(tokenizer, inputs=self.data['question'], outputs=self.data['not_matching'],
                                        template=tokenizer_template, system_prompt=system_prompt)
            matching_response_mask = tokenizer(list(self.data['matching']), return_tensors="pt", 
                                                padding=True, padding_side='left', add_special_tokens=False)['attention_mask']
            not_matching_response_mask = tokenizer(list(self.data['not_matching']), return_tensors="pt", 
                                                    padding=True, padding_side='left', add_special_tokens=False)['attention_mask']
            labels_match = inputs_match['input_ids'].clone()
            labels_match[:, :prompt_inputs['input_ids'].shape[1]] = -100
            for i, x in enumerate(matching_response_mask.sum(dim=1)): labels_match[i, :-x] = -100
            labels_not_match = inputs_not_match['input_ids'].clone()
            labels_not_match[:, :prompt_inputs['input_ids'].shape[1]] = -100
            for i, x in enumerate(not_matching_response_mask.sum(dim=1)): labels_not_match[i, :-x] = -100
            self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask'], 
                            **{f'matching_{k}': v for k, v in inputs_match.items()}, **{f'not_matching_{k}': v for k, v in inputs_not_match.items()}}
            self.labels = {'matching_labels': labels_match, 'not_matching_labels': labels_not_match}
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in {**self.inputs, **self.labels}.items()}
    

class AxBenchDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        concept,
        concept_dir,
        data_root: str = 'data/axbench',
        training: str = 'sft',
        split: str = 'train',
        evaluation: str = 'alpaca',
        num_train_samples: int = -1,
        num_test_samples: int = 50,
        system_prompt: Optional[str] = None,
        tokenizer_template: Optional[str] = 'input_output',
        **kwargs
    ) -> None:
        super().__init__()
        tokenizer.pad_token = tokenizer.eos_token
        self.data_root = data_root
        self.concept = concept
        self.training = training
            
        if split == 'test':
            if evaluation == 'alpaca':
                self.data = pd.read_json(f'{data_root}/data/alpaca_eval.json')
                if num_train_samples != -1:
                    # dataset = Subset(dataset, indices=torch.randperm(len(dataset))[:10])
                    self.data = self.data.sample(frac=1).reset_index(drop=True)
                    self.data = self.data.loc[:num_test_samples]
                prompt_inputs = tokenize(tokenizer, inputs=self.data["instruction"], outputs=[""]*len(self.data.index), 
                                         template=tokenizer_template, system_prompt=system_prompt)
                self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
                self.labels = {}
        elif split == 'train':
            print (f'{data_root}/{concept_dir}/generate/train.parquet')
            df = pd.read_parquet(f'{data_root}/{concept_dir}/generate/train_data.parquet')
            if num_train_samples != -1:
                self.data = df.loc[(df['output_concept'] == concept)]
                self.data = self.data.sample(frac=1).reset_index(drop=True)
                self.data = self.data.loc[:num_train_samples-1]
            else:
                self.data = df
            df = df.loc[(df['category'] == 'negative')]
            df = df.sample(frac=1).reset_index(drop=True)
            df = df.loc[:len(self.data.index)-1]
            self.data['input'] = df.loc[:, "input"]
            self.data['neg_output'] = df.loc[:, "output"]
            prompt_inputs = tokenize (tokenizer, inputs=self.data['input'],
                                      outputs=[""]*len(self.data.index), 
                                      template=tokenizer_template, system_prompt=system_prompt)
            # 
            inputs_match = tokenize(tokenizer, inputs=self.data['input'],
                                    outputs=self.data['output'],
                                    template=tokenizer_template, system_prompt=system_prompt)
            matching_response_mask = tokenizer(list(self.data['output']), return_tensors="pt", 
                                                padding=True, padding_side='left', add_special_tokens=False)['attention_mask']
            labels_match = inputs_match['input_ids'].clone()
            labels_match[:, :prompt_inputs['input_ids'].shape[1]] = -100
            for i, x in enumerate(matching_response_mask.sum(dim=1)): labels_match[i, :-x] = -100
            inputs_not_match = tokenize(tokenizer, inputs=self.data['input'],
                                        outputs=self.data['neg_output'],
                                        template=tokenizer_template, system_prompt=system_prompt)
            not_matching_response_mask = tokenizer(list(self.data['neg_output']), return_tensors="pt", 
                                                padding=True, padding_side='left', add_special_tokens=False)['attention_mask']
            labels_not_match = inputs_not_match['input_ids'].clone()
            labels_not_match[:, :prompt_inputs['input_ids'].shape[1]] = -100
            for i, x in enumerate(not_matching_response_mask.sum(dim=1)): labels_not_match[i, :-x] = -100
            self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 
                           'prompt_attention_mask': prompt_inputs['attention_mask'], 
                            **{f'matching_{k}': v for k, v in inputs_match.items()},
                            **{f'not_matching_{k}': v for k, v in inputs_not_match.items()}}
            self.labels = {'matching_labels': labels_match, 'not_matching_labels': labels_not_match}
            
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
        binarize: type = False,
        **kwargs
    ) -> None:
        super().__init__()
        tokenizer.pad_token = tokenizer.eos_token
        self.data_root = data_root
        self.demographic_group = demographic_group
        self.wave = wave
        self.training = training
        self.binarize = binarize
        
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
            self.data = self.data.loc[split_point:]
            prompt_inputs = tokenize(tokenizer, inputs=self.data['question'], outputs=[""]*len(self.data.index), 
                                        template=tokenizer_template, system_prompt=system_prompt)
            self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask']}
            if self.binarize:
                self.labels = {'token_matching_behavior': torch.tensor([token_ids[x] for x in self.data[option_cols].idxmax(axis=1)]).to("cuda")}
            else:
                self.labels = {token_ids[x]: torch.tensor(list(self.data[x])).to("cuda") for x in option_cols}
        elif split == 'train':
            self.data = self.data.loc[:split_point]
            if num_train_samples != -1:
                self.data = self.data.sample(frac=1).reset_index(drop=True)
                # dataset = Subset(dataset, indices=torch.randperm(len(dataset))[:10])
                self.data = self.data.loc[:num_train_samples-1]
            prompt_inputs = tokenize (tokenizer, inputs=self.data['question'], outputs=[""]*len(self.data.index), 
                                      template=tokenizer_template, system_prompt=system_prompt)
            # 
            self.inputs = {'prompt_input_ids': prompt_inputs['input_ids'], 'prompt_attention_mask': prompt_inputs['attention_mask'],}
            if self.binarize:
                self.labels = {'token_matching_behavior': torch.tensor([token_ids[x] for x in self.data[option_cols].idxmax(axis=1)]).to("cuda")}
            else:
                self.labels = {token_ids[x]: torch.tensor(list(self.data[x])).to("cuda") for x in option_cols}
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.binarize:
            return {**{k: v[idx] for k, v in self.labels.items()}, **{k: v[idx] for k, v in self.inputs.items()}}
        else:
            return {'probs': {k: v[idx] for k, v in self.labels.items()}, 
                    **{k: v[idx] for k, v in self.inputs.items()}}
    
