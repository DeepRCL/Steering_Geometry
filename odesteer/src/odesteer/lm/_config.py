_FULL_LLM_NAMES = {
    'TinyLlama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    # Llama2
    'Llama2-7B-Base': 'meta-llama/Llama-2-7b-hf',
    'Llama2-7B-Chat': 'meta-llama/Llama-2-7b-chat-hf',
    # Llama3
    'Llama3-8B-Base': 'meta-llama/Meta-Llama-3-8B',
    'Llama3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    # Llama3.1
    'Llama3.1-8B-Base': 'meta-llama/Llama-3.1-8B',
    'Llama3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    # Qwen
    'Qwen2-7B-Base': 'Qwen/Qwen2-7B',
    'Qwen2.5-7B-Base': 'Qwen/Qwen2.5-7B',
    'Qwen3-8B-Base': 'Qwen/Qwen3-8B-Base', 
    # Falcon
    'Falcon-7B-Base': 'tiiuae/falcon-7b',
    # Phi3
    'Phi3-7B-Instruct': 'microsoft/Phi-3-small-8k-instruct',
    # Pythia
    'Pythia-7B-Base': 'EleutherAI/pythia-6.9b',
    # Gemma
    'Gemma-7B-Base': 'google/gemma-7b',
    'Gemma2-9B-Base': 'google/gemma-2-9b',
    # Mistral
    'Mistral-7B-Base': 'mistralai/Mistral-7B-v0.3',
    # Others
    'Vicuna': 'lmsys/vicuna-7b-v1.5',
    'RawVicuna': 'AlekseyKorshuk/vicuna-7b',
    'Alpaca': 'chavinlo/alpaca-native',
}

_DEFAULT_SAMPLING_PARAMS = {
    'temperature': 0.0,
    'top_p': 1.0,
    'seed': 42,
    'top_k': -1,
    'max_tokens': 64,
    'repetition_penalty': 1.00,
    'no_repeat_ngram_size': 0,
}

_DEFAULT_CHAT_TEMPLATE =  (
    "{{- bos_token -}}"
    "{%- set default_system = '' -%}"
    "{%- if messages and messages[0]['role'] == 'system' -%}"
    "{{- messages[0]['content'] -}}"
    "{%- set idx = 1 -%}"
    "{%- else -%}"
    "{{- default_system -}}"
    "{%- set idx = 0 -%}"
    "{%- endif -%}"
    "{%- for message in messages[idx:] -%}"
    "{%- if message['role'] == 'user' -%}"
    "{{ '\\n' }}Q: {{ message['content'] }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{ '\\n' }}A: {{ message['content'] }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '\\n' }}A:"
    "{%- else -%}"
    "{{ eos_token }}"
    "{%- endif -%}"
)