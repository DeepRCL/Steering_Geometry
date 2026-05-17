#!/bin/bash

# format dataset
uv run python -u data/ultrafeedback/format_dataset.py

# extract activations
uv run python -u data/ultrafeedback/extract_activations.py -m Falcon-7B-Base -l 14 -b 10
uv run python -u data/ultrafeedback/extract_activations.py -m Mistral-7B-Base -l 15 -b 10
uv run python -u data/ultrafeedback/extract_activations.py -m Llama3.1-8B-Base -l 13 -b 10
uv run python -u data/ultrafeedback/extract_activations.py -m Qwen2.5-7B-Base -l 13 -b 10
