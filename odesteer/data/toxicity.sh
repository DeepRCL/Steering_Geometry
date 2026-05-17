#!/bin/bash

# training data 
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
unzip jigsaw-unintended-bias-in-toxicity-classification.zip -d data/toxicity/jigsaw-unintended-bias-in-toxicity-classification
rm jigsaw-unintended-bias-in-toxicity-classification.zip

# create new dir
mkdir -p data/toxicity/jigsaw
mv data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/* data/toxicity/jigsaw
rmdir data/toxicity/jigsaw-unintended-bias-in-toxicity-classification

# preprocess data
uv run python -u data/toxicity/preprocess_jigsaw.py
uv run python -u data/toxicity/preprocess_rtp.py
uv run python -u data/toxicity/format_dataset.py

# extract activations
uv run python -u data/toxicity/extract_activations.py -m Falcon-7B-Base -l 14 -b 10
uv run python -u data/toxicity/extract_activations.py -m Mistral-7B-Base -l 15 -b 10
uv run python -u data/toxicity/extract_activations.py -m Llama3.1-8B-Base -l 13 -b 10
uv run python -u data/toxicity/extract_activations.py -m Qwen2.5-7B-Base -l 13 -b 10