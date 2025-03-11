#!/bin/bash

# Create model directories
mkdir -p models/llm
mkdir -p models/embedding
mkdir -p models/reranker

# Download embedding model
echo "Downloading embedding model (all-MiniLM-L6-v2)..."
python -c "
from huggingface_hub import snapshot_download
import os

# Download model files
model_path = snapshot_download(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    local_dir='models/embedding/all-MiniLM-L6-v2',
    local_dir_use_symlinks=False
)

print(f'Embedding model downloaded to {model_path}')
"

# Download reranker model
echo "Downloading reranker model (ms-marco-MiniLM-L-6-v2)..."
python -c "
from huggingface_hub import snapshot_download
import os

# Download model files
model_path = snapshot_download(
    repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2',
    local_dir='models/reranker/ms-marco-MiniLM-L-6-v2',
    local_dir_use_symlinks=False
)

print(f'Reranker model downloaded to {model_path}')
"

# Download Llama model
echo "Downloading Llama-3.2-1B-Instruct model..."
echo "This requires a Hugging Face token with access to the Meta Llama models."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):"
  read -s HF_TOKEN
  echo
fi

# Run the Python script to download the Llama model
python download_llama.py

echo "All models downloaded successfully!" 