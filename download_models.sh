#!/bin/bash

# Create model directories
mkdir -p models/llm
mkdir -p models/embedding
mkdir -p models/reranker

# Set default model IDs if not provided in environment
HF_EMBEDDING_MODEL_ID=${HF_EMBEDDING_MODEL_ID:-"sentence-transformers/all-MiniLM-L6-v2"}
HF_RERANKER_MODEL_ID=${HF_RERANKER_MODEL_ID:-"cross-encoder/ms-marco-MiniLM-L-6-v2"}

# Extract model names for local directories
EMBEDDING_MODEL_NAME=$(echo $HF_EMBEDDING_MODEL_ID | awk -F/ '{print $NF}')
RERANKER_MODEL_NAME=$(echo $HF_RERANKER_MODEL_ID | awk -F/ '{print $NF}')

# Download embedding model
echo "Downloading embedding model ($HF_EMBEDDING_MODEL_ID)..."
python -c "
from huggingface_hub import snapshot_download
import os

# Get model ID from environment
repo_id = os.environ.get('HF_EMBEDDING_MODEL_ID', 'sentence-transformers/all-MiniLM-L6-v2')
model_name = repo_id.split('/')[-1]
local_dir = f'models/embedding/{model_name}'

# Download model files
model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f'Embedding model downloaded to {model_path}')
"

# Download reranker model
echo "Downloading reranker model ($HF_RERANKER_MODEL_ID)..."
python -c "
from huggingface_hub import snapshot_download
import os

# Get model ID from environment
repo_id = os.environ.get('HF_RERANKER_MODEL_ID', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
model_name = repo_id.split('/')[-1]
local_dir = f'models/reranker/{model_name}'

# Download model files
model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f'Reranker model downloaded to {model_path}')
"

# Download Llama model
echo "Downloading LLM model (${HF_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct})..."
echo "This requires a Hugging Face token with access to the model."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):"
  read -s HF_TOKEN
  echo
fi

# Run the Python script to download the LLM model
python download_llama.py

echo "All models downloaded successfully!" 