import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
import torch

def download_embedding_model():
    """Download the embedding model for offline use."""
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"Downloading embedding model to {EMBEDDING_MODEL_PATH}...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        os.makedirs(os.path.dirname(EMBEDDING_MODEL_PATH), exist_ok=True)
        model.save(EMBEDDING_MODEL_PATH)
        print("Embedding model downloaded successfully.")
        
        # Test the model
        test_embedding = model.encode(["Hello world"])
        print(f"Test embedding shape: {test_embedding.shape}")
        
    except Exception as e:
        print(f"Error downloading embedding model: {str(e)}")
        sys.exit(1)

def download_reranker_model():
    """Download the reranker model for offline use."""
    try:
        from sentence_transformers import CrossEncoder
        
        print(f"Downloading reranker model to {RERANKER_MODEL_PATH}...")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        os.makedirs(os.path.dirname(RERANKER_MODEL_PATH), exist_ok=True)
        model.save(RERANKER_MODEL_PATH)
        print("Reranker model downloaded successfully.")
        
    except Exception as e:
        print(f"Error downloading reranker model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_embedding_model()
    download_reranker_model()
    print("Please manually download the LLM model using the instructions in the README.")