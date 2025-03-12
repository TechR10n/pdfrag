#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import necessary modules
from app.config.settings import BASE_DIR, HF_TOKEN, HF_MODEL_ID
from app.utils.model_downloader import download_huggingface_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_llama_model():
    """Download the Llama model from Hugging Face."""
    # Get the Hugging Face token from environment
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set. Please set it before running this script.")
        return False
    
    # Get model ID from environment or use the default from settings
    model_id = os.environ.get("HF_MODEL_ID", HF_MODEL_ID)
    
    # Extract model name for the local directory
    model_name = model_id.split('/')[-1]
    local_dir = f"models/llm/{model_name}"
    
    try:
        logger.info(f"Starting download of {model_id}...")
        
        # Download model files
        model_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        
        logger.info(f"Model downloaded successfully to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    # Try to import huggingface_hub
    try:
        import huggingface_hub
    except ImportError:
        logger.error("huggingface_hub not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    
    success = download_llama_model()
    if success:
        # Get the model name from the environment or settings
        model_id = os.environ.get("HF_MODEL_ID", HF_MODEL_ID)
        model_name = model_id.split('/')[-1]
        logger.info(f"Model download complete. You can now use the {model_name} model.")
    else:
        logger.error("Failed to download the model. Please check the logs for details.") 