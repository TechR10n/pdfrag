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
    """Download the Llama-3.2-1B-Instruct model from Hugging Face."""
    # Get the Hugging Face token from environment
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set. Please set it before running this script.")
        return False
    
    # Model details
    repo_id = "meta-llama/Llama-3.2-1B-Instruct"
    local_dir = "models/llm/Llama-3.2-1B-Instruct"
    
    try:
        logger.info(f"Starting download of {repo_id}...")
        
        # Download model files
        model_path = snapshot_download(
            repo_id=repo_id,
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
        logger.info("Model download complete. You can now use the Llama-3.2-1B-Instruct model.")
    else:
        logger.error("Failed to download the model. Please check the logs for details.") 