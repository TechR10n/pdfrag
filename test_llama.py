#!/usr/bin/env python3

import logging
import sys
import time
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import necessary modules
from app.config.settings import BASE_DIR, HF_TOKEN, HF_MODEL_ID, MODEL_PATH
from app.utils.model_downloader import find_or_download_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llama_model():
    """Test the Llama-3.2-1B-Instruct model with a simple prompt."""
    logger.info("===== Starting Llama-3.2-1B-Instruct simple test =====")
    
    # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Please download it first.")
        return False
    
    # Try to import transformers
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        logger.error("transformers not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
    # Load the model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        start_time = time.time()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Generate text
        prompt = "What is retrieval-augmented generation?"
        logger.info(f"Generating text for prompt: {prompt}")
        
        start_time = time.time()
        outputs = pipe(
            prompt,
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Text generated in {generation_time:.2f} seconds")
        
        # Print the generated text
        generated_text = outputs[0]["generated_text"]
        logger.info(f"Generated text: {generated_text}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_llama_model()
    if success:
        logger.info("===== Completed Llama-3.2-1B-Instruct simple test =====")
    else:
        logger.error("===== Failed Llama-3.2-1B-Instruct simple test =====") 