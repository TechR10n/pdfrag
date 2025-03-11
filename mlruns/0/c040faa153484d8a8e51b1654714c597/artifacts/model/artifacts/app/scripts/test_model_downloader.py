#!/usr/bin/env python
"""
Test script to verify model loading and downloading functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from app.config.settings import MODEL_PATH, ALT_MODEL_PATHS, HF_MODEL_ID, HF_TOKEN
from app.utils.model_downloader import find_or_download_model
from app.utils.llm import LLMProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test the model loading/downloading functionality"""
    
    logger.info("Testing model loading/downloading functionality")
    logger.info(f"Primary model path: {MODEL_PATH}")
    logger.info(f"Alternative model paths: {ALT_MODEL_PATHS}")
    logger.info(f"HF Model ID: {HF_MODEL_ID}")
    logger.info(f"HF Token available: {'Yes' if HF_TOKEN else 'No'}")
    
    # First test direct find_or_download function
    model_path = find_or_download_model(
        MODEL_PATH,
        ALT_MODEL_PATHS,
        HF_MODEL_ID,
        HF_TOKEN
    )
    
    logger.info(f"Found or downloaded model at: {model_path}")
    
    if model_path:
        exists = os.path.exists(model_path)
        size = os.path.getsize(model_path) if exists else 0
        logger.info(f"Model exists: {exists}, Size: {size/1024/1024:.2f} MB")
    
    # Now test through the LLMProcessor
    logger.info("Testing model loading through LLMProcessor")
    
    processor = LLMProcessor(MODEL_PATH)
    
    if processor.use_mock:
        logger.warning("Using mock processor - real model not available or invalid")
    else:
        logger.info("Successfully loaded real LLM model!")
        
        # Test with a simple query
        prompt = "Tell me a short fact about language models."
        logger.info(f"Testing with prompt: {prompt}")
        
        response = processor.generate_response(prompt)
        
        logger.info(f"Response: {response['text']}")
        logger.info(f"Tokens used: {response['metadata']['tokens_used']}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    test_model_loading() 