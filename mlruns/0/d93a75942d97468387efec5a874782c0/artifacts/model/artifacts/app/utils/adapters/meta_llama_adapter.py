import os
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class MetaLlamaAdapter:
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        """Initialize the Meta Llama adapter."""
        logger.info(f"Loading Meta Llama model from {model_path}")
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        logger.info("Meta Llama model loaded successfully")
    
    def __call__(self, prompt: str, **kwargs):
        """Generate text using the Meta Llama model."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.2) > 0,
        }
        
        # Generate text
        outputs = self.pipe(
            prompt,
            **generation_kwargs
        )
        
        # Format to match llama-cpp-python output format
        generated_text = outputs[0]["generated_text"][len(prompt):]
        
        # Count tokens
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(generated_text))
        
        return {
            "choices": [
                {
                    "text": generated_text,
                    "finish_reason": "length" if output_tokens >= generation_kwargs["max_new_tokens"] else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
