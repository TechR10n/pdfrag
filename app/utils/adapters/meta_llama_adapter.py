import os
import logging
import time
import threading
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class GenerationTimeoutError(Exception):
    """Exception raised when text generation times out."""
    pass

class MetaLlamaAdapter:
    def __init__(self, model_path: str, max_new_tokens: int = 512, generation_timeout: int = 120):
        """Initialize the Meta Llama adapter."""
        logger.info(f"Loading Meta Llama model from {model_path}")
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.generation_timeout = generation_timeout  # Timeout in seconds
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check GPU availability and set up appropriate configuration
        import torch
        
        # Set default model loading kwargs
        model_kwargs = {
            "low_cpu_mem_usage": True,
        }
        
        # Set pipeline kwargs
        pipe_kwargs = {}
        
        if torch.cuda.is_available():
            logger.info("Using CUDA for model inference")
            device_map = "auto"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs.update({
                "device_map": device_map,
                "torch_dtype": torch_dtype,
            })
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using MPS (Metal) for model inference")
            # For Apple Silicon, use more optimized settings
            device_map = "mps" 
            torch_dtype = torch.float16
            
            # Set torch to use MPS (Metal Performance Shaders)
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use less GPU memory
            
            model_kwargs.update({
                "device_map": device_map,
                "torch_dtype": torch_dtype,
            })
            
            # Enable memory efficient attention
            if torch.__version__ >= "2.0.0":
                pipe_kwargs["use_cache"] = True

        else:
            logger.info("Using CPU for model inference - this will be slow")
            # For CPU, enable 4-bit quantization if possible to save memory
            try:
                from transformers import BitsAndBytesConfig
                
                model_kwargs.update({
                    "device_map": "auto",
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    ),
                })
                logger.info("Enabled 4-bit quantization for CPU inference")
            except ImportError:
                model_kwargs.update({
                    "device_map": "auto",
                    "torch_dtype": torch.float32,
                })
                logger.warning("Quantization not available - model will use full precision")
        
        # Set max length parameter
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For MPS (Metal), limit context size
            pipe_kwargs["max_length"] = 2048
        
        # Load model with performance settings
        logger.info(f"Loading model with settings: {model_kwargs}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                **model_kwargs
            )
            
            # Create pipeline with appropriate settings
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **pipe_kwargs
            )
            
            logger.info("Meta Llama model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _generate_with_timeout(self, prompt, generation_kwargs):
        """Generate text with a timeout."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = self.pipe(prompt, **generation_kwargs)
            except Exception as e:
                exception[0] = e
                logger.error(f"Generation error: {str(e)}")
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        
        start_time = time.time()
        thread.start()
        thread.join(self.generation_timeout)
        
        if thread.is_alive():
            logger.warning(f"Generation timed out after {self.generation_timeout} seconds")
            raise GenerationTimeoutError(f"Text generation timed out after {self.generation_timeout} seconds")
        
        if exception[0]:
            raise exception[0]
            
        logger.info(f"Generation completed in {time.time() - start_time:.2f} seconds")
        return result[0]
    
    def __call__(self, prompt: str, **kwargs):
        """Generate text using the Meta Llama model."""
        generation_kwargs = {
            "max_new_tokens": min(kwargs.get("max_tokens", self.max_new_tokens), 256),  # Limit token generation
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.2) > 0,
        }
        
        logger.info(f"Generating with parameters: {generation_kwargs}")
        
        # Generate text with timeout
        try:
            outputs = self._generate_with_timeout(prompt, generation_kwargs)
        except GenerationTimeoutError:
            # Return a partial response if timed out
            return {
                "choices": [
                    {
                        "text": "[Generation timed out. The model is taking too long to respond.]",
                        "finish_reason": "timeout",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(prompt)),
                    "completion_tokens": 0,
                    "total_tokens": len(self.tokenizer.encode(prompt)),
                }
            }
        
        # Format to match llama-cpp-python output format
        try:
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
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return {
                "choices": [
                    {
                        "text": f"[Error processing model response: {str(e)}]",
                        "finish_reason": "error",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(prompt)),
                    "completion_tokens": 0,
                    "total_tokens": len(self.tokenizer.encode(prompt)),
                }
            }
