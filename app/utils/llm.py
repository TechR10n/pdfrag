import os
from typing import Dict, Any, List, Optional
import logging
import sys
from pathlib import Path

# Import settings and model utilities
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.config.settings import ALT_MODEL_PATHS, HF_MODEL_ID, HF_TOKEN
from app.utils.model_downloader import find_or_download_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available, model loading will be limited to transformers")

# Try to import transformers for non-GGUF models
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, model loading will be limited to llama-cpp")

# Try to import MetaLlamaAdapter
try:
    from app.utils.adapters.meta_llama_adapter import MetaLlamaAdapter
    META_LLAMA_ADAPTER_AVAILABLE = True
except ImportError:
    logger.warning("MetaLlamaAdapter not available")
    META_LLAMA_ADAPTER_AVAILABLE = False

class TransformersLLMProcessor:
    """LLM processor implementation using HuggingFace Transformers."""
    
    def __init__(self, model_path: str, context_size: int = 2048, max_tokens: int = 512):
        """
        Initialize the Transformers LLM processor.
        
        Args:
            model_path: Path to the transformer model
            context_size: Context size for the model
            max_tokens: Maximum number of tokens to generate
        """
        logger.info(f"Loading Transformers model from {model_path}")
        
        try:
            # Determine if model_path is a directory with model files
            if os.path.isdir(model_path):
                # For Llama 3.2, we need to modify the config file to fix rope_scaling
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Fix rope_scaling if it exists
                    if 'rope_scaling' in config:
                        logger.info("Fixing rope_scaling in config.json")
                        # Set to a valid format
                        config['rope_scaling'] = {"type": "dynamic", "factor": 2.0}
                        
                        # Save the modified config
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)
                
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Determine device
                if torch.cuda.is_available():
                    logger.info("Using CUDA for model inference")
                    device = "cuda"
                elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("Using MPS (Metal) for model inference")
                    device = "mps"
                else:
                    logger.info("Using CPU for model inference")
                    device = "cpu"
                
                # Use pipeline directly with text-generation task
                self.pipe = pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16,
                    device_map=device,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                    },
                    trust_remote_code=True,
                )
                
                logger.info("Transformers model loaded successfully")
            else:
                raise ValueError(f"Model path {model_path} is not a directory with model files")
            
        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            raise
            
        self.max_tokens = max_tokens
        self.context_size = context_size
    
    def create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for Llama 3.
        """
        # Format context
        context_text = ""
        for i, doc in enumerate(context):
            context_text += f"Document {i+1}:\n{doc['chunk_text']}\n\n"
        
        # Llama 3.x prompt format
        prompt = f"""<|system|>
You are a helpful AI assistant that provides accurate and concise answers based on the provided context documents. 
If the answer is not contained in the documents, say "I don't have enough information to answer this question."
Do not make up or hallucinate any information that is not supported by the documents.
</|system|>

<|user|>
I need information about the following topic:

{query}

Here are relevant documents to help answer this question:

{context_text}
</|user|>

<|assistant|>
"""
        return prompt
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the Transformers model.
        
        Args:
            prompt: Prompt for the model
            
        Returns:
            Response with text and metadata
        """
        logger.info("Generating response with Transformers model")
        
        # Generate the response
        # Truncate input if it's too long
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.context_size)
        prompt_tokens = len(inputs.input_ids[0])
        
        # Move inputs to the same device as the model
        if hasattr(self.pipe, 'device'):
            inputs = {k: v.to(self.pipe.device) for k, v in inputs.items()}
        
        # Generate the text
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            return_full_text=False  # Only return the newly generated text
        )
        
        # Get the generated text
        response_text = outputs[0]['generated_text']
        
        # Get metadata (estimated)
        response_tokens = len(self.tokenizer(response_text, return_tensors="pt").input_ids[0])
        
        metadata = {
            'tokens_used': prompt_tokens + response_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': response_tokens,
        }
        
        logger.info(f"Generated response with {metadata['completion_tokens']} tokens")
        
        return {
            'text': response_text,
            'metadata': metadata
        }

class LLMProcessor:
    def __init__(self, model_path: str, context_size: int = 2048, max_tokens: int = 512):
        """
        Initialize the LLM processor.
        
        Args:
            model_path: Path to the LLM model
            context_size: Context size for the model
            max_tokens: Maximum number of tokens to generate
        """
        logger.info(f"Loading LLM from {model_path}")
        
        # Try to find the model in various locations or download it
        actual_model_path = find_or_download_model(
            model_path,
            ALT_MODEL_PATHS,
            HF_MODEL_ID,
            HF_TOKEN
        )
        
        # Default values
        self.model = None
        self.use_transformers = False
        self.use_meta_adapter = False
        
        # Check if it's a Llama-3.2 Instruct model (either 1B or 3B)
        is_llama_3_2_instruct = any(model_name in actual_model_path or model_name in os.path.dirname(actual_model_path) 
                                   for model_name in ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"])
        
        if is_llama_3_2_instruct and META_LLAMA_ADAPTER_AVAILABLE and TRANSFORMERS_AVAILABLE:
            
            # Determine the actual model directory
            if os.path.isdir(actual_model_path):
                model_dir = actual_model_path
            else:
                model_dir = os.path.dirname(actual_model_path)
                # Check for either 1B or 3B model directories
                for model_name in ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]:
                    if model_name in os.listdir(model_dir):
                        model_dir = os.path.join(model_dir, model_name)
                        break
            
            # Get the model name from the directory path
            model_name = os.path.basename(model_dir)
            
            try:
                logger.info(f"Loading {model_name} with MetaLlamaAdapter from {model_dir}")
                self.meta_adapter = MetaLlamaAdapter(
                    model_path=model_dir,
                    max_new_tokens=max_tokens
                )
                self.use_meta_adapter = True
                logger.info(f"Loaded {model_name} with MetaLlamaAdapter")
            except Exception as e:
                logger.error(f"Failed to load {model_name} with MetaLlamaAdapter: {e}")
                # Fall back to TransformersLLMProcessor
                try:
                    self.transformers_processor = TransformersLLMProcessor(
                        model_path=model_dir,
                        context_size=context_size,
                        max_tokens=max_tokens
                    )
                    self.use_transformers = True
                    logger.info(f"Loaded {model_name} with TransformersLLMProcessor as fallback")
                except Exception as e2:
                    logger.error(f"Failed to load with TransformersLLMProcessor: {e2}")
        
        # Case 1: Valid GGUF file and llama-cpp available
        elif (os.path.exists(actual_model_path) and 
            os.path.getsize(actual_model_path) > 1000000 and  # >1MB is probably valid
            actual_model_path.endswith(".gguf") and
            LLAMA_CPP_AVAILABLE):
            
            try:
                self.model = Llama(
                    model_path=actual_model_path,
                    n_ctx=context_size,
                    n_batch=512,  # Adjust based on available RAM
                )
                logger.info("Loaded GGUF model with llama-cpp")
            except Exception as e:
                logger.error(f"Failed to load GGUF model with llama-cpp: {e}")
        
        # Case 2: Directory with model files and transformers available
        elif (os.path.isdir(os.path.dirname(actual_model_path)) and
              TRANSFORMERS_AVAILABLE):
            
            # Check if there's a directory with model files
            model_dir = os.path.dirname(actual_model_path)
            
            try:
                # Use the directory with model files
                self.transformers_processor = TransformersLLMProcessor(
                    model_path=model_dir,
                    context_size=context_size,
                    max_tokens=max_tokens
                )
                self.use_transformers = True
                logger.info("Loaded model with Transformers")
            except Exception as e:
                logger.error(f"Failed to load model with Transformers: {e}")
        
        # Case 3: No valid model found
        else:
            logger.error(f"Model file {actual_model_path} not found or invalid, and no alternative available.")
            # Instead of using a mock, raise an exception
            raise ValueError(f"Model file {actual_model_path} not found or invalid, and no alternative available. Please ensure the model file exists and is valid.")
        
        self.max_tokens = max_tokens
    
    def create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the LLM.
        """
        # If using transformers, delegate to transformers processor
        if self.use_transformers:
            return self.transformers_processor.create_prompt(query, context)
            
        # Format context
        context_text = ""
        for i, doc in enumerate(context):
            context_text += f"Document {i+1}:\n{doc['chunk_text']}\n\n"
        
        # Llama 3.x prompt format
        prompt = f"""<|system|>
    You are a helpful AI assistant that provides accurate and concise answers based on the provided context documents. 
    If the answer is not contained in the documents, say "I don't have enough information to answer this question."
    Do not make up or hallucinate any information that is not supported by the documents.
    </|system|>

    <|user|>
    I need information about the following topic:

    {query}

    Here are relevant documents to help answer this question:

    {context_text}
    </|user|>

    <|assistant|>
    """
        return prompt        
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Response from the LLM
        """
        # If using meta adapter, use it
        if self.use_meta_adapter:
            response = self.meta_adapter(prompt, max_tokens=self.max_tokens)
            # Extract text
            response_text = response['choices'][0]['text'].strip()
            # Get metadata
            metadata = {
                'tokens_used': response['usage']['total_tokens'],
                'prompt_tokens': response['usage']['prompt_tokens'],
                'completion_tokens': response['usage']['completion_tokens'],
            }
            return {
                'text': response_text,
                'metadata': metadata
            }
        # If using transformers, delegate to transformers processor
        elif self.use_transformers:
            return self.transformers_processor.generate_response(prompt)
            
        logger.info("Generating LLM response with llama-cpp")
        
        # Generate response using llama-cpp
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            stop=["User query:", "\n\n"],
            temperature=0.2,  # Lower temperature for more factual responses
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        # Extract text
        response_text = response['choices'][0]['text'].strip()
        
        # Get metadata
        metadata = {
            'tokens_used': len(response['usage']['prompt_tokens']) + len(response['usage']['completion_tokens']),
            'prompt_tokens': len(response['usage']['prompt_tokens']),
            'completion_tokens': len(response['usage']['completion_tokens']),
        }
        
        logger.info(f"Generated response with {metadata['completion_tokens']} tokens")
        
        return {
            'text': response_text,
            'metadata': metadata
        }

class RAGProcessor:
    def __init__(self, search_pipeline, llm_processor):
        """
        Initialize the RAG processor.
        
        Args:
            search_pipeline: Search pipeline
            llm_processor: LLM processor
        """
        self.search_pipeline = search_pipeline
        self.llm_processor = llm_processor
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using RAG.
        
        Args:
            query: User query
            
        Returns:
            Response with text, sources, and metadata
        """
        logger.info(f"Processing RAG query: {query}")
        
        # Search for relevant documents
        search_results = self.search_pipeline.search(query)
        
        if not search_results:
            return {
                'text': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'metadata': {'search_results': 0}
            }
        
        # Create prompt
        prompt = self.llm_processor.create_prompt(query, search_results)
        
        # Generate response
        response = self.llm_processor.generate_response(prompt)
        
        # Format sources
        sources = []
        for result in search_results:
            sources.append({
                'filename': result['filename'],
                'chunk_text': result['chunk_text'],
                'rerank_score': result['rerank_score'],
                'vector_score': result['score']
            })
        
        # Combine results
        return {
            'text': response['text'],
            'sources': sources,
            'metadata': {
                'llm': response['metadata'],
                'search_results': len(search_results)
            }
        }

def create_rag_processor(search_pipeline, llm_model_path: str, 
                       context_size: int = 2048, max_tokens: int = 512) -> RAGProcessor:
    """
    Create a RAG processor.
    
    Args:
        search_pipeline: Search pipeline
        llm_model_path: Path to the LLM model
        context_size: Context size for the model
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        RAG processor
    """
    # Create LLM processor
    llm_processor = LLMProcessor(llm_model_path, context_size, max_tokens)
    
    # Create RAG processor
    rag_processor = RAGProcessor(search_pipeline, llm_processor)
    
    return rag_processor

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, LLM_MODEL_PATH
    )
    from app.utils.search import create_search_pipeline
    
    # Create search pipeline
    search_pipeline = create_search_pipeline(
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
    )
    
    # Create RAG processor
    rag_processor = create_rag_processor(search_pipeline, LLM_MODEL_PATH)
    
    # Process query
    query = "What is retrieval-augmented generation?"
    response = rag_processor.process_query(query)
    
    # Print response
    print(f"Query: {query}")
    print(f"Response: {response['text']}")
    print(f"Sources: {len(response['sources'])}")
    for i, source in enumerate(response['sources']):
        print(f"Source {i+1}: {source['filename']} (Score: {source['rerank_score']:.4f})")