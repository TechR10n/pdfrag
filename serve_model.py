#!/usr/bin/env python3
import os
import json
import logging
import time
import sys
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Get model IDs from environment variables
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
HF_EMBEDDING_MODEL_ID = os.environ.get("HF_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
HF_RERANKER_MODEL_ID = os.environ.get("HF_RERANKER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Extract model names (part after the last slash)
MODEL_NAME = HF_MODEL_ID.split('/')[-1]
EMBEDDING_MODEL_NAME = HF_EMBEDDING_MODEL_ID.split('/')[-1]
RERANKER_MODEL_NAME = HF_RERANKER_MODEL_ID.split('/')[-1]

# Environment variables
PORT = int(os.environ.get("PORT", 5000))
MODEL_PATH = os.environ.get("MODEL_PATH", f"/model_server/models/llm/{MODEL_NAME}")
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", f"/model_server/models/embedding/{EMBEDDING_MODEL_NAME}")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", f"/model_server/models/reranker/{RERANKER_MODEL_NAME}")

# Log the paths for debugging
logger.info(f"MODEL_PATH: {MODEL_PATH}")
logger.info(f"EMBEDDING_MODEL_PATH: {EMBEDDING_MODEL_PATH}")
logger.info(f"RERANKER_MODEL_PATH: {RERANKER_MODEL_PATH}")

# Initialize Flask app
app = Flask(__name__)

# Global variables for models
tokenizer = None
model = None
embedding_model = None
reranker_model = None

# Monkey patch the rope_scaling validation function to bypass the validation
try:
    from transformers.models.llama.configuration_llama import LlamaConfig
    
    # Store the original validation function
    original_validation = LlamaConfig._rope_scaling_validation
    
    # Define a new validation function that does nothing
    def no_validation(self):
        pass
    
    # Replace the validation function
    LlamaConfig._rope_scaling_validation = no_validation
    logger.info("Successfully monkey patched LlamaConfig._rope_scaling_validation")
except Exception as e:
    logger.warning(f"Failed to monkey patch LlamaConfig._rope_scaling_validation: {str(e)}")

def load_models():
    """Load all models."""
    global tokenizer, model, embedding_model, reranker_model
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load embedding model
    logger.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH}...")
    start_time = time.time()
    
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise
    
    # Load reranker model
    logger.info(f"Loading reranker model from {RERANKER_MODEL_PATH}...")
    start_time = time.time()
    
    try:
        reranker_model = CrossEncoder(RERANKER_MODEL_PATH)
        logger.info(f"Reranker model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error loading reranker model: {str(e)}")
        raise
    
    # Load LLM
    logger.info(f"Loading LLM from {MODEL_PATH}...")
    start_time = time.time()
    
    try:
        # First load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load the model with trust_remote_code=True to bypass validation
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        logger.info(f"LLM loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error loading LLM: {str(e)}")
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Transformers version: {__import__('transformers').__version__}")
        raise

def generate_rag_response(
    question: str,
    context: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate a response using the LLM with RAG context."""
    if not model or not tokenizer:
        error_msg = "LLM model or tokenizer not initialized"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        
    # Format context
    formatted_context = ""
    for i, ctx in enumerate(context):
        formatted_context += f"[{i+1}] {ctx['text']}\n"
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.
If the context doesn't contain relevant information, say you don't know.
Don't make up information that's not in the context.

Context:
{formatted_context}

Question: {question}

Answer:"""
    
    # Generate response
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Extract sources from context
    sources = [
        {
            "title": ctx.get("title", ""),
            "url": ctx.get("url", ""),
            "text": ctx.get("text", "")[:200] + "..." if len(ctx.get("text", "")) > 200 else ctx.get("text", "")
        }
        for ctx in context
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "confidence": 0.95,  # Placeholder confidence score
        "processed_timestamp": time.time()
    }

def embed_text(text: str) -> List[float]:
    """Embed text using the embedding model."""
    if not embedding_model:
        error_msg = "Embedding model not initialized"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        error_msg = f"Error embedding text: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def rerank_passages(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank passages using the reranker model."""
    if not reranker_model:
        error_msg = "Reranker model not initialized"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        
    if not passages:
        return []
    
    try:
        # Prepare passage pairs for reranking
        passage_pairs = [[query, passage] for passage in passages]
        
        # Get scores
        scores = reranker_model.predict(passage_pairs)
        
        # Create result with scores
        result = [
            {"passage": passage, "score": float(score)}
            for passage, score in zip(passages, scores)
        ]
        
        # Sort by score in descending order
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        
        return result
    except Exception as e:
        error_msg = f"Error reranking passages: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    models_loaded = all([model, tokenizer, embedding_model, reranker_model])
    return jsonify({
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "llm_loaded": model is not None and tokenizer is not None,
        "embedding_model_loaded": embedding_model is not None,
        "reranker_model_loaded": reranker_model is not None
    })

@app.route("/embed", methods=["POST"])
def embed():
    """Embed text endpoint."""
    data = request.json
    
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data["text"]
    
    try:
        embedding = embed_text(text)
        return jsonify({"embedding": embedding})
    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/rerank", methods=["POST"])
def rerank():
    """Rerank passages endpoint."""
    data = request.json
    
    if not data or "query" not in data or "passages" not in data:
        return jsonify({"error": "Missing 'query' or 'passages' field"}), 400
    
    query = data["query"]
    passages = data["passages"]
    
    try:
        # Prepare passage pairs for reranking
        passage_pairs = [[query, passage] for passage in passages]
        
        # Get scores
        scores = reranker_model.predict(passage_pairs)
        
        # Create result with scores
        result = [
            {"passage": passage, "score": float(score)}
            for passage, score in zip(passages, scores)
        ]
        
        # Sort by score in descending order
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        
        return jsonify({"results": result})
    except Exception as e:
        error_message = f"Error reranking passages: {str(e)}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 500

@app.route("/invocations", methods=["POST"])
def invocations():
    """MLflow model server invocations endpoint."""
    try:
        data = request.json
        logger.info(f"Received invocation request: {data}")
        
        if not data:
            return jsonify({"error_code": "BAD_REQUEST", "message": "Request body is empty"}), 400
            
        # Check for MLflow format (dataframe_split, instances, inputs, or dataframe_records)
        if "dataframe_split" in data:
            # Handle dataframe_split format
            query = data["dataframe_split"]["data"][0][0]  # Assuming first column is the query
            context = []  # No context in standard MLflow format
            max_new_tokens = 512  # Default value
            temperature = 0.7  # Default value
        elif "instances" in data:
            # Handle instances format
            query = data["instances"][0]  # Assuming first instance is the query
            context = []  # No context in standard MLflow format
            max_new_tokens = 512  # Default value
            temperature = 0.7  # Default value
        elif "inputs" in data:
            # Handle inputs format
            query = data["inputs"]
            context = []  # No context in standard MLflow format
            max_new_tokens = 512  # Default value
            temperature = 0.7  # Default value
        elif "dataframe_records" in data:
            # Handle dataframe_records format
            query = data["dataframe_records"][0]["query"]  # Assuming column name is "query"
            context = []  # No context in standard MLflow format
            max_new_tokens = 512  # Default value
            temperature = 0.7  # Default value
        elif "query" in data:
            # Handle direct query format (non-standard but useful for testing)
            query = data["query"]
            context = []  # No context in standard MLflow format
            max_new_tokens = 512  # Default value
            temperature = 0.7  # Default value
        elif "question" in data:
            # Handle the format previously used by /generate endpoint
            query = data["question"]
            context = data.get("context", [])
            max_new_tokens = data.get("max_new_tokens", 512)
            temperature = data.get("temperature", 0.7)
        else:
            return jsonify({
                "error_code": "BAD_REQUEST", 
                "message": "The input must be a JSON dictionary with one of the input fields {'dataframe_split', 'instances', 'inputs', 'dataframe_records', 'query', 'question'}. Received dictionary with input fields: " + str(list(data.keys()))
            }), 400
        
        # Generate response
        response = generate_rag_response(
            question=query,
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Format response according to MLflow protocol
        predictions = {
            "text": response.get("answer", ""),
            "sources": response.get("sources", [])
        }
        
        return jsonify({"predictions": predictions})
    except Exception as e:
        error_message = f"Error processing invocation: {str(e)}"
        logger.error(error_message)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error_code": "INTERNAL_ERROR", "message": error_message}), 500

if __name__ == "__main__":
    # Load models
    load_models()
    
    # Start server
    logger.info(f"Starting server on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT) 