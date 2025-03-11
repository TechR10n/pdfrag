import os
from typing import Dict, Any, List, Optional
from llama_cpp import Llama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.model = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_batch=512,  # Adjust based on available RAM
        )
        self.max_tokens = max_tokens
        logger.info("LLM loaded")
    
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
        Generate a response from the LLM.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Response with text and metadata
        """
        logger.info("Generating LLM response")
        
        # Generate response
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