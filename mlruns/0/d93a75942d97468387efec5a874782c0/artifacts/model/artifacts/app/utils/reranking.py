import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_path: str):
        """
        Initialize the reranker.
        
        Args:
            model_path: Path to the reranker model
        """
        logger.info(f"Loading reranker model from {model_path}")
        self.model = CrossEncoder(model_path, max_length=512)
        logger.info("Reranker model loaded")
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results.
        
        Args:
            query: Query text
            results: List of search results
            
        Returns:
            Reranked results
        """
        logger.info(f"Reranking {len(results)} results for query: {query}")
        
        if not results:
            return []
        
        # Create pairs for the cross-encoder
        pairs = [(query, result['chunk_text']) for result in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to results
        for i, score in enumerate(scores):
            results[i]['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info("Reranking complete")
        return reranked_results

def rerank_results(query: str, results: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
    """
    Rerank search results.
    
    Args:
        query: Query text
        results: List of search results
        model_path: Path to the reranker model
        
    Returns:
        Reranked results
    """
    reranker = Reranker(model_path)
    return reranker.rerank(query, results)

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        RERANKER_MODEL_PATH, EMBEDDING_MODEL_PATH,
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION
    )
    from app.utils.query_processing import process_query
    from app.utils.vector_db import VectorDBClient
    
    # Process a query
    query = "What is retrieval-augmented generation?"
    embedding = process_query(query, EMBEDDING_MODEL_PATH)
    
    # Search the vector database
    vector_db = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION)
    results = vector_db.search(embedding, limit=10)
    
    # Rerank results
    reranked_results = rerank_results(query, results, RERANKER_MODEL_PATH)
    
    # Print results
    print("Reranked results:")
    for i, result in enumerate(reranked_results[:5]):
        print(f"{i+1}. Score: {result['rerank_score']:.4f}, Original Score: {result['score']:.4f}")
        print(f"   Text: {result['chunk_text'][:100]}...")