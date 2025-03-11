import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchPipeline:
    def __init__(self, vector_db_client, query_processor, reranker,
                max_results: int = 10, rerank_results: int = 10):
        """
        Initialize the search pipeline.
        
        Args:
            vector_db_client: Vector database client
            query_processor: Query processor
            reranker: Reranker
            max_results: Maximum number of results to return
            rerank_results: Number of results to rerank
        """
        self.vector_db_client = vector_db_client
        self.query_processor = query_processor
        self.reranker = reranker
        self.max_results = max_results
        self.rerank_results = rerank_results
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for documents relevant to a query.
        
        Args:
            query: Query text
            
        Returns:
            Search results
        """
        logger.info(f"Searching for: {query}")
        
        # Process query
        query_embedding = self.query_processor.process_query(query)
        
        # Search vector database
        vector_results = self.vector_db_client.search(query_embedding, limit=self.rerank_results)
        logger.info(f"Found {len(vector_results)} results from vector search")
        
        if not vector_results:
            logger.warning("No results found in vector search")
            return []
        
        # Rerank results
        reranked_results = self.reranker.rerank(query, vector_results)
        logger.info("Results reranked")
        
        # Return top results
        return reranked_results[:self.max_results]

def create_search_pipeline(vector_db_host: str, vector_db_port: int, collection_name: str, 
                         vector_dimension: int, embedding_model_path: str, reranker_model_path: str,
                         max_results: int = 5, rerank_top_k: int = 10):
    """
    Create a search pipeline.
    
    Args:
        vector_db_host: Host of the vector database
        vector_db_port: Port of the vector database
        collection_name: Name of the collection
        vector_dimension: Dimension of the embedding vectors
        embedding_model_path: Path to the embedding model
        reranker_model_path: Path to the reranker model
        max_results: Maximum number of results to return
        rerank_top_k: Number of results to rerank
        
    Returns:
        Search pipeline
    """
    # Import here to avoid circular imports
    from app.utils.vector_db import VectorDBClient
    from app.utils.query_processing import QueryProcessor
    from app.utils.reranking import Reranker
    
    # Create components
    vector_db_client = VectorDBClient(vector_db_host, vector_db_port, collection_name, vector_dimension)
    query_processor = QueryProcessor(embedding_model_path)
    reranker = Reranker(reranker_model_path)
    
    # Create pipeline
    pipeline = SearchPipeline(
        vector_db_client, query_processor, reranker,
        max_results, rerank_top_k
    )
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
    )
    
    # Create search pipeline
    pipeline = create_search_pipeline(
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
    )
    
    # Search
    query = "What is retrieval-augmented generation?"
    results = pipeline.search(query)
    
    # Print results
    print(f"Top {len(results)} results for query: {query}")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['rerank_score']:.4f}, Vector Score: {result['score']:.4f}")
        print(f"   File: {result['filename']}")
        print(f"   Text: {result['chunk_text'][:200]}...")
        print()