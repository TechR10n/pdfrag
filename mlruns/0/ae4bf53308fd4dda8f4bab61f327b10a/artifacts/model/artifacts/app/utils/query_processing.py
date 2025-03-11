import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, model_path: str):
        """
        Initialize the query processor.
        
        Args:
            model_path: Path to the embedding model
        """
        logger.info(f"Loading embedding model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
    
    def process_query(self, query: str) -> np.ndarray:
        """
        Process a query and generate an embedding.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        logger.info(f"Processing query: {query}")
        
        # Generate embedding
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info(f"Generated query embedding with shape: {embedding.shape}")
        return embedding

def process_query(query: str, model_path: str) -> np.ndarray:
    """
    Process a query and generate an embedding.
    
    Args:
        query: Query text
        model_path: Path to the embedding model
        
    Returns:
        Query embedding
    """
    processor = QueryProcessor(model_path)
    return processor.process_query(query)

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import EMBEDDING_MODEL_PATH
    
    # Process a query
    query = "What is retrieval-augmented generation?"
    embedding = process_query(query, EMBEDDING_MODEL_PATH)
    print(f"Query embedding shape: {embedding.shape}")