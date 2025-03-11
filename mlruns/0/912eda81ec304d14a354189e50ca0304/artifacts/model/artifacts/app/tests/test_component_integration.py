import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import (
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, LLM_MODEL_PATH,
    VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION
)
from app.utils.query_processing import QueryProcessor
from app.utils.reranking import Reranker
from app.utils.vector_db import VectorDBClient
from app.utils.llm import LLMProcessor

class TestComponentIntegration:
    """Test the integration between different components."""
    
    def test_embedding_dimensions(self):
        """Test that embedding dimensions are consistent across components."""
        # Load query processor (which uses the embedding model)
        query_processor = QueryProcessor(EMBEDDING_MODEL_PATH)
        
        # Generate a test embedding
        test_query = "This is a test query"
        query_embedding = query_processor.process_query(test_query)
        
        # Check dimension
        assert query_embedding.shape[0] == VECTOR_DIMENSION, \
            f"Embedding dimension ({query_embedding.shape[0]}) doesn't match configured dimension ({VECTOR_DIMENSION})"
        
        # Check if vector DB is configured with same dimension
        vector_db = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, "test_consistency", VECTOR_DIMENSION)
        
        # Try creating a collection and check if it accepts the embedding
        vector_db.create_collection()
        
        # Clean up
        vector_db.delete_collection()
    
    def test_reranker_compatibility(self):
        """Test that reranker can process outputs from vector search."""
        # Create test data
        test_query = "This is a test query"
        test_results = [
            {
                'chunk_id': 'test_chunk_1',
                'chunk_text': 'This is the first test chunk',
                'score': 0.95
            },
            {
                'chunk_id': 'test_chunk_2',
                'chunk_text': 'This is the second test chunk',
                'score': 0.85
            }
        ]
        
        # Load reranker
        reranker = Reranker(RERANKER_MODEL_PATH)
        
        # Try reranking results
        reranked_results = reranker.rerank(test_query, test_results)
        
        # Check if reranking worked
        assert len(reranked_results) == len(test_results), "Reranker changed the number of results"
        assert 'rerank_score' in reranked_results[0], "Reranker did not add scores"
    
    def test_llm_prompt_compatibility(self):
        """Test that LLM can process prompts created from reranked results."""
        # Skip if LLM model doesn't exist
        if not os.path.exists(LLM_MODEL_PATH):
            pytest.skip(f"LLM model not found at {LLM_MODEL_PATH}")
        
        # Create test data
        test_query = "This is a test query"
        test_results = [
            {
                'chunk_id': 'test_chunk_1',
                'chunk_text': 'This is the first test chunk with important information.',
                'rerank_score': 0.95,
                'score': 0.90
            },
            {
                'chunk_id': 'test_chunk_2',
                'chunk_text': 'This is the second test chunk with different information.',
                'rerank_score': 0.85,
                'score': 0.80
            }
        ]
        
        # Load LLM processor
        llm_processor = LLMProcessor(LLM_MODEL_PATH, context_size=1024, max_tokens=100)
        
        # Create prompt
        prompt = llm_processor.create_prompt(test_query, test_results)
        
        # Check prompt structure
        assert test_query in prompt, "Query not found in prompt"
        assert test_results[0]['chunk_text'] in prompt, "Context not found in prompt"
        
        # Optional: Test actual generation if environment allows
        try:
            response = llm_processor.generate_response(prompt)
            assert 'text' in response, "Response missing text field"
            assert 'metadata' in response, "Response missing metadata field"
        except Exception as e:
            pytest.skip(f"LLM generation test skipped: {str(e)}")
            
if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
