import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import (
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, MODEL_PATH,
    VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION
)
from app.utils.query_processing import QueryProcessor
from app.utils.reranking import Reranker
from app.utils.vector_db import VectorDBClient
from app.utils.llm import LLMProcessor

@pytest.mark.integration
class TestComponentIntegration:
    """Test the integration between different components."""
    
    @pytest.fixture
    def mock_vector_db(self, monkeypatch):
        """Create a mock vector database client."""
        class MockVectorDBClient:
            def __init__(self, *args, **kwargs):
                self.documents = []
                self.vectors = []
            
            def create_collection(self):
                pass
                
            def upload_vectors(self, documents, vector_column='embedding', batch_size=100):
                self.documents.extend(documents)
                return len(documents)
            
            def search(self, query_vector, limit=3):
                return [
                    {"text": "Document 1", "metadata": {"source": "doc1.pdf"}, "score": 0.95},
                    {"text": "Document 2", "metadata": {"source": "doc2.pdf"}, "score": 0.85},
                ]
        
        monkeypatch.setattr("app.utils.vector_db.VectorDBClient", MockVectorDBClient)
        return MockVectorDBClient()
    
    @pytest.fixture
    def mock_reranker(self, monkeypatch):
        """Create a mock reranker."""
        class MockReranker:
            def __init__(self, *args, **kwargs):
                pass
            
            def rerank(self, query, documents, top_k=3):
                return [
                    {"text": documents[0]["text"], "metadata": documents[0]["metadata"], "score": 0.98},
                    {"text": documents[1]["text"], "metadata": documents[1]["metadata"], "score": 0.75},
                ]
        
        monkeypatch.setattr("app.utils.reranking.Reranker", MockReranker)
        return MockReranker()
    
    @pytest.fixture
    def mock_llm_processor(self, monkeypatch):
        """Create a mock LLM processor."""
        class MockLLMProcessor:
            def __init__(self, *args, **kwargs):
                pass
            
            def generate_response(self, query, context):
                return f"Answer based on context: {context[:50]}..."
        
        monkeypatch.setattr("app.utils.llm.LLMProcessor", MockLLMProcessor)
        return MockLLMProcessor()
    
    @pytest.mark.integration
    def test_query_to_vector_db(self, mock_vector_db):
        """Test the flow from query to vector database retrieval."""
        # Create a query processor
        query_processor = QueryProcessor()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        processed_query = query_processor.process(query)
        
        # Query the vector database
        results = mock_vector_db.search(processed_query, limit=3)
        
        # Check results
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]
        assert "text" in results[0]
        assert "metadata" in results[0]
    
    @pytest.mark.integration
    def test_reranking_flow(self, mock_vector_db, mock_reranker):
        """Test the reranking flow."""
        # Create a query processor
        query_processor = QueryProcessor()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        processed_query = query_processor.process(query)
        
        # Query the vector database
        initial_results = mock_vector_db.search(processed_query, limit=5)
        
        # Rerank the results
        reranked_results = mock_reranker.rerank(query, initial_results, top_k=2)
        
        # Check results
        assert len(reranked_results) == 2
        assert reranked_results[0]["score"] > reranked_results[1]["score"]
        assert reranked_results[0]["score"] > initial_results[0]["score"]
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_flow(self, mock_vector_db, mock_reranker, mock_llm_processor):
        """Test the end-to-end flow from query to response."""
        # Create a query processor
        query_processor = QueryProcessor()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        processed_query = query_processor.process(query)
        
        # Query the vector database
        initial_results = mock_vector_db.search(processed_query, limit=5)
        
        # Rerank the results
        reranked_results = mock_reranker.rerank(query, initial_results, top_k=2)
        
        # Generate a response
        context = " ".join([doc["text"] for doc in reranked_results])
        response = mock_llm_processor.generate_response(query, context)
        
        # Check response
        assert response is not None
        assert isinstance(response, str)
        assert "Answer based on context" in response

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
