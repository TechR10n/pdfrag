"""Integration tests for the PDF RAG System."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path

from app.utils.pdf_ingestion import process_pdfs
from app.utils.vector_db import VectorDBClient
from app.utils.embedding_generation import EmbeddingGenerator


class TestPDFRagIntegration:
    """Integration tests for the PDF RAG system."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Create a temporary directory with test PDFs."""
        temp_dir = tempfile.mkdtemp()
        
        # Use the sample_pdf_dir fixture to populate our test directory
        sample_dir = pytest.lazy_fixture("sample_pdf_dir")
        
        # Copy files from sample_dir to temp_dir
        for item in os.listdir(sample_dir):
            s = os.path.join(sample_dir, item)
            d = os.path.join(temp_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.integration
    @pytest.mark.pdf
    def test_pdf_to_vector_db(self, test_data_dir, monkeypatch):
        """Test the full pipeline from PDF processing to vector database."""
        # Mock the embedding generator to return fixed vectors
        class MockEmbeddingGenerator:
            def embed_documents(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            def embed_query(self, query):
                return [0.1, 0.2, 0.3]
        
        # Mock the vector database client
        class MockVectorDBClient:
            def __init__(self, *args, **kwargs):
                pass
                
            def create_collection(self):
                pass
                
            def upload_vectors(self, df, vector_column='embedding', batch_size=100):
                pass
                
            def search(self, query_vector, limit=5):
                return [
                    {"text": "Document 1 content", "metadata": {"source": "doc1.pdf"}, "score": 0.95},
                    {"text": "Document 2 content", "metadata": {"source": "doc2.pdf"}, "score": 0.85},
                ]
        
        # Patch the embedding generator and vector database client
        monkeypatch.setattr("app.utils.embedding_generation.EmbeddingGenerator", MockEmbeddingGenerator)
        monkeypatch.setattr("app.utils.vector_db.VectorDBClient", MockVectorDBClient)
        
        # Process PDFs
        pdf_df = process_pdfs(test_data_dir)
        
        # Check that we have data
        assert not pdf_df.empty
        assert "text" in pdf_df.columns
        assert "path" in pdf_df.columns
        
        # Create a vector database client
        vector_db = VectorDBClient("localhost", 6333, "test_collection", 384)
        
        # Create collection
        vector_db.create_collection()
        
        # Add documents to vector database
        vector_db.upload_vectors(pdf_df)
        
        # Query the vector database
        results = vector_db.search([0.1, 0.2, 0.3], limit=2)
        
        # Check results
        assert len(results) > 0
        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert "score" in result
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_with_mocks(self, test_data_dir, monkeypatch):
        """Test the end-to-end flow with mocked components."""
        # Mock the necessary components
        class MockLLM:
            def generate(self, prompt, context):
                return f"Answer based on context: {context[:50]}..."
        
        class MockVectorDBClient:
            def search(self, query_vector, limit=3):
                return [
                    {"text": "Document 1 content", "metadata": {"source": "doc1.pdf"}, "score": 0.95},
                    {"text": "Document 2 content", "metadata": {"source": "doc2.pdf"}, "score": 0.85},
                ]
        
        # Patch the components
        monkeypatch.setattr("app.utils.llm.LLMProcessor", MockLLM)
        monkeypatch.setattr("app.utils.vector_db.VectorDBClient", MockVectorDBClient)
        
        # Import the RAG module (after patching)
        from app.utils.search import RAGSearch
        
        # Initialize the RAG system
        rag_system = RAGSearch()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        response = rag_system.process_query(query)
        
        # Check the response
        assert response is not None
        assert isinstance(response, str)
        assert "Answer based on context" in response