import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.vector_db import VectorDBClient
from app.config.settings import VECTOR_DB_HOST, VECTOR_DB_PORT, VECTOR_DIMENSION

def test_vector_db_connection():
    """Test connecting to the vector database."""
    # Create client
    client = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, "test_collection", VECTOR_DIMENSION)
    
    # Check connection
    try:
        collections = client.client.get_collections()
        assert collections is not None, "Failed to get collections"
    except Exception as e:
        pytest.fail(f"Failed to connect to vector database: {str(e)}")

def test_collection_operations():
    """Test collection operations."""
    # Create client
    client = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, "test_collection", VECTOR_DIMENSION)
    
    # Create collection
    client.create_collection()
    
    # Check if collection exists
    collections = client.client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    assert "test_collection" in collection_names, "Collection not created"
    
    # Delete collection
    client.delete_collection()
    
    # Check if collection is deleted
    collections = client.client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    assert "test_collection" not in collection_names, "Collection not deleted"

def test_vector_operations():
    """Test vector operations."""
    # Create client
    client = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, "test_vectors", VECTOR_DIMENSION)
    
    # Create collection and clear any existing data
    client.delete_collection()
    client.create_collection()
    
    # Create test vectors
    import pandas as pd
    
    # Create 10 random test vectors
    np.random.seed(42)  # For reproducibility
    test_vectors = []
    for i in range(10):
        vec = np.random.rand(VECTOR_DIMENSION)
        # Normalize for cosine similarity
        vec = vec / np.linalg.norm(vec)
        test_vectors.append(vec)
    
    # Create test dataframe
    df = pd.DataFrame({
        'chunk_id': [f"chunk_{i}" for i in range(10)],
        'pdf_path': [f"/path/to/pdf_{i}.pdf" for i in range(10)],
        'filename': [f"pdf_{i}.pdf" for i in range(10)],
        'chunk_index': list(range(10)),
        'chunk_text': [f"This is test chunk {i}" for i in range(10)],
        'token_count': [len(f"This is test chunk {i}".split()) for i in range(10)],
        'embedding': test_vectors
    })
    
    # Upload vectors
    client.upload_vectors(df)
    
    # Check if vectors are uploaded
    count = client.count_vectors()
    assert count == 10, f"Expected 10 vectors, got {count}"
    
    # Search for similar vector
    results = client.search(test_vectors[0])
    assert len(results) > 0, "No search results returned"
    assert results[0]['chunk_id'] == "chunk_0", "First result should be the query vector itself"
    
    # Clean up
    client.delete_collection()

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
