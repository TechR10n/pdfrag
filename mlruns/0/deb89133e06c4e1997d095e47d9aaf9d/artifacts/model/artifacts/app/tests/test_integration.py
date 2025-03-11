import os
import sys
import time
import pytest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.clients.mlflow_client import create_mlflow_client

@pytest.fixture
def mlflow_client():
    """Create an MLflow client for testing."""
    client = create_mlflow_client()
    if not client.is_alive():
        pytest.skip("MLflow endpoint is not available. Make sure the model is deployed.")
    return client

def test_mlflow_endpoint_alive(mlflow_client):
    """Test that the MLflow endpoint is alive."""
    assert mlflow_client.is_alive(), "MLflow endpoint is not alive"

def test_simple_query(mlflow_client):
    """Test a simple query."""
    response = mlflow_client.predict("What is machine learning?")
    assert 'text' in response, "Response missing 'text' field"
    assert len(response['text']) > 0, "Response text is empty"
    assert 'sources' in response, "Response missing 'sources' field"
    assert 'metadata' in response, "Response missing 'metadata' field"

def test_query_with_no_results(mlflow_client):
    """Test a query that should not have results in the corpus."""
    response = mlflow_client.predict("What is the capital of Jupiter?")
    assert 'text' in response, "Response missing 'text' field"
    # The response should indicate that the information is not available
    assert len(response['text']) > 0, "Response text is empty"

def test_response_timing(mlflow_client):
    """Test the response time of the endpoint."""
    query = "What is retrieval-augmented generation?"
    
    # Measure response time
    start_time = time.time()
    response = mlflow_client.predict(query)
    end_time = time.time()
    
    response_time = end_time - start_time
    print(f"Response time: {response_time:.2f} seconds")
    
    # We expect a response in under 10 seconds for a simple query
    assert response_time < 10, f"Response time too slow: {response_time:.2f} seconds"

def test_multiple_queries(mlflow_client):
    """Test multiple consecutive queries."""
    queries = [
        "What is vector search?",
        "How does re-ranking work?",
        "What are embeddings?",
        "How does Llama 2 compare to other language models?"
    ]
    
    for query in queries:
        response = mlflow_client.predict(query)
        assert 'text' in response, f"Response missing 'text' field for query: {query}"
        assert len(response['text']) > 0, f"Response text is empty for query: {query}"

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])