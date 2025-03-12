import os
import sys
import time
import pytest
import requests
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import PDF_UPLOAD_FOLDER, VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME
from app.utils.vector_db import VectorDBClient
from app.clients.mlflow_client import create_mlflow_client

# Test configuration
FLASK_BASE_URL = "http://localhost:8000"
TEST_PDF_PATH = os.path.join(Path(__file__).resolve().parent, "data", "sample.pdf")
TEST_QUESTION = "What is machine learning?"

def check_service_availability():
    """Check if all services are available."""
    services = {
        "Flask Web App": f"{FLASK_BASE_URL}/api/health",
        "MLflow": f"http://localhost:5001/ping",
        "Vector DB": f"http://localhost:6333/healthz",
    }
    
    available = {}
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            available[name] = response.status_code == 200
        except:
            available[name] = False
    
    return available

@pytest.fixture(scope="module")
def check_system():
    """Check if the entire system is ready for testing."""
    available = check_service_availability()
    
    if not all(available.values()):
        unavailable = [name for name, status in available.items() if not status]
        pytest.skip(f"Some services are not available: {', '.join(unavailable)}")

def test_health_endpoints(check_system):
    """Test health endpoints of all services."""
    # Flask health
    response = requests.get(f"{FLASK_BASE_URL}/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    
    # Check MLflow through Flask health endpoint
    assert data["mlflow"] == True

def test_vector_db_connection(check_system):
    """Test connection to vector database."""
    vector_db = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, 384)
    assert vector_db.client is not None
    
    # Check if collection exists
    collections = vector_db.client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    # If collection doesn't exist, this test is inconclusive
    if COLLECTION_NAME not in collection_names:
        pytest.skip(f"Collection {COLLECTION_NAME} does not exist yet")
    
    # Check collection count
    count = vector_db.count_vectors()
    print(f"Vector count in collection: {count}")

def test_mlflow_client(check_system):
    """Test MLflow client."""
    mlflow_client = create_mlflow_client()
    assert mlflow_client.is_alive() == True

def test_document_list(check_system):
    """Test document list API."""
    response = requests.get(f"{FLASK_BASE_URL}/documents")
    assert response.status_code == 200

def test_ask_question(check_system):
    """Test asking a question."""
    response = requests.post(
        f"{FLASK_BASE_URL}/api/ask",
        json={"question": TEST_QUESTION}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "text" in data
    assert len(data["text"]) > 0
    
    # Sources might be empty if no relevant documents are found
    assert "sources" in data

def test_end_to_end_flow(check_system):
    """Test the complete end-to-end flow."""
    # This test is more of a recipe for manual testing
    print("\nEnd-to-end test steps:")
    print("1. Upload a PDF document through the web interface")
    print("2. Trigger indexing process")
    print("3. Wait for indexing to complete")
    print("4. Ask a question related to the document content")
    print("5. Verify that the answer references information from the document")
    
    # Skip the actual test for automation
    pytest.skip("This test is a manual procedure")

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
