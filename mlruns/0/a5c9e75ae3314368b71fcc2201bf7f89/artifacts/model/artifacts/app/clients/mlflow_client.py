import os
import json
import requests
from typing import Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowClient:
    def __init__(self, endpoint_url: str):
        """
        Initialize the MLflow client.
        
        Args:
            endpoint_url: URL of the MLflow serving endpoint
        """
        self.endpoint_url = endpoint_url
        logger.info(f"Initialized MLflow client for endpoint: {endpoint_url}")
    
    def predict(self, query: str) -> Dict[str, Any]:
        """
        Make a prediction using the MLflow serving endpoint.
        
        Args:
            query: Query text
            
        Returns:
            Prediction result
        """
        logger.info(f"Sending query to MLflow endpoint: {query}")
        
        # Create payload
        payload = {
            "query": query
        }
        
        # Send request
        response = requests.post(
            f"{self.endpoint_url}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Error from MLflow endpoint: {response.text}")
            raise Exception(f"Error from MLflow endpoint: {response.text}")
        
        # Parse response
        result = response.json()
        
        logger.info(f"Received response from MLflow endpoint")
        return result
    
    def is_alive(self) -> bool:
        """
        Check if the MLflow endpoint is alive.
        
        Returns:
            True if the endpoint is alive, False otherwise
        """
        try:
            response = requests.get(f"{self.endpoint_url}/ping")
            return response.status_code == 200
        except:
            return False

def create_mlflow_client(host: str = "localhost", port: int = 5001) -> MLflowClient:
    """
    Create an MLflow client.
    
    Args:
        host: Host of the MLflow server
        port: Port of the MLflow server
        
    Returns:
        MLflow client
    """
    endpoint_url = f"http://{host}:{port}"
    return MLflowClient(endpoint_url)

if __name__ == "__main__":
    # Example usage
    client = create_mlflow_client()
    
    # Check if endpoint is alive
    if client.is_alive():
        print("MLflow endpoint is alive")
        
        # Make a prediction
        response = client.predict("What is retrieval-augmented generation?")
        print(f"Response: {response['text']}")
    else:
        print("MLflow endpoint is not available. Make sure the model is deployed.")