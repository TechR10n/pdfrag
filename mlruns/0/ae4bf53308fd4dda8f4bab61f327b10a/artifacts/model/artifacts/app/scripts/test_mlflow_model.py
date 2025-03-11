import os
import sys
import mlflow
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME

def test_model():
    """Test loading and using the model from MLflow."""
    logger.info(f"Loading model {MLFLOW_MODEL_NAME} from MLflow at {MLFLOW_TRACKING_URI}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load the model
    model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/latest")
    
    # Test prediction
    query = "What is retrieval-augmented generation?"
    logger.info(f"Testing prediction with query: {query}")
    
    result = model.predict(query)
    
    logger.info(f"Prediction result: {result}")
    
    return result

if __name__ == "__main__":
    # Test the model
    result = test_model()
    print(f"Response: {result['text']}") 