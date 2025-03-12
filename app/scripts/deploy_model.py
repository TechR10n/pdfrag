import os
import sys
import mlflow
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME

def deploy_model(run_id=None, port=5002):
    """
    Deploy the RAG model using MLflow.
    
    Args:
        run_id: Run ID to deploy (if None, use latest version)
        port: Port to deploy on (default is 5002, which is the model server port)
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get model URI
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Deploying model from run {run_id}")
    else:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        logger.info(f"Deploying latest version of model {MLFLOW_MODEL_NAME}")
    
    # DISABLED: Local MLflow serving is no longer needed as we're using the model server's /invocations endpoint
    logger.info(f"Local MLflow serving on port 5004 is disabled.")
    logger.info(f"Using model server's /invocations endpoint on port {port} instead.")
    logger.info(f"Model URI: {model_uri}")
    
    # Instead of starting a local MLflow server, we're now using the model server's /invocations endpoint
    # The model server is already running on port 5002 (external) and handles requests to /invocations
    # os.system(f"mlflow models serve -m {model_uri} -p {port} --no-conda")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Deploy the RAG model')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID to deploy')
    parser.add_argument('--port', type=int, default=5002,
                        help='Port to deploy on (default is 5002, which is the model server port)')
    args = parser.parse_args()
    
    # Deploy model
    deploy_model(args.run_id, args.port)
