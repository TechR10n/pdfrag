import os
import sys
import mlflow
import logging
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME
from app.models.rag_model import RAGModel, get_pip_requirements

def log_model():
    """Log the RAG model to MLflow."""
    logger.info(f"Logging model {MLFLOW_MODEL_NAME} to MLflow at {MLFLOW_TRACKING_URI}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create a temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory for artifacts: {temp_dir}")
        
        # Get project root
        project_root = Path(__file__).resolve().parent.parent.parent
        app_dir = os.path.join(project_root, "app")
        
        # Create a copy of the app directory in the temp directory
        temp_app_dir = os.path.join(temp_dir, "app")
        shutil.copytree(app_dir, temp_app_dir)
        
        # Log model
        with mlflow.start_run(run_name=f"{MLFLOW_MODEL_NAME}_deployment") as run:
            # Log parameters
            mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
            mlflow.log_param("reranker_model", "ms-marco-MiniLM-L-6-v2")
            mlflow.log_param("llm_model", "llama-2-7b-chat-q4_0.gguf")
            
            # Log model using the temporary directory for artifacts
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RAGModel(),
                artifacts={
                    "app_dir": temp_app_dir
                },
                pip_requirements=get_pip_requirements(),
                registered_model_name=MLFLOW_MODEL_NAME
            )
            
            logger.info(f"Model logged: {model_info.model_uri}")
            
            return model_info

if __name__ == "__main__":
    # Log model
    model_info = log_model()
    print(f"Model logged: {model_info.model_uri}")
    print(f"Run ID: {model_info.run_id}")
