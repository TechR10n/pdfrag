import os
import mlflow
from app.models.rag_model import RAGModel

# Set MLflow tracking URI
mlflow_uri = f"file://{os.path.join(os.getcwd(), 'mlruns')}"
mlflow.set_tracking_uri(mlflow_uri)
print(f"MLflow tracking URI: {mlflow_uri}")

# Set experiment
mlflow.set_experiment('rag_model')

# Log model
with mlflow.start_run() as run:
    print(f"MLflow run ID: {run.info.run_id}")
    
    # Create model instance
    model = RAGModel()
    
    # Log model
    result = mlflow.pyfunc.log_model(
        artifact_path='model',
        python_model=model,
        artifacts={'app_dir': os.path.join(os.getcwd(), 'app')},
        registered_model_name='rag_model'
    )
    
    print(f"Model logged to: {result.model_uri}")
    print(f"Model registered as: {result.model_uri}") 