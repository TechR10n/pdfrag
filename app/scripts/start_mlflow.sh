#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "NOTICE: Local MLflow serving is disabled."
echo "Using model server's /invocations endpoint on port 5002 instead."
echo "If you need to start the MLflow UI server (not the serving component), use port 5001."

# Create MLflow directories if they don't exist
mkdir -p "$PROJECT_ROOT/mlflow/artifacts" "$PROJECT_ROOT/mlflow/db"

# DISABLED: Local MLflow serving is no longer needed as we're using the model server's /invocations endpoint
# Start MLflow server with absolute paths
# mlflow server \
#   --backend-store-uri "sqlite:///$PROJECT_ROOT/mlflow/db/mlflow.db" \
#   --default-artifact-root "$PROJECT_ROOT/mlflow/artifacts" \
#   --host 0.0.0.0 \
#   --port 5004 

# If you need to start the MLflow UI server (not the serving component), use this command:
echo "To start the MLflow UI server, run:"
echo "mlflow server --backend-store-uri sqlite:///$PROJECT_ROOT/mlflow/db/mlflow.db --default-artifact-root $PROJECT_ROOT/mlflow/artifacts --host 0.0.0.0 --port 5001" 