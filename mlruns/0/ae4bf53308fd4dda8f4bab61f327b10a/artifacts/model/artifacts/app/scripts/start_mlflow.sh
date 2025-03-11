#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Create MLflow directories if they don't exist
mkdir -p "$PROJECT_ROOT/mlflow/artifacts" "$PROJECT_ROOT/mlflow/db"

# Start MLflow server with absolute paths
mlflow server \
  --backend-store-uri "sqlite:///$PROJECT_ROOT/mlflow/db/mlflow.db" \
  --default-artifact-root "$PROJECT_ROOT/mlflow/artifacts" \
  --host 0.0.0.0 \
  --port 5001 