#!/bin/bash

# Create MLflow directories if they don't exist
mkdir -p mlflow/artifacts mlflow/db

# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow/db/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5001 