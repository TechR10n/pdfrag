#!/bin/bash

# Check container status
echo "Container status:"
docker-compose ps

# Check model-server health
echo -e "\nChecking model-server health..."
curl -s http://localhost:5002/health | jq || echo "Failed to connect to model-server"

# Check flask-app health
echo -e "\nChecking flask-app health..."
curl -s http://localhost:8000/api/health | jq || echo "Failed to connect to flask-app"

# Check vector-db health
echo -e "\nChecking vector-db health..."
curl -s http://localhost:6333/healthz | jq || echo "Failed to connect to vector-db"

# Check mlflow health
echo -e "\nChecking mlflow health..."
curl -s http://localhost:5001/api/2.0/mlflow/experiments/list | jq || echo "Failed to connect to mlflow"

# Show recent logs
echo -e "\nRecent logs from model-server:"
docker-compose logs --tail=20 model-server

echo -e "\nRecent logs from flask-app:"
docker-compose logs --tail=20 flask-app 