#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Offline Mode Test${NC}"
echo "This script will test if the system works properly without internet connectivity."
echo "It will temporarily disable network access for Docker containers."

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root to modify network settings${NC}"
  exit 1
fi

# Check if system is running
echo "Checking if system is running..."
if ! docker ps | grep -q "vector-db"; then
    echo -e "${RED}Vector database container not running. Please start the system first.${NC}"
    exit 1
fi

if ! docker ps | grep -q "mlflow"; then
    echo -e "${RED}MLflow container not running. Please start the system first.${NC}"
    exit 1
fi

if ! docker ps | grep -q "flask-app"; then
    echo -e "${RED}Flask app container not running. Please start the system first.${NC}"
    exit 1
fi

echo -e "${GREEN}All containers are running.${NC}"

# Create a Docker network with no internet access
echo "Creating isolated Docker network..."
docker network create --internal isolated-network

# Move containers to isolated network
echo "Moving containers to isolated network..."
docker network connect isolated-network $(docker ps -q --filter name=vector-db)
docker network connect isolated-network $(docker ps -q --filter name=mlflow)
docker network connect isolated-network $(docker ps -q --filter name=flask-app)

# Disconnect from default network
echo "Disconnecting from default network..."
docker network disconnect bridge $(docker ps -q --filter name=vector-db)
docker network disconnect bridge $(docker ps -q --filter name=mlflow)
docker network disconnect bridge $(docker ps -q --filter name=flask-app)

echo -e "${YELLOW}Containers are now in offline mode.${NC}"
echo "Testing system functionality..."

# Test if services are still responding
echo "Testing vector database..."
if curl -s http://localhost:6333/health > /dev/null; then
    echo -e "${GREEN}Vector database is responding in offline mode.${NC}"
else
    echo -e "${RED}Vector database is not responding!${NC}"
fi

echo "Testing MLflow..."
if curl -s http://localhost:5001/ping > /dev/null; then
    echo -e "${GREEN}MLflow is responding in offline mode.${NC}"
else
    echo -e "${RED}MLflow is not responding!${NC}"
fi

echo "Testing Flask app..."
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}Flask app is responding in offline mode.${NC}"
else
    echo -e "${RED}Flask app is not responding!${NC}"
fi

# Test a query to verify end-to-end functionality
echo "Testing query functionality..."
QUERY_RESULT=$(curl -s -X POST -H "Content-Type: application/json" -d '{"question":"What is machine learning?"}' http://localhost:8000/api/ask)

if [[ $QUERY_RESULT == *"text"* ]]; then
    echo -e "${GREEN}Query functionality is working in offline mode.${NC}"
else
    echo -e "${RED}Query functionality is not working in offline mode!${NC}"
    echo "Response: $QUERY_RESULT"
fi

# Wait for user to check the system
echo -e "${YELLOW}The system is now in offline mode. You can test it manually.${NC}"
echo "Press Enter to restore network connectivity..."
read

# Restore network connectivity
echo "Restoring network connectivity..."
docker network connect bridge $(docker ps -q --filter name=vector-db)
docker network connect bridge $(docker ps -q --filter name=mlflow)
docker network connect bridge $(docker ps -q --filter name=flask-app)

docker network disconnect isolated-network $(docker ps -q --filter name=vector-db)
docker network disconnect isolated-network $(docker ps -q --filter name=mlflow)
docker network disconnect isolated-network $(docker ps -q --filter name=flask-app)

# Remove isolated network
docker network rm isolated-network

echo -e "${GREEN}Network connectivity restored.${NC}"
echo "Offline mode test completed."