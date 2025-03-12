#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ROOT=$(pwd)
PDF_DIR="${PROJECT_ROOT}/data/documents"
MODELS_DIR="${PROJECT_ROOT}/models"
VENV_DIR="${PROJECT_ROOT}/venv"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Local RAG System...${NC}"

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check for requirements
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found.${NC}"
    exit 1
fi

# Install requirements if needed
pip -q install -r requirements.txt

# Check for models
if [ ! -d "$MODELS_DIR/embedding" ] || [ ! -d "$MODELS_DIR/reranker" ] || [ ! -d "$MODELS_DIR/llm" ]; then
    echo -e "${YELLOW}Some models are missing. Please download them first.${NC}"
    echo "You can use the following commands:"
    echo "  python app/scripts/download_models.py"
    echo "  ./app/scripts/download_llm.sh"
fi

# Create data directories if they don't exist
mkdir -p "$PDF_DIR"

# Start Docker services (including Flask app)
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Check if MLflow is running
echo "Checking MLflow service..."
if curl -s http://localhost:5001/ping > /dev/null; then
    echo -e "${GREEN}MLflow service is running.${NC}"
else
    echo -e "${RED}MLflow service is not running. Check Docker logs.${NC}"
    echo "You can run: docker-compose logs mlflow"
fi

# Check if vector database is running
echo "Checking vector database service..."
if curl -s http://localhost:6333/healthz > /dev/null; then
    echo -e "${GREEN}Vector database service is running.${NC}"
else
    echo -e "${RED}Vector database service is not running. Check Docker logs.${NC}"
    echo "You can run: docker-compose logs vector-db"
fi

# Check if Flask app is running
echo "Checking Flask application..."
if curl -s http://localhost:8001 > /dev/null; then
    echo -e "${GREEN}Flask application is running.${NC}"
else
    echo -e "${RED}Flask application is not running. Check Docker logs.${NC}"
    echo "You can run: docker-compose logs flask-app"
fi

# Check if there are PDFs to process
PDF_COUNT=$(find "$PDF_DIR" -name "*.pdf" | wc -l)
if [ "$PDF_COUNT" -gt 0 ]; then
    echo -e "${GREEN}Found $PDF_COUNT PDF files.${NC}"
    
    # Ask if user wants to process them
    read -p "Do you want to process them now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running processing pipeline..."
        python app/pipeline.py
    else
        echo "Skipping processing."
    fi
else
    echo "No PDF files found. You can upload them through the web interface."
fi

# Deploy the model to MLflow
echo "Deploying model to MLflow..."
# First, check if model exists
if mlflow models search -f "name = 'rag_model'" | grep -q "rag_model"; then
    echo "Model already exists, deploying latest version..."
    python app/scripts/deploy_model.py &
else
    echo "Model not found, logging new model..."
    python app/scripts/log_model.py
    python app/scripts/deploy_model.py &
fi

echo -e "${GREEN}All services started!${NC}"
echo "You can access the web interface at: http://localhost:8001"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for user interrupt
trap "echo 'Stopping services...'; docker-compose down; exit 0" INT
wait