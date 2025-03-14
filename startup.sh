#!/bin/bash
# startup.sh - Script to start the application with optional vector database reset

set -e  # Exit immediately if a command exits with a non-zero status

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

# Determine which Python command to use
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo -e "${GREEN}Using 'python' command.${NC}"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}Using 'python3' command.${NC}"
else
    echo -e "${RED}Error: Neither 'python' nor 'python3' command found. Please install Python 3.${NC}"
    exit 1
fi

# Function to display help
show_help() {
  echo "Usage: ./startup.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --reset-vector-db    Reset the vector database (removes all indexed documents)"
  echo "  --rebuild-index      Rebuild the vector index"
  echo "  --help               Display this help message"
  echo ""
  echo "Examples:"
  echo "  ./startup.sh                           # Start the application normally"
  echo "  ./startup.sh --reset-vector-db         # Reset the vector database and start the application"
  echo "  ./startup.sh --rebuild-index           # Rebuild the vector index and start the application"
  echo "  ./startup.sh --reset-vector-db --rebuild-index  # Reset and rebuild the vector database"
  echo ""
}

# Parse command line arguments
RESET_VECTOR_DB=false
REBUILD_INDEX=false

for arg in "$@"; do
  case $arg in
    --reset-vector-db)
      RESET_VECTOR_DB=true
      shift
      ;;
    --rebuild-index)
      REBUILD_INDEX=true
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      # Unknown option
      echo -e "${RED}Unknown option: $arg${NC}"
      show_help
      exit 1
      ;;
  esac
done

echo -e "${YELLOW}Starting PDFrag...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Check if vector database directory exists
if [ ! -d "./data/vectors" ]; then
  echo -e "${YELLOW}Vector database directory not found. Creating it...${NC}"
  mkdir -p ./data/vectors
  chmod 777 ./data/vectors
  echo -e "${GREEN}Vector database directory created.${NC}"
fi

# Reset vector database if requested
if [ "$RESET_VECTOR_DB" = true ]; then
  echo -e "${YELLOW}Resetting vector database as requested...${NC}"
  
  # Stop the vector-db container if it's running
  echo -e "${YELLOW}Stopping vector-db container...${NC}"
  if docker-compose ps | grep -q vector-db; then
    docker-compose stop vector-db
    echo -e "${GREEN}Vector-db container stopped.${NC}"
  else
    echo -e "${YELLOW}Vector-db container is not running.${NC}"
  fi
  
  # Remove the data directory
  echo -e "${YELLOW}Removing vector database data...${NC}"
  if [ -d "./data/vectors" ]; then
    rm -rf ./data/vectors
    echo -e "${GREEN}Vector database data removed.${NC}"
  else
    echo -e "${YELLOW}Vector database data directory does not exist.${NC}"
  fi
  
  # Create a fresh data directory with proper permissions
  echo -e "${YELLOW}Creating fresh data directory...${NC}"
  mkdir -p ./data/vectors
  chmod 777 ./data/vectors
  echo -e "${GREEN}Fresh data directory created with proper permissions.${NC}"
  
  # Start the vector-db container
  echo -e "${YELLOW}Starting vector-db container...${NC}"
  docker-compose up -d vector-db
  echo -e "${GREEN}Vector-db container started.${NC}"
  
  # Wait for the container to be healthy
  echo -e "${YELLOW}Waiting for vector-db to be ready...${NC}"
  attempt=1
  max_attempts=10
  until [ $attempt -gt $max_attempts ] || docker-compose ps | grep -q "vector-db.*Up"; do
    echo -e "${YELLOW}Waiting for vector-db to be ready (attempt $attempt/$max_attempts)...${NC}"
    sleep 5
    attempt=$((attempt+1))
  done
  
  if [ $attempt -gt $max_attempts ]; then
    echo -e "${RED}Vector-db did not become ready after $max_attempts attempts.${NC}"
    echo -e "${YELLOW}You may need to check the logs with: docker-compose logs vector-db${NC}"
    echo -e "${RED}Startup completed with warnings.${NC}"
    exit 1
  else
    echo -e "${GREEN}Vector-db is now ready!${NC}"
    
    # Set rebuild index to true since we reset the database
    REBUILD_INDEX=true
  fi
else
  # Just start all containers if not resetting
  echo -e "${YELLOW}Starting all containers...${NC}"
  docker-compose up -d
  echo -e "${GREEN}All containers started.${NC}"
  
  # Wait for the vector-db container to be ready
  echo -e "${YELLOW}Waiting for vector-db to be ready...${NC}"
  attempt=1
  max_attempts=10
  until [ $attempt -gt $max_attempts ] || docker-compose ps | grep -q "vector-db.*Up"; do
    echo -e "${YELLOW}Waiting for vector-db to be ready (attempt $attempt/$max_attempts)...${NC}"
    sleep 5
    attempt=$((attempt+1))
  done
  
  if [ $attempt -gt $max_attempts ]; then
    echo -e "${RED}Vector-db did not become ready after $max_attempts attempts.${NC}"
    echo -e "${YELLOW}You may need to check the logs with: docker-compose logs vector-db${NC}"
    echo -e "${RED}Startup completed with warnings.${NC}"
  else
    echo -e "${GREEN}Vector-db is now ready!${NC}"
  fi
fi

# Rebuild index if requested or if we reset the database
if [ "$REBUILD_INDEX" = true ]; then
  echo -e "${YELLOW}Rebuilding vector index...${NC}"
  
  # Check if pandas is installed
  if ! $PYTHON_CMD -c "import pandas" &> /dev/null; then
    echo -e "${YELLOW}Required Python dependencies not found. Installing...${NC}"
    if [ -f "requirements.txt" ]; then
      $PYTHON_CMD -m pip install -r requirements.txt
      echo -e "${GREEN}Dependencies installed.${NC}"
    else
      echo -e "${YELLOW}requirements.txt not found. Installing common dependencies...${NC}"
      $PYTHON_CMD -m pip install pandas numpy scikit-learn torch transformers qdrant-client
      echo -e "${GREEN}Common dependencies installed.${NC}"
    fi
  fi
  
  VECTOR_DB_HOST=localhost $PYTHON_CMD -m app.pipeline --pdf-dir ./data/documents --rebuild
  echo -e "${GREEN}Vector index rebuilt.${NC}"
fi

echo -e "${GREEN}Startup completed successfully!${NC}"
echo -e "${YELLOW}You can access the web interface at http://localhost:8001${NC}"