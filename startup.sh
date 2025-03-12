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
    *)
      # Unknown option
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
  echo -e "${YELLOW}Waiting for vector-db to be healthy...${NC}"
  attempt=1
  max_attempts=10
  until [ $attempt -gt $max_attempts ] || docker-compose ps | grep -q "vector-db.*healthy"; do
    echo -e "${YELLOW}Waiting for vector-db to be healthy (attempt $attempt/$max_attempts)...${NC}"
    sleep 5
    attempt=$((attempt+1))
  done
  
  if [ $attempt -gt $max_attempts ]; then
    echo -e "${RED}Vector-db did not become healthy after $max_attempts attempts.${NC}"
    echo -e "${YELLOW}You may need to check the logs with: docker-compose logs vector-db${NC}"
    echo -e "${RED}Startup completed with warnings.${NC}"
    exit 1
  else
    echo -e "${GREEN}Vector-db is now healthy!${NC}"
    
    # Set rebuild index to true since we reset the database
    REBUILD_INDEX=true
  fi
else
  # Just start all containers if not resetting
  echo -e "${YELLOW}Starting all containers...${NC}"
  docker-compose up -d
  echo -e "${GREEN}All containers started.${NC}"
  
  # Wait for the vector-db container to be healthy
  echo -e "${YELLOW}Waiting for vector-db to be healthy...${NC}"
  attempt=1
  max_attempts=10
  until [ $attempt -gt $max_attempts ] || docker-compose ps | grep -q "vector-db.*healthy"; do
    echo -e "${YELLOW}Waiting for vector-db to be healthy (attempt $attempt/$max_attempts)...${NC}"
    sleep 5
    attempt=$((attempt+1))
  done
  
  if [ $attempt -gt $max_attempts ]; then
    echo -e "${RED}Vector-db did not become healthy after $max_attempts attempts.${NC}"
    echo -e "${YELLOW}You may need to check the logs with: docker-compose logs vector-db${NC}"
    echo -e "${RED}Startup completed with warnings.${NC}"
  else
    echo -e "${GREEN}Vector-db is now healthy!${NC}"
  fi
fi

# Rebuild index if requested or if we reset the database
if [ "$REBUILD_INDEX" = true ]; then
  echo -e "${YELLOW}Rebuilding vector index...${NC}"
  VECTOR_DB_HOST=localhost python -m app.pipeline --pdf-dir ./data/documents --rebuild
  echo -e "${GREEN}Vector index rebuilt.${NC}"
fi

echo -e "${GREEN}Startup completed successfully!${NC}"
echo -e "${YELLOW}You can access the web interface at http://localhost:8001${NC}"