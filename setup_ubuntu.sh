#!/bin/bash
# setup_ubuntu.sh - Complete setup script for Ubuntu servers

set -e  # Exit immediately if a command exits with a non-zero status

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up PDFrag on Ubuntu...${NC}"

# Update system packages
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y
echo -e "${GREEN}System packages updated.${NC}"

# Install required dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    python3 \
    python3-pip \
    python3-venv
echo -e "${GREEN}Dependencies installed.${NC}"

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installed. You may need to log out and back in for group changes to take effect.${NC}"
else
    echo -e "${GREEN}Docker is already installed.${NC}"
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Installing Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installed.${NC}"
else
    echo -e "${GREEN}Docker Compose is already installed.${NC}"
fi

# Create project directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p ./data/documents
mkdir -p ./data/vectors
mkdir -p ./models/llm
mkdir -p ./models/embedding
mkdir -p ./models/reranker
chmod -R 777 ./data
chmod -R 777 ./models
echo -e "${GREEN}Project directories created.${NC}"

# Set up environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}.env file created from template.${NC}"
        echo -e "${YELLOW}Please edit .env file to add your Hugging Face token.${NC}"
    else
        echo -e "${RED}Error: .env.example file not found.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi

# Download models if needed
if [ ! -d "./models/embedding/all-MiniLM-L6-v2" ]; then
    echo -e "${YELLOW}Downloading models...${NC}"
    if [ -f "./download_models.sh" ]; then
        bash ./download_models.sh
        echo -e "${GREEN}Models downloaded.${NC}"
    else
        echo -e "${RED}Warning: download_models.sh not found. You'll need to download models manually.${NC}"
    fi
else
    echo -e "${GREEN}Models already downloaded.${NC}"
fi

# Initialize vector database
echo -e "${YELLOW}Initializing vector database...${NC}"
mkdir -p ./data/vectors
chmod 777 ./data/vectors
echo -e "${GREEN}Vector database initialized.${NC}"

# Build and start containers
echo -e "${YELLOW}Building and starting containers...${NC}"
docker-compose build
docker-compose up -d
echo -e "${GREEN}Containers built and started.${NC}"

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
  echo -e "${RED}Setup completed with warnings.${NC}"
else
  echo -e "${GREEN}Vector-db is now healthy!${NC}"
  
  # Ask if the user wants to index sample documents
  echo -e "${YELLOW}Do you want to index sample documents now? (y/n)${NC}"
  read -r answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Indexing sample documents...${NC}"
    VECTOR_DB_HOST=localhost python -m app.pipeline --pdf-dir ./data/documents --rebuild
    echo -e "${GREEN}Sample documents indexed.${NC}"
  else
    echo -e "${YELLOW}Skipping document indexing.${NC}"
    echo -e "${YELLOW}You can index documents later with:${NC}"
    echo -e "${YELLOW}VECTOR_DB_HOST=localhost python -m app.pipeline --pdf-dir ./data/documents --rebuild${NC}"
  fi
fi

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}You can access the web interface at http://localhost:8001${NC}"
echo -e "${YELLOW}If you need to reset the vector database in the future, run:${NC}"
echo -e "${YELLOW}./startup.sh --reset-vector-db${NC}"
