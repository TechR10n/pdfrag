#!/bin/bash

# PDFrag - Ubuntu Linux Setup Script
# This script sets up the PDFrag system on a fresh Ubuntu installation

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PDFrag Setup Script for Ubuntu Linux ===${NC}"
echo "This script will set up the PDFrag system on your Ubuntu Linux machine."
echo "It requires sudo privileges to install system dependencies."
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then
  echo -e "${RED}Please do not run this script as root or with sudo.${NC}"
  echo "The script will prompt for sudo password when needed."
  exit 1
fi

# Check Ubuntu version
if ! grep -q "Ubuntu" /etc/os-release; then
  echo -e "${RED}This script is designed for Ubuntu Linux. Your system may not be compatible.${NC}"
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

echo -e "${YELLOW}Updating package lists...${NC}"
sudo apt-get update

echo -e "${YELLOW}Installing required system packages...${NC}"
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  git \
  curl \
  wget \
  build-essential \
  cmake \
  libcairo2-dev \
  pkg-config \
  python3-dev \
  libssl-dev \
  docker-compose

# Install Docker using the official Docker repository
echo -e "${YELLOW}Installing Docker from official Docker repository...${NC}"

# Uninstall old versions if they exist
sudo apt-get remove -y docker docker-engine docker.io containerd runc || true

# Update apt and install required dependencies
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up the stable repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Check Docker installation
echo -e "${YELLOW}Configuring Docker...${NC}"
if ! systemctl is-active --quiet docker; then
  sudo systemctl start docker
  sudo systemctl enable docker
fi

# Add current user to docker group to avoid using sudo with docker
if ! groups | grep -q docker; then
  sudo usermod -aG docker $USER
  echo -e "${YELLOW}Added user to docker group. You may need to log out and back in for this to take effect.${NC}"
  echo "Alternatively, run 'newgrp docker' to apply changes in the current session."
  newgrp docker
fi

# Clone repository if not already in it
REPO_DIR="pdfrag"
if [ ! -f "docker-compose.yml" ]; then
  echo -e "${YELLOW}Cloning PDFrag repository...${NC}"
  git clone https://github.com/yourusername/pdfrag.git $REPO_DIR
  cd $REPO_DIR
fi

# Create virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create data directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/documents
mkdir -p models/llm
mkdir -p models/embedding
mkdir -p models/reranker

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
  echo -e "${YELLOW}Creating .env file...${NC}"
  cp .env.example .env
  
  # Prompt for Hugging Face token
  echo
  echo "A Hugging Face token with read access is required to download models."
  echo "You can get your token from https://huggingface.co/settings/tokens"
  echo -n "Please enter your Hugging Face token: "
  read -s HF_TOKEN
  echo
  
  # Update the token in .env file
  sed -i "s/your_huggingface_token_here/$HF_TOKEN/g" .env
  
  # Prompt for Hugging Face Model IDs (with defaults)
  echo
  echo "Enter the Hugging Face model IDs to use:"
  
  echo "1. LLM Model ID (default: meta-llama/Llama-3.2-1B-Instruct):"
  read HF_MODEL_ID_INPUT
  
  echo "2. Embedding Model ID (default: sentence-transformers/all-MiniLM-L6-v2):"
  read HF_EMBEDDING_MODEL_ID_INPUT
  
  echo "3. Reranker Model ID (default: cross-encoder/ms-marco-MiniLM-L-6-v2):"
  read HF_RERANKER_MODEL_ID_INPUT
  
  # Use defaults if empty
  HF_MODEL_ID=${HF_MODEL_ID_INPUT:-meta-llama/Llama-3.2-1B-Instruct}
  HF_EMBEDDING_MODEL_ID=${HF_EMBEDDING_MODEL_ID_INPUT:-sentence-transformers/all-MiniLM-L6-v2}
  HF_RERANKER_MODEL_ID=${HF_RERANKER_MODEL_ID_INPUT:-cross-encoder/ms-marco-MiniLM-L-6-v2}
  
  # Update the model IDs in .env file
  sed -i "s|HF_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct|HF_MODEL_ID=$HF_MODEL_ID|g" .env
  sed -i "s|HF_EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2|HF_EMBEDDING_MODEL_ID=$HF_EMBEDDING_MODEL_ID|g" .env
  sed -i "s|HF_RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2|HF_RERANKER_MODEL_ID=$HF_RERANKER_MODEL_ID|g" .env
  
  # Generate a random Flask secret key
  FLASK_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
  sed -i "s/change-this-in-production/$FLASK_SECRET/g" .env
  
  # Update PlantUML path if installed
  PLANTUML_PATH=$(which plantuml 2>/dev/null || echo "")
  if [ -n "$PLANTUML_PATH" ]; then
    sed -i "s|/path/to/plantuml.jar|$PLANTUML_PATH|g" .env
  fi
  
  echo -e "${GREEN}Created .env file with your settings.${NC}"
fi

# Download models
echo -e "${YELLOW}Downloading models...${NC}"
echo "This may take some time depending on your internet connection."

# Export environment variables from .env
export $(grep -v '^#' .env | xargs)

# Derived variables
MODEL_NAME=$(echo $HF_MODEL_ID | awk -F/ '{print $NF}')
EMBEDDING_MODEL_NAME=$(echo $HF_EMBEDDING_MODEL_ID | awk -F/ '{print $NF}')
RERANKER_MODEL_NAME=$(echo $HF_RERANKER_MODEL_ID | awk -F/ '{print $NF}')

if [ -n "$HF_TOKEN" ]; then
  bash download_models.sh
else
  echo -e "${RED}Error: HF_TOKEN not found in .env file.${NC}"
  exit 1
fi

# Check if models were downloaded successfully - use dynamic paths
if [ ! -d "models/llm/$MODEL_NAME" ] || [ ! -d "models/embedding/$EMBEDDING_MODEL_NAME" ] || [ ! -d "models/reranker/$RERANKER_MODEL_NAME" ]; then
  echo -e "${RED}Error: Some models failed to download. Please check the logs.${NC}"
  exit 1
fi

echo -e "${YELLOW}Building Docker containers...${NC}"
docker-compose build

echo -e "${GREEN}Setup complete!${NC}"
echo
echo "To start the PDFrag system, run:"
echo "  ./startup.sh"
echo
echo "To access the web interface once started:"
echo "  http://localhost:8001"
echo
echo "If you encounter any issues:"
echo "1. Check the logs with 'docker-compose logs'"
echo "2. Make sure your Hugging Face token has access to the Meta Llama models"
echo "3. Ensure all Docker containers are running with 'docker-compose ps'"
echo
echo -e "${YELLOW}Important Note:${NC} If you just added your user to the docker group,"
echo "you may need to log out and log back in for the changes to take effect."

# Deactivate virtual environment
deactivate 