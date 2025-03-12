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
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
if [ -n "$HF_TOKEN" ]; then
  bash download_models.sh
else
  echo -e "${RED}Error: HF_TOKEN not found in .env file.${NC}"
  exit 1
fi

# Check if models were downloaded successfully
if [ ! -d "models/llm/Llama-3.2-1B-Instruct" ] || [ ! -d "models/embedding/all-MiniLM-L6-v2" ] || [ ! -d "models/reranker/ms-marco-MiniLM-L-6-v2" ]; then
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
echo "  http://localhost:8000"
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