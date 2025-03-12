#!/bin/bash

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking Docker containers status...${NC}"
docker-compose ps

echo -e "\n${YELLOW}Checking Model Server health...${NC}"
if curl -s http://localhost:5002/health > /dev/null; then
    MODEL_SERVER_HEALTH=$(curl -s http://localhost:5002/health)
    echo -e "${GREEN}Model Server is up and running!${NC}"
    echo "Health response: $MODEL_SERVER_HEALTH"
else
    echo -e "${RED}Model Server is not responding!${NC}"
fi

echo -e "\n${YELLOW}Checking Flask App health...${NC}"
if curl -s http://localhost:8001/api/health > /dev/null; then
    FLASK_APP_HEALTH=$(curl -s http://localhost:8001/api/health)
    echo -e "${GREEN}Flask App is up and running!${NC}"
    echo "Health response: $FLASK_APP_HEALTH"
else
    echo -e "${RED}Flask App is not responding!${NC}"
fi

echo -e "\n${YELLOW}Testing API with a sample question...${NC}"
API_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d '{"question":"What is RAG?"}' http://localhost:8001/api/ask)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}API is responding to questions!${NC}"
    echo "Response: $API_RESPONSE"
else
    echo -e "${RED}API failed to process the question!${NC}"
fi

echo -e "\n${YELLOW}Recent logs from Model Server:${NC}"
docker-compose logs --tail=10 model-server

echo -e "\n${YELLOW}Recent logs from Flask App:${NC}"
docker-compose logs --tail=10 flask-app

echo -e "\n${GREEN}System check completed!${NC}" 