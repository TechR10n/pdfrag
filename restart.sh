#!/bin/bash

# Stop and remove containers
echo "Stopping and removing containers..."
docker-compose down

# Rebuild containers
echo "Rebuilding containers..."
docker-compose build

# Start containers
echo "Starting containers..."
docker-compose up -d

# Show container status
echo "Container status:"
docker-compose ps

# Show logs
echo "Showing logs (press Ctrl+C to exit)..."
docker-compose logs -f 