#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default backup directory
BACKUP_DIR="./backups"
mkdir -p "$BACKUP_DIR"

function backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${BACKUP_DIR}/rag_backup_${timestamp}.tar.gz"
    
    echo -e "${YELLOW}Creating backup...${NC}"
    
    # Stop services to ensure data consistency
    echo "Stopping services..."
    docker-compose stop
    
    # Create backup
    echo "Creating backup archive..."
    tar -czf "$backup_file" \
        --exclude="venv" \
        --exclude="__pycache__" \
        --exclude=".git" \
        --exclude="*.log" \
        --exclude="*.tar.gz" \
        --exclude="mlflow/artifacts/*/tmp" \
        ./data ./models ./mlflow ./app ./flask-app
    
    # Restart services
    echo "Restarting services..."
    docker-compose up -d
    
    echo -e "${GREEN}Backup created: ${backup_file}${NC}"
    echo "You can restore this backup using:"
    echo "./backup_restore.sh restore ${backup_file}"
}

function restore() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        echo -e "${RED}Backup file not found: ${backup_file}${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Restoring from backup: ${backup_file}${NC}"
    
    # Stop services
    echo "Stopping services..."
    docker-compose stop
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    
    # Extract backup
    echo "Extracting backup..."
    tar -xzf "$backup_file" -C "$temp_dir"
    
    # Restore data
    echo "Restoring data..."
    rsync -a --delete "$temp_dir/data/" ./data/
    rsync -a --delete "$temp_dir/models/" ./models/
    rsync -a --delete "$temp_dir/mlflow/" ./mlflow/
    
    # Clean up
    rm -rf "$temp_dir"
    
    # Restart services
    echo "Restarting services..."
    docker-compose up -d
    
    echo -e "${GREEN}Restore completed.${NC}"
}

function list_backups() {
    echo -e "${YELLOW}Available backups:${NC}"
    
    local count=0
    for file in "$BACKUP_DIR"/rag_backup_*.tar.gz; do
        if [ -f "$file" ]; then
            local size=$(du -h "$file" | cut -f1)
            local date=$(stat -c %y "$file" | cut -d. -f1)
            echo "$(basename "$file") (${size}, ${date})"
            count=$((count + 1))
        fi
    done
    
    if [ $count -eq 0 ]; then
        echo "No backups found."
    fi
}

# Main script
case "$1" in
    backup)
        backup
        ;;
    restore)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: No backup file specified.${NC}"
            echo "Usage: $0 restore <backup_file>"
            exit 1
        fi
        restore "$2"
        ;;
    list)
        list_backups
        ;;
    *)
        echo "Usage: $0 {backup|restore|list}"
        echo "  backup         Create a new backup"
        echo "  restore <file> Restore from backup file"
        echo "  list           List available backups"
        exit 1
        ;;
esac