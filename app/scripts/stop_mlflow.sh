#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Stopping MLflow processes running on port 5004..."

# Find and kill processes running on port 5004
PIDS=$(lsof -ti:5004)
if [ -n "$PIDS" ]; then
    echo "Killing processes: $PIDS"
    kill -9 $PIDS
    echo "MLflow processes stopped."
else
    echo "No MLflow processes found running on port 5004."
fi 