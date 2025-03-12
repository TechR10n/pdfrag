#!/usr/bin/env python3
"""
Vector Database Connection Checker

This script checks if the Qdrant vector database is accessible and working properly.
It helps diagnose connection issues between your application and the vector database.
"""

import os
import sys
import logging
import argparse
import requests
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import settings
from app.config.settings import VECTOR_DB_HOST, VECTOR_DB_PORT

def check_vector_db_connection(host, port, max_retries=3):
    """Check if the vector database is accessible."""
    url = f"http://{host}:{port}/healthz"
    
    logger.info(f"Checking vector database connection at {url}")
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Successfully connected to vector database at {url}")
                logger.info(f"Response: {response.text}")
                return True
            else:
                logger.warning(f"❌ Vector database responded with status code {response.status_code}")
                logger.warning(f"Response: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ Connection attempt {attempt}/{max_retries} failed: {str(e)}")
            
        if attempt < max_retries:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.info(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    
    logger.error(f"❌ Failed to connect to vector database after {max_retries} attempts")
    return False

def check_dns_resolution(host):
    """Check if the hostname can be resolved."""
    logger.info(f"Checking DNS resolution for {host}")
    
    try:
        import socket
        ip = socket.gethostbyname(host)
        logger.info(f"✅ Successfully resolved {host} to {ip}")
        return True
    except socket.gaierror as e:
        logger.error(f"❌ Failed to resolve hostname {host}: {str(e)}")
        return False

def check_port_accessibility(host, port):
    """Check if the port is accessible."""
    logger.info(f"Checking if port {port} is accessible on {host}")
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✅ Port {port} is open on {host}")
            return True
        else:
            logger.error(f"❌ Port {port} is not accessible on {host}")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking port accessibility: {str(e)}")
        return False

def check_docker_status():
    """Check if Docker is running and the vector-db container is up."""
    logger.info("Checking Docker status")
    
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=vector-db", "--format", "{{.Names}} {{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "vector-db" in result.stdout:
            logger.info(f"✅ Docker container found: {result.stdout.strip()}")
            return True
        else:
            logger.warning("❌ Vector database container not found or not running")
            logger.info("Docker ps output:")
            subprocess.run(["docker", "ps"], check=False)
            return False
    except Exception as e:
        logger.error(f"❌ Error checking Docker status: {str(e)}")
        return False

def suggest_fixes(host, port):
    """Suggest fixes based on the diagnostics."""
    logger.info("\n=== SUGGESTED FIXES ===")
    
    if host == "vector-db":
        logger.info("You're using 'vector-db' as the hostname, which is only accessible within Docker:")
        logger.info("1. If running outside Docker, change VECTOR_DB_HOST to 'localhost' in your .env file")
        logger.info("2. If running inside Docker, ensure the container is on the same Docker network")
        logger.info("3. Check if the vector-db container is running with 'docker-compose ps'")
    else:
        logger.info("You're using a hostname other than 'vector-db':")
        logger.info("1. Ensure the Qdrant vector database is running")
        logger.info("2. Check if the port mapping is correct in docker-compose.yml")
        logger.info("3. Try restarting the containers with 'docker-compose down && docker-compose up -d'")
    
    logger.info("\nFor any machine:")
    logger.info("1. Ensure the .env file has the correct VECTOR_DB_HOST value for your environment")
    logger.info("2. When running locally (outside Docker): VECTOR_DB_HOST=localhost")
    logger.info("3. When running in Docker: VECTOR_DB_HOST=vector-db")

def main():
    parser = argparse.ArgumentParser(description='Check vector database connection')
    parser.add_argument('--host', type=str, default=VECTOR_DB_HOST,
                        help=f'Vector database host (default: {VECTOR_DB_HOST})')
    parser.add_argument('--port', type=int, default=VECTOR_DB_PORT,
                        help=f'Vector database port (default: {VECTOR_DB_PORT})')
    args = parser.parse_args()
    
    logger.info("=== VECTOR DATABASE CONNECTION CHECKER ===")
    logger.info(f"Current settings from environment: VECTOR_DB_HOST={VECTOR_DB_HOST}, VECTOR_DB_PORT={VECTOR_DB_PORT}")
    
    if args.host != VECTOR_DB_HOST or args.port != VECTOR_DB_PORT:
        logger.info(f"Using override settings: host={args.host}, port={args.port}")
    
    # Run checks
    dns_ok = check_dns_resolution(args.host)
    port_ok = check_port_accessibility(args.host, args.port) if dns_ok else False
    connection_ok = check_vector_db_connection(args.host, args.port) if port_ok else False
    
    # Check Docker status if using vector-db hostname
    if args.host == "vector-db":
        docker_ok = check_docker_status()
    
    # Suggest fixes
    if not connection_ok:
        suggest_fixes(args.host, args.port)
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"DNS Resolution: {'✅ PASSED' if dns_ok else '❌ FAILED'}")
    logger.info(f"Port Accessibility: {'✅ PASSED' if port_ok else '❌ FAILED'}")
    logger.info(f"Vector DB Connection: {'✅ PASSED' if connection_ok else '❌ FAILED'}")
    
    if connection_ok:
        logger.info("\n✅ All checks passed! Vector database is accessible and working properly.")
        return 0
    else:
        logger.error("\n❌ Some checks failed. Please review the suggested fixes above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 