#!/usr/bin/env python3
import os
import sys
import subprocess
import requests
import time
import argparse

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_status(name, status, message=""):
    color = Colors.GREEN if status else Colors.RED
    status_text = "RUNNING" if status else "STOPPED"
    print(f"{name:20}: {color}{status_text}{Colors.ENDC} {message}")

def check_docker_service(service_name):
    try:
        output = subprocess.check_output(
            ["docker", "ps", "--filter", f"name={service_name}", "--format", "{{.Status}}"],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return len(output.strip()) > 0, output.strip()
    except subprocess.CalledProcessError:
        return False, "Docker not running"

def check_http_endpoint(url, timeout=2):
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200, f"Status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)

def check_system_status():
    print(f"\n{Colors.BOLD}Local RAG System Status{Colors.ENDC}\n")
    
    # Docker services
    print(f"{Colors.BOLD}Docker Services:{Colors.ENDC}")
    docker_services = [
        ("vector-db", "Qdrant Vector DB"),
        ("mlflow", "MLflow Server"),
        ("flask-app", "Flask Web App")
    ]
    
    for service_id, service_name in docker_services:
        status, message = check_docker_service(service_id)
        print_status(service_name, status, message)
    
    # HTTP endpoints
    print(f"\n{Colors.BOLD}Service Endpoints:{Colors.ENDC}")
    endpoints = [
        ("http://localhost:6333/health", "Vector DB API"),
        ("http://localhost:5001/ping", "MLflow API"),
        ("http://localhost:8000/api/health", "Flask API"),
    ]
    
    for url, name in endpoints:
        status, message = check_http_endpoint(url)
        print_status(name, status, message)
    
    # MLflow model
    print(f"\n{Colors.BOLD}MLflow Model:{Colors.ENDC}")
    try:
        # Check if the flask API can communicate with MLflow
        response = requests.get("http://localhost:8000/api/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            mlflow_status = data.get("mlflow", False)
            print_status("MLflow Model", mlflow_status, 
                        "Model is deployed and ready" if mlflow_status else "Model not deployed or not responding")
        else:
            print_status("MLflow Model", False, "Could not check via Flask API")
    except:
        print_status("MLflow Model", False, "Could not check via Flask API")
    
    # Data and Model directories
    print(f"\n{Colors.BOLD}Data Directories:{Colors.ENDC}")
    directories = [
        ("./data/pdfs", "PDF Storage"),
        ("./data/vectors", "Vector Storage"),
        ("./models/embedding", "Embedding Model"),
        ("./models/reranker", "Reranker Model"),
        ("./models/llm", "LLM Model")
    ]
    
    for path, name in directories:
        exists = os.path.exists(path)
        if exists and os.path.isdir(path):
            items = len(os.listdir(path))
            print_status(name, True, f"{items} items")
        else:
            print_status(name, False, "Directory not found")

def start_service(service_name):
    print(f"Starting {service_name}...")
    try:
        if service_name == "all":
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            print("All Docker services started")
            
            # Start the MLflow model server
            print("Starting MLflow model server...")
            subprocess.Popen(["python", "app/scripts/deploy_model.py"], 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        elif service_name in ["vector-db", "mlflow", "flask-app"]:
            subprocess.run(["docker-compose", "up", "-d", service_name], check=True)
            print(f"{service_name} started")
            
        elif service_name == "mlflow-model":
            subprocess.run(["python", "app/scripts/deploy_model.py"], check=True)
            print("MLflow model deployed")
            
        else:
            print(f"Unknown service: {service_name}")
            return
            
        print("Waiting for service to be ready...")
        time.sleep(5)
        check_system_status()
        
    except subprocess.CalledProcessError as e:
        print(f"Error starting service: {e}")

def stop_service(service_name):
    print(f"Stopping {service_name}...")
    try:
        if service_name == "all":
            subprocess.run(["docker-compose", "down"], check=True)
            print("All Docker services stopped")
            
        elif service_name in ["vector-db", "mlflow", "flask-app"]:
            subprocess.run(["docker-compose", "stop", service_name], check=True)
            print(f"{service_name} stopped")
            
        elif service_name == "mlflow-model":
            print("MLflow model server cannot be stopped directly")
            print("To stop it, restart the MLflow Docker container:")
            print("  docker-compose restart mlflow")
            
        else:
            print(f"Unknown service: {service_name}")
            return
            
        check_system_status()
        
    except subprocess.CalledProcessError as e:
        print(f"Error stopping service: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check status of Local RAG System")
    parser.add_argument("--start", help="Start a service (vector-db, mlflow, flask-app, mlflow-model, all)")
    parser.add_argument("--stop", help="Stop a service (vector-db, mlflow, flask-app, all)")
    
    args = parser.parse_args()
    
    if args.start:
        start_service(args.start)
    elif args.stop:
        stop_service(args.stop)
    else:
        check_system_status()
