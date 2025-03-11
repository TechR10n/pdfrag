import os
import sys
import time
import random
import threading
import concurrent.futures
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.clients.mlflow_client import create_mlflow_client

# Test queries
TEST_QUERIES = [
    "What is retrieval-augmented generation?",
    "How do vector databases work?",
    "What is semantic search?",
    "How do transformers work?",
    "What is the difference between bi-encoders and cross-encoders?",
    "How does Llama 2 compare to other language models?",
    "What is prompt engineering?",
    "How do you evaluate RAG systems?",
    "What is the role of re-ranking in search?",
    "How do embeddings capture semantic meaning?",
]

def run_query(client, query):
    """Run a query and return the response time."""
    start_time = time.time()
    try:
        response = client.predict(query)
        success = True
    except Exception as e:
        print(f"Error: {str(e)}")
        success = False
    end_time = time.time()
    
    return {
        'query': query,
        'response_time': end_time - start_time,
        'success': success,
        'timestamp': time.time()
    }

def worker(client, num_queries, results):
    """Worker function for concurrent queries."""
    for _ in range(num_queries):
        query = random.choice(TEST_QUERIES)
        result = run_query(client, query)
        results.append(result)
        time.sleep(random.uniform(0.5, 2.0))  # Random delay between queries

def run_load_test(num_workers=3, queries_per_worker=5):
    """
    Run a load test with multiple concurrent workers.
    
    Args:
        num_workers: Number of concurrent workers
        queries_per_worker: Number of queries per worker
    """
    print(f"Running load test with {num_workers} workers, {queries_per_worker} queries per worker")
    
    # Create client
    client = create_mlflow_client()
    
    # Check if endpoint is alive
    if not client.is_alive():
        print("MLflow endpoint is not available. Make sure the model is deployed.")
        return
    
    # Shared results list
    results = []
    
    # Run workers in parallel
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(worker, client, queries_per_worker, results)
            for _ in range(num_workers)
        ]
        concurrent.futures.wait(futures)
    end_time = time.time()
    
    # Analyze results
    total_queries = len(results)
    successful_queries = sum(1 for r in results if r['success'])
    failed_queries = total_queries - successful_queries
    
    if total_queries > 0:
        avg_response_time = sum(r['response_time'] for r in results) / total_queries
        max_response_time = max(r['response_time'] for r in results)
        min_response_time = min(r['response_time'] for r in results)
    else:
        avg_response_time = max_response_time = min_response_time = 0
    
    total_time = end_time - start_time
    queries_per_second = total_queries / total_time if total_time > 0 else 0
    
    # Print results
    print("\nLoad Test Results")
    print("----------------")
    print(f"Total queries: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Failed queries: {failed_queries}")
    print(f"Success rate: {successful_queries / total_queries * 100:.2f}%")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    print(f"Min response time: {min_response_time:.2f} seconds")
    print(f"Max response time: {max_response_time:.2f} seconds")
    print(f"Total test time: {total_time:.2f} seconds")
    print(f"Queries per second: {queries_per_second:.2f}")

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run a load test on the RAG system')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of concurrent workers')
    parser.add_argument('--queries', type=int, default=5,
                        help='Number of queries per worker')
    args = parser.parse_args()
    
    # Run load test
    run_load_test(args.workers, args.queries)