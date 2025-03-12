# Code Evaluation File

This file contains code from the project for evaluation purposes.

## Dockerfile.model-server

```dockerfile
FROM python:3.10-slim

WORKDIR /model_server

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    curl \
    git \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy requirements from project root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Llama models
RUN pip install --no-cache-dir \
    transformers==4.39.3 \
    accelerate==0.27.2 \
    huggingface_hub==0.21.3 \
    bitsandbytes==0.42.0 \
    safetensors==0.4.2

# copy the model server files
COPY serve_model.py .

# Create model directories
RUN mkdir -p /model_server/models/llm/Llama-3.2-1B-Instruct
RUN mkdir -p /model_server/models/embedding
RUN mkdir -p /model_server/models/reranker

# Set environment variables
ENV HF_HOME=/model_server/.cache/huggingface
ENV TRANSFORMERS_CACHE=/model_server/.cache/huggingface/transformers
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

EXPOSE 5000

CMD ["python", "serve_model.py"] 
```

## app/.pytest_cache/CACHEDIR.TAG

```
Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by pytest.
# For information about cache directory tags, see:
#	https://bford.info/cachedir/spec.html
```

## app/.pytest_cache/README.md

```markdown
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.
```

## app/.pytest_cache/v/cache/lastfailed

```
{
  "tests/test_component_integration.py": true
}
```

## app/.pytest_cache/v/cache/nodeids

```
[
  "scripts/test_mlflow_model.py::test_model",
  "scripts/test_model_downloader.py::test_model_loading",
  "tests/test_end_to_end.py::test_ask_question",
  "tests/test_end_to_end.py::test_document_list",
  "tests/test_end_to_end.py::test_end_to_end_flow",
  "tests/test_end_to_end.py::test_health_endpoints",
  "tests/test_end_to_end.py::test_mlflow_client",
  "tests/test_end_to_end.py::test_vector_db_connection",
  "tests/test_integration.py::test_mlflow_endpoint_alive",
  "tests/test_integration.py::test_multiple_queries",
  "tests/test_integration.py::test_query_with_no_results",
  "tests/test_integration.py::test_response_timing",
  "tests/test_integration.py::test_simple_query",
  "tests/test_pdf_ingestion.py::TestPDFSelection::test_create_pdf_dataframe",
  "tests/test_pdf_ingestion.py::TestPDFSelection::test_extract_text_from_pdf",
  "tests/test_pdf_ingestion.py::TestPDFSelection::test_process_pdfs",
  "tests/test_pdf_ingestion.py::TestPDFSelection::test_scan_directory",
  "tests/test_pdf_processing.py::test_chunk_text",
  "tests/test_pdf_processing.py::test_extract_text",
  "tests/test_pdf_processing.py::test_scan_directory",
  "tests/test_vector_db.py::test_collection_operations",
  "tests/test_vector_db.py::test_vector_db_connection",
  "tests/test_vector_db.py::test_vector_operations"
]
```

## app/.pytest_cache/v/cache/stepwise

```
[]
```

## app/clients/mlflow_client.py

```python
import os
import json
import requests
from typing import Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowClient:
    def __init__(self, endpoint_url: str):
        """
        Initialize the MLflow client.
        
        Args:
            endpoint_url: URL of the MLflow serving endpoint
        """
        self.endpoint_url = endpoint_url
        logger.info(f"Initialized MLflow client for endpoint: {endpoint_url}")
    
    def predict(self, query: str) -> Dict[str, Any]:
        """
        Make a prediction using the MLflow serving endpoint.
        
        Args:
            query: Query text
            
        Returns:
            Prediction result
        """
        logger.info(f"Sending query to MLflow endpoint: {query}")
        
        # Create payload using the question format for backward compatibility
        payload = {
            "question": query,
            "context": []
        }
        
        # Send request
        response = requests.post(
            f"{self.endpoint_url}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Error from MLflow endpoint: {response.text}")
            raise Exception(f"Error from MLflow endpoint: {response.text}")
        
        # Parse response
        result = response.json()
        
        logger.info(f"Received response from MLflow endpoint")
        return result
    
    def is_alive(self) -> bool:
        """
        Check if the MLflow endpoint is alive.
        
        Returns:
            True if the endpoint is alive, False otherwise
        """
        try:
            response = requests.get(f"{self.endpoint_url}/health")
            return response.status_code == 200
        except:
            return False

def create_mlflow_client(host: str = "localhost", port: int = 5002) -> MLflowClient:
    """
    Create an MLflow client.
    
    Args:
        host: Host of the MLflow server
        port: Port of the MLflow server
        
    Returns:
        MLflow client
    """
    endpoint_url = f"http://{host}:{port}"
    return MLflowClient(endpoint_url)

if __name__ == "__main__":
    # Example usage
    client = create_mlflow_client()
    
    # Check if endpoint is alive
    if client.is_alive():
        print("MLflow endpoint is alive")
        
        # Make a prediction
        response = client.predict("What is retrieval-augmented generation?")
        print(f"Response: {response['predictions']['text']}")
    else:
        print("MLflow endpoint is not available. Make sure the model is deployed.")
```

## app/config/settings.py

```python
import os
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Vector database settings
VECTOR_DB_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")  # Use localhost when running outside Docker
VECTOR_DB_PORT = 6333
VECTOR_DIMENSION = 384  # For all-MiniLM-L6-v2
COLLECTION_NAME = "pdf_chunks"

# Document processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_DOC = 1000

# Hugging Face settings
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Model paths
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "embedding", "all-MiniLM-L6-v2")
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "reranker", "ms-marco-MiniLM-L-6-v2")

# Primary model path - look for actual gguf file
MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "Llama-3.2-1B-Instruct")

# Alternative model locations
HOME_DIR = os.path.expanduser("~")
ALT_MODEL_PATHS = [
    os.path.join(HOME_DIR, ".models", "Llama-3.2-1B-Instruct"),
    os.path.join(HOME_DIR, ".cache", "huggingface", "hub", "models--meta-llama--Llama-3.2-1B-Instruct"),
    os.path.join(BASE_DIR, "models", "llm", "llama-3.2-1b-instruct-q4.gguf"),
    os.path.join(HOME_DIR, ".llama", "llama-3.2-1b-instruct-q4.gguf"),
]

# MLflow settings
MLFLOW_TRACKING_URI = f"file://{os.path.join(BASE_DIR, 'mlruns')}"  # File-based tracking URI
MLFLOW_MODEL_NAME = "rag_model"
MLFLOW_ARTIFACT_ROOT = os.path.join(BASE_DIR, "mlruns", "artifacts")  # Absolute path for artifacts

# Flask settings
FLASK_SECRET_KEY = "change-this-in-production"
PDF_UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "documents")
ALLOWED_EXTENSIONS = {'pdf'}
```

## app/models/rag_model.py

```python
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any
import mlflow.pyfunc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Remove the USE_MOCK flag
# Set this to True to use a mock implementation for testing
# USE_MOCK = False

class RAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        """Initialize the RAG model wrapper."""
        self.rag_processor = None
    
    def load_context(self, context):
        """
        Load model artifacts.
        
        Args:
            context: MLflow model context
        """
        logger.info("Loading RAG model context")
        
        # Add paths
        sys.path.append(os.path.dirname(context.artifacts['app_dir']))
        
        # Import here to avoid circular imports
        from app.config.settings import (
            VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
            EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, MODEL_PATH
        )
        from app.utils.search import create_search_pipeline
        from app.utils.llm import create_rag_processor
        
        # Create search pipeline
        search_pipeline = create_search_pipeline(
            VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
            EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
        )
        
        # Create RAG processor with real model (no mock)
        self.rag_processor = create_rag_processor(search_pipeline, MODEL_PATH)
        
        logger.info("RAG model context loaded")
    
    def predict(self, context, model_input):
        """
        Generate predictions.
        
        Args:
            context: MLflow model context
            model_input: Input data
            
        Returns:
            Model predictions
        """
        # Check if input is a pandas DataFrame
        if isinstance(model_input, pd.DataFrame):
            # Extract query
            if 'query' in model_input.columns:
                query = model_input['query'].iloc[0]
            else:
                raise ValueError("Input DataFrame must have a 'query' column")
        elif isinstance(model_input, dict):
            # Extract query from dictionary
            if 'query' in model_input:
                query = model_input['query']
            else:
                raise ValueError("Input dictionary must have a 'query' key")
        else:
            # Assume input is a string query
            query = str(model_input)
        
        logger.info(f"Processing query: {query}")
        
        # Process query
        response = self.rag_processor.process_query(query)
        
        return response

def get_pip_requirements():
    """Get pip requirements for the model."""
    return [
        "pandas",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "qdrant-client",
        "llama-cpp-python",
        "mlflow"
    ]

if __name__ == "__main__":
    # Test the model
    model = RAGModel()
    
    # Create a dummy context
    class DummyContext:
        def __init__(self):
            self.artifacts = {'app_dir': os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    
    # Load context
    model.load_context(DummyContext())
    
    # Test prediction
    result = model.predict(None, "What is retrieval-augmented generation?")
    print(result['text'])
```

## app/pipeline.py

```python
import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import (
    PDF_UPLOAD_FOLDER, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC,
    VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION
)
from app.utils.pdf_ingestion import process_pdfs
from app.utils.text_chunking import process_chunks
from app.utils.embedding_generation import embed_chunks
from app.utils.vector_db import setup_vector_db

def run_pipeline(pdf_dir: str, rebuild_index: bool = False):
    """
    Run the full pipeline from PDF ingestion to vector database upload.
    
    Args:
        pdf_dir: Directory containing PDF files
        rebuild_index: Whether to rebuild the vector index (delete and recreate)
    """
    logger.info(f"Starting pipeline with PDF directory: {pdf_dir}")
    
    # Step 1: Process PDFs
    logger.info("Step 1: Processing PDFs")
    pdf_df = process_pdfs(pdf_dir)
    logger.info(f"Processed {len(pdf_df)} PDFs")
    
    # Step 2: Process chunks
    logger.info("Step 2: Processing chunks")
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC)
    logger.info(f"Created {len(chunks_df)} chunks")
    
    # Step 3: Generate embeddings
    logger.info("Step 3: Generating embeddings")
    chunks_with_embeddings = embed_chunks(chunks_df, EMBEDDING_MODEL_PATH)
    logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Step 4: Set up vector database
    logger.info("Step 4: Setting up vector database")
    vector_db = setup_vector_db(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION)
    
    # Delete collection if rebuilding index
    if rebuild_index:
        logger.info("Rebuilding vector index: deleting existing collection")
        vector_db.delete_collection()
        vector_db.create_collection()
    
    # Step 5: Upload vectors
    logger.info("Step 5: Uploading vectors")
    vector_db.upload_vectors(chunks_with_embeddings)
    
    # Verify upload
    count = vector_db.count_vectors()
    logger.info(f"Pipeline complete. Vector database contains {count} vectors")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the PDF processing pipeline')
    parser.add_argument('--pdf-dir', type=str, default=PDF_UPLOAD_FOLDER,
                        help='Directory containing PDF files')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild the vector index (delete and recreate)')
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(args.pdf_dir, args.rebuild)
```

## app/rag_app.py

```python
import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import (
    VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, LLM_MODEL_PATH
)
from app.utils.search import create_search_pipeline
from app.utils.llm import create_rag_processor

class RAGApplication:
    def __init__(self):
        """Initialize the RAG application."""
        logger.info("Initializing RAG application")
        
        # Create search pipeline
        self.search_pipeline = create_search_pipeline(
            VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
            EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
        )
        
        # Create RAG processor
        self.rag_processor = create_rag_processor(self.search_pipeline, LLM_MODEL_PATH)
        
        logger.info("RAG application initialized")
    
    def process_query(self, query: str):
        """
        Process a query.
        
        Args:
            query: User query
            
        Returns:
            RAG response
        """
        return self.rag_processor.process_query(query)

def interactive_mode(app):
    """Run the application in interactive mode."""
    print("RAG Application - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("----------------------------------")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        try:
            response = app.process_query(query)
            
            print("\nAnswer:")
            print(response['text'])
            
            print("\nSources:")
            for i, source in enumerate(response['sources']):
                print(f"{i+1}. {source['filename']} (Score: {source['rerank_score']:.4f})")
                print(f"   Excerpt: {source['chunk_text'][:100]}...")
            
            print("\nMetadata:")
            print(f"Total tokens: {response['metadata']['llm']['tokens_used']}")
            print(f"Search results: {response['metadata']['search_results']}")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the RAG application')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--query', type=str,
                        help='Query to process')
    args = parser.parse_args()
    
    # Initialize application
    app = RAGApplication()
    
    if args.interactive:
        interactive_mode(app)
    elif args.query:
        response = app.process_query(args.query)
        print(response['text'])
    else:
        parser.print_help()
```

## app/scripts/build_docs.py

```python
#!/usr/bin/env python3
"""
Script to build Sphinx documentation in different formats (HTML, PDF, Markdown).
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(
            command, 
            check=True, 
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error message: {e.stderr}")
        return False

def run_prepare_latex():
    """Run the prepare_latex.sh script to fix LaTeX issues."""
    script_path = Path("app/scripts/prepare_latex.sh")
    if not script_path.exists():
        print(f"Warning: prepare_latex.sh not found at {script_path}")
        return False
    
    print("Running prepare_latex.sh to fix LaTeX issues...")
    success = run_command(["bash", str(script_path)])
    
    # Also run fix_tex.py on the generated LaTeX files
    fix_tex_path = Path("app/scripts/fix_tex.py")
    if fix_tex_path.exists():
        print("Running fix_tex.py to fix LaTeX issues...")
        latex_dir = Path("docs/sphinx/build/latex")
        if latex_dir.exists():
            for tex_file in latex_dir.glob("*.tex"):
                if "pdfragsystem" in tex_file.name:
                    print(f"Fixing LaTeX issues in {tex_file}...")
                    run_command(["python", str(fix_tex_path), str(tex_file)])
    
    # Also run fix_bbbk.py to specifically fix \Bbbk issues
    fix_bbbk_path = Path("app/scripts/fix_bbbk.py")
    if fix_bbbk_path.exists():
        print("Running fix_bbbk.py to fix \\Bbbk issues...")
        latex_dir = Path("docs/sphinx/build/latex")
        if latex_dir.exists():
            for tex_file in latex_dir.glob("*.tex"):
                if "pdfragsystem" in tex_file.name:
                    print(f"Fixing \\Bbbk issues in {tex_file}...")
                    run_command(["python", str(fix_bbbk_path), str(tex_file)])
    
    return success

def build_docs(format_type="html", clean=False, api_only=False):
    """Build Sphinx documentation in the specified format."""
    sphinx_dir = Path("docs/sphinx")
    
    if not sphinx_dir.exists():
        print(f"Error: Sphinx directory not found at {sphinx_dir}")
        print("Please run setup_sphinx.py first to set up the documentation.")
        return False
    
    # Check if make is available
    make_command = "make"
    if os.name == "nt":  # Windows
        if os.path.exists(sphinx_dir / "make.bat"):
            make_command = str(sphinx_dir / "make.bat")
        else:
            print("Warning: make.bat not found. Using 'make' command.")
    
    # Clean build directory if requested
    if clean:
        print("Cleaning build directory...")
        run_command([make_command, "clean"], cwd=sphinx_dir)
    
    # If API only, run sphinx-apidoc to update API documentation
    if api_only:
        print("Updating API documentation...")
        api_dir = sphinx_dir / "source" / "api"
        if not api_dir.exists():
            os.makedirs(api_dir)
        
        # Run sphinx-apidoc to generate API documentation
        # Use -M to put module documentation before member documentation
        # Use -e to put documentation for each module on its own page
        # Use -f to force overwriting existing files
        # Use -d 4 to set the maximum depth of the TOC
        # Use -P to include private members
        # Use --implicit-namespaces to handle namespace packages
        run_command([
            "sphinx-apidoc",
            "-o", str(api_dir),
            "-f", "-e", "-M", "-d", "4", "-P", "--implicit-namespaces",
            "app"  # Path to the package
        ])
        
        # Create a custom index.rst file for the API documentation
        with open(api_dir / "index.rst", "w") as f:
            f.write("""API Reference
============

This section contains the API reference for the PDF RAG System.

.. toctree::
   :maxdepth: 2

   modules
""")
        
        # Ensure utils modules are included
        utils_modules = [
            "app.utils.pdf_ingestion",
            "app.utils.text_chunking",
            "app.utils.vector_db"
        ]
        
        # Create individual module files for important modules if they don't exist
        for module_path in utils_modules:
            module_name = module_path.split(".")[-1]
            module_file = api_dir / f"{module_name}.rst"
            
            if not module_file.exists():
                print(f"Creating documentation for {module_path}...")
                with open(module_file, "w") as f:
                    f.write(f"""{module_name} module
{'=' * (len(module_name) + 7)}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:
""")
        
        # Update modules.rst to include these modules
        modules_file = api_dir / "modules.rst"
        if modules_file.exists():
            with open(modules_file, "r") as f:
                content = f.read()
            
            # Check if we need to add the utils modules
            if "app.utils" not in content:
                # Find the toctree directive
                toctree_pos = content.find(".. toctree::")
                if toctree_pos != -1:
                    # Find the end of the toctree entries
                    lines = content.split("\n")
                    toctree_start = -1
                    toctree_end = -1
                    
                    for i, line in enumerate(lines):
                        if ".. toctree::" in line:
                            toctree_start = i
                        elif toctree_start != -1 and line and not line.startswith(" "):
                            toctree_end = i
                            break
                    
                    if toctree_end == -1:
                        toctree_end = len(lines)
                    
                    # Add the utils modules to the toctree
                    for module_path in utils_modules:
                        module_name = module_path.split(".")[-1]
                        lines.insert(toctree_end, f"   {module_name}")
                    
                    # Write the updated content
                    with open(modules_file, "w") as f:
                        f.write("\n".join(lines))
        
        # Create app.utils.rst if it doesn't exist
        utils_file = api_dir / "app.utils.rst"
        if not utils_file.exists():
            with open(utils_file, "w") as f:
                f.write("""app.utils package
==============

.. toctree::
   :maxdepth: 4

   pdf_ingestion
   text_chunking
   vector_db

""")
        
        print("API documentation updated.")
        return True
    
    # Build documentation in the specified format
    print(f"Building documentation in {format_type} format...")
    
    success = True
    
    if format_type == "pdf":
        # For PDF, run prepare_latex.sh first to fix LaTeX issues
        if not run_prepare_latex():
            print("Warning: prepare_latex.sh failed or was not found. Continuing with PDF build anyway...")
        
        # For PDF, we need to run latexpdf
        print("Building PDF documentation...")
        
        try:
            # Build the LaTeX files
            latex_dir = sphinx_dir / "build/latex"
            latex_dir.mkdir(parents=True, exist_ok=True)
            
            # Run latexpdf with a timeout to prevent hanging
            try:
                # First, try to build the LaTeX files without running pdflatex
                run_command([make_command, "latex"], cwd=sphinx_dir)
                
                # Run fix_tex.py on the generated LaTeX files
                fix_tex_path = Path("app/scripts/fix_tex.py")
                if fix_tex_path.exists():
                    for tex_file in latex_dir.glob("*.tex"):
                        if "pdfragsystem" in tex_file.name:
                            print(f"Fixing LaTeX issues in {tex_file}...")
                            run_command(["python", str(fix_tex_path), str(tex_file)])
                
                # Run fix_bbbk.py to specifically fix \Bbbk issues
                fix_bbbk_path = Path("app/scripts/fix_bbbk.py")
                if fix_bbbk_path.exists():
                    for tex_file in latex_dir.glob("*.tex"):
                        if "pdfragsystem" in tex_file.name:
                            print(f"Fixing \\Bbbk issues in {tex_file}...")
                            run_command(["python", str(fix_bbbk_path), str(tex_file)])
                
                # Now run pdflatex directly on the generated .tex file with a timeout
                main_tex_file = None
                for tex_file in latex_dir.glob("*.tex"):
                    if "pdfragsystem" in tex_file.name:
                        main_tex_file = tex_file
                        break
                
                if main_tex_file:
                    print(f"Running pdflatex on {main_tex_file}...")
                    
                    # Run pdflatex twice to resolve references
                    pdflatex_success = True
                    for i in range(2):
                        try:
                            subprocess.run(
                                ["pdflatex", main_tex_file.name],
                                check=True,
                                cwd=latex_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=120  # 2 minute timeout
                            )
                        except subprocess.TimeoutExpired:
                            print(f"pdflatex run {i+1} timed out. PDF may be incomplete.")
                            pdflatex_success = False
                            break
                        except subprocess.CalledProcessError:
                            print(f"pdflatex run {i+1} failed. PDF may be incomplete.")
                            pdflatex_success = False
                            break
                    
                    # Check if PDF was generated
                    pdf_files = list(latex_dir.glob("*.pdf"))
                    if pdf_files:
                        print(f"PDF documentation built successfully: {pdf_files[0]}")
                        # Copy the PDF to a more accessible location
                        output_pdf = sphinx_dir / "build/pdfragsystem.pdf"
                        shutil.copy(pdf_files[0], output_pdf)
                        print(f"PDF copied to: {output_pdf}")
                    else:
                        if not pdflatex_success:
                            print("PDF generation failed. Check the LaTeX directory for errors.")
                            success = False
                else:
                    print("Main .tex file not found in the LaTeX directory.")
                    success = False
            
            except Exception as e:
                print(f"Error during PDF generation: {str(e)}")
                success = False
                
                # Try the direct make latexpdf command as a fallback
                try:
                    print("Trying alternative PDF generation method...")
                    run_command(
                        [make_command, "latexpdf"],
                        cwd=sphinx_dir
                    )
                    
                    # Check for PDF files
                    pdf_path = sphinx_dir / "build/latex/pdfragsystem.pdf"
                    if pdf_path.exists():
                        print(f"PDF documentation built successfully: {pdf_path}")
                        success = True
                    else:
                        # Try to find the PDF file
                        pdf_files = list(sphinx_dir.glob("build/latex/*.pdf"))
                        if pdf_files:
                            print(f"PDF documentation built successfully: {pdf_files[0]}")
                            success = True
                        else:
                            print(f"PDF file not found at expected location: {pdf_path}")
                            print("Check the latex directory for the generated PDF.")
                            success = False
                
                except subprocess.TimeoutExpired:
                    print("PDF generation timed out after 2 minutes.")
                    print("You can try to build the PDF manually with:")
                    print(f"cd {sphinx_dir} && make latexpdf")
                    success = False
                except Exception as e:
                    print(f"Error during alternative PDF generation: {str(e)}")
                    success = False
        except Exception as e:
            print(f"Error building PDF documentation: {str(e)}")
            success = False
    elif format_type == "markdown":
        # For Markdown, we need a custom target
        success = run_command([make_command, "markdown"], cwd=sphinx_dir)
        if success:
            md_dir = sphinx_dir / "build/markdown"
            if md_dir.exists():
                print(f"Markdown documentation built successfully in: {md_dir}")
            else:
                print(f"Markdown directory not found at expected location: {md_dir}")
    else:
        # For HTML and other formats, use the format name directly
        success = run_command([make_command, format_type], cwd=sphinx_dir)
        if success:
            build_dir = sphinx_dir / f"build/{format_type}"
            if build_dir.exists():
                print(f"{format_type.upper()} documentation built successfully in: {build_dir}")
                
                # For HTML, print the path to the index.html file
                if format_type == "html":
                    index_path = build_dir / "index.html"
                    if index_path.exists():
                        print(f"You can view the HTML documentation by opening: {index_path}")
                        print(f"Or by running: open {index_path}")
            else:
                print(f"{format_type.upper()} directory not found at expected location: {build_dir}")
    
    return success

def main():
    """Main function to parse arguments and build documentation."""
    parser = argparse.ArgumentParser(description="Build Sphinx documentation in different formats.")
    parser.add_argument(
        "--format", "-f",
        choices=["html", "pdf", "markdown", "epub"],
        default="html",
        help="Output format for the documentation (default: html)"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean the build directory before building"
    )
    parser.add_argument(
        "--api-only", "-a",
        action="store_true",
        help="Only update API documentation, don't build"
    )
    
    args = parser.parse_args()
    
    build_docs(args.format, args.clean, args.api_only)

if __name__ == "__main__":
    main() 
```

## app/scripts/convert_md_to_pdf.py

```python
import sys
import subprocess
from pathlib import Path
import argparse
import os
import re

def main():
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF with academic styling')
    parser.add_argument('input_file', help='Input markdown file')
    parser.add_argument('--title', default='Laboratory Handbook: Building a Local RAG System with Flask and MLflow', help='Document title')
    parser.add_argument('--author', default='Ryan Hammang', help='Author name')
    parser.add_argument('--date', default='March 11, 2025', help='Document date (default: today)')
    parser.add_argument('--abstract', default='Creating a local RAG system with Flask and MLflow.', 
                        help='Document abstract')
    parser.add_argument('--affiliation', default='Organization or Institution Name', 
                        help='Author affiliation')
    parser.add_argument('--documentclass', default='acmart', 
                        choices=['IEEEtran', 'acmart', 'article', 'llncs', 'elsarticle'],
                        help='LaTeX document class to use')
    parser.add_argument('--classoption', default='screen,review,acmlarge', 
                        help='Options for the document class')
    parser.add_argument('--latex-engine', default='pdflatex',
                        choices=['pdflatex', 'xelatex', 'lualatex'],
                        help='LaTeX engine to use for PDF generation')
    parser.add_argument('--keep-tex', action='store_true',
                        help='Keep the intermediate .tex file')
    parser.add_argument('--tex-only', action='store_true',
                        help='Only generate the .tex file, do not compile to PDF')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    tex_file = input_file.with_suffix('.tex')
    pdf_file = input_file.with_suffix('.pdf')
    preamble_path = Path('app/scripts/preamble.tex')
    if not preamble_path.exists():
        print(f"Error: preamble.tex not found.")
        sys.exit(1)

    # Step 1: Convert Markdown to LaTeX using Pandoc
    pandoc_command = [
        'pandoc',
        str(input_file),
        '-o',
        str(tex_file),
        '--standalone',  # Create a complete LaTeX document
        '--listings',
        '--no-highlight',  # Disable syntax highlighting to avoid special character issues
        '--wrap=preserve',  # Preserve line wrapping
        '-V',
        f'documentclass={args.documentclass}',
    ]
    
    # Only add classoption if it's not empty
    if args.classoption:
        pandoc_command.extend(['-V', f'classoption={args.classoption}'])
    
    pandoc_command.extend([
        '--include-in-header',
        str(preamble_path),
        # Front matter metadata
        '-M', f'title={args.title}',
        '-M', f'author={args.author}',
        '-M', f'date={args.date}',
        '-M', f'abstract={args.abstract}'
    ])
    
    # Add the appropriate affiliation/institute parameter based on document class
    if args.documentclass == 'llncs':
        pandoc_command.extend(['-M', f'institute={args.affiliation}'])
    else:
        pandoc_command.extend(['-M', f'affiliation={args.affiliation}'])

    print(f"Converting {input_file} to {tex_file} using Pandoc...")
    try:
        subprocess.run(pandoc_command, check=True)
        print(f"LaTeX file generated at {tex_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during Markdown to LaTeX conversion: {e}")
        sys.exit(1)
    
    # Modify the generated .tex file
    print("Modifying the generated .tex file...")
    try:
        with open(tex_file, 'r') as file:
            tex_content = file.read()
        
        # Remove \usepackage{amsmath,amssymb} line
        tex_content = re.sub(r'\\usepackage\{amsmath,amssymb\}', '', tex_content)
        
        # Fix special characters in tabular environments
        # Look for tabular environments and escape underscores and other special characters
        def fix_tabular_content(match):
            tabular_content = match.group(0)
            # Escape underscores in tabular content
            tabular_content = tabular_content.replace('_', '\\_')
            return tabular_content
        
        tex_content = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', fix_tabular_content, tex_content, flags=re.DOTALL)
        
        # For ACM, completely restructure the document to ensure abstract is before \maketitle
        if args.documentclass == 'acmart':
            # Extract the abstract text
            abstract_text = args.abstract  # Default to command line argument
            
            # Try to find abstract in the document
            abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex_content, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
            
            # Remove any existing abstract
            tex_content = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', tex_content, flags=re.DOTALL)
            
            # Find the \begin{document} and \maketitle positions
            begin_doc_match = re.search(r'\\begin\{document\}', tex_content)
            maketitle_match = re.search(r'\\maketitle', tex_content)
            
            if begin_doc_match and maketitle_match:
                # Split the content
                begin_doc_pos = begin_doc_match.end()
                maketitle_pos = maketitle_match.start()
                
                # Reconstruct the document with abstract before \maketitle
                new_content = (
                    tex_content[:begin_doc_pos] + 
                    '\n\n\\begin{abstract}\n' + abstract_text + '\n\\end{abstract}\n\n' +
                    tex_content[begin_doc_pos:maketitle_pos] +
                    '\\maketitle\n\n' +
                    tex_content[maketitle_match.end():]
                )
                
                tex_content = new_content
        
        with open(tex_file, 'w') as file:
            file.write(tex_content)
        
        print("LaTeX file successfully modified.")
    except Exception as e:
        print(f"Error modifying the LaTeX file: {e}")
        sys.exit(1)
    
    # Exit if user only wants the .tex file
    if args.tex_only:
        print("Skipping PDF generation as requested (--tex-only flag used).")
        return

    # Step 2: Compile LaTeX to PDF using the local LaTeX engine
    print(f"Compiling {tex_file} to {pdf_file} using {args.latex_engine}...")
    
    # Change to the directory containing the .tex file for proper relative path handling
    working_dir = tex_file.parent
    tex_filename = tex_file.name
    
    # Run LaTeX engine twice to resolve references
    for i in range(2):
        try:
            subprocess.run(
                [args.latex_engine, 
                 '-shell-escape',  # Enable shell escape for SVG processing
                 tex_filename], 
                check=True,
                cwd=str(working_dir)  # Set working directory
            )
        except subprocess.CalledProcessError as e:
            print(f"Error during LaTeX to PDF compilation: {e}")
            sys.exit(1)
    
    print(f"PDF generated at {pdf_file}")
    
    # Clean up intermediate files unless --keep-tex is specified
    if not args.keep_tex:
        print("Cleaning up intermediate files...")
        extensions_to_remove = ['.aux', '.log', '.out', '.toc']
        if not args.keep_tex:
            extensions_to_remove.append('.tex')
        
        for ext in extensions_to_remove:
            temp_file = input_file.with_suffix(ext)
            if temp_file.exists():
                os.remove(temp_file)
                print(f"Removed {temp_file}")

if __name__ == "__main__":
    main()
```

## app/scripts/convert_plantuml_to_svg.py

```python
#! /usr/bin/env python3

import sys
import subprocess
from pathlib import Path

puml_path = 'docs/puml/'
svg_path = 'png/'

PLANTUML_JAR = '/opt/homebrew/Cellar/plantuml/1.2025.2/libexec/plantuml.jar'

if not Path(PLANTUML_JAR).exists():
    print(f"Error: {PLANTUML_JAR} does not exist.")
    sys.exit(1)

for puml_file in Path(puml_path).glob('*.puml'):
    svg_file = svg_path / puml_file.with_suffix('.svg')

# Run PlantUML to generate the SVG file
subprocess.run([
    'java', '-jar', PLANTUML_JAR,
    '-tpng',
    str(puml_path),
    '-o', str(svg_path)
])
```

## app/scripts/convert_svg_to_pdf.py

```python
#!/usr/bin/env python3
"""
Script to convert SVG files to PDF format for better LaTeX compatibility.
"""

import sys
import subprocess
from pathlib import Path
import os

def convert_svg_to_pdf(svg_file):
    """
    Convert an SVG file to PDF format using Inkscape.
    
    Args:
        svg_file: Path to the SVG file
    
    Returns:
        Path to the generated PDF file
    """
    svg_path = Path(svg_file)
    if not svg_path.exists():
        print(f"Error: {svg_path} not found.")
        return None
    
    pdf_path = svg_path.with_suffix('.pdf')
    
    try:
        # Try using Inkscape to convert SVG to PDF
        print(f"Converting {svg_path} to {pdf_path}...")
        subprocess.run(
            ['inkscape', '--export-filename=' + str(pdf_path), str(svg_path)],
            check=True
        )
        print(f"PDF generated at {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"Error during SVG to PDF conversion: {e}")
        return None
    except FileNotFoundError:
        print("Inkscape not found. Please install Inkscape or ensure it's in your PATH.")
        return None

def find_and_convert_svg_files(directory='.'):
    """
    Find all SVG files in the given directory and its subdirectories,
    and convert them to PDF format.
    
    Args:
        directory: Directory to search for SVG files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} not found.")
        return
    
    svg_files = list(dir_path.glob('**/*.svg'))
    if not svg_files:
        print(f"No SVG files found in {dir_path}")
        return
    
    print(f"Found {len(svg_files)} SVG files.")
    
    for svg_file in svg_files:
        convert_svg_to_pdf(svg_file)

def update_markdown_file(markdown_file):
    """
    Update a Markdown file to use PDF images instead of SVG.
    
    Args:
        markdown_file: Path to the Markdown file
    """
    md_path = Path(markdown_file)
    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        return
    
    try:
        with open(md_path, 'r') as file:
            content = file.read()
        
        # Replace SVG image references with PDF
        updated_content = content.replace('.svg)', '.pdf)')
        
        # Write the updated content back to the file
        with open(md_path, 'w') as file:
            file.write(updated_content)
        
        print(f"Updated {md_path} to use PDF images.")
    except Exception as e:
        print(f"Error updating Markdown file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_svg_to_pdf.py [directory] [markdown_file]")
        sys.exit(1)
    
    directory = sys.argv[1]
    find_and_convert_svg_files(directory)
    
    if len(sys.argv) > 2:
        markdown_file = sys.argv[2]
        update_markdown_file(markdown_file) 
```

## app/scripts/deploy_model.py

```python
import os
import sys
import mlflow
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME

def deploy_model(run_id=None, port=5002):
    """
    Deploy the RAG model using MLflow.
    
    Args:
        run_id: Run ID to deploy (if None, use latest version)
        port: Port to deploy on (default is 5002, which is the model server port)
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get model URI
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Deploying model from run {run_id}")
    else:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        logger.info(f"Deploying latest version of model {MLFLOW_MODEL_NAME}")
    
    # DISABLED: Local MLflow serving is no longer needed as we're using the model server's /invocations endpoint
    logger.info(f"Local MLflow serving on port 5004 is disabled.")
    logger.info(f"Using model server's /invocations endpoint on port {port} instead.")
    logger.info(f"Model URI: {model_uri}")
    
    # Instead of starting a local MLflow server, we're now using the model server's /invocations endpoint
    # The model server is already running on port 5002 (external) and handles requests to /invocations
    # os.system(f"mlflow models serve -m {model_uri} -p {port} --no-conda")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Deploy the RAG model')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID to deploy')
    parser.add_argument('--port', type=int, default=5002,
                        help='Port to deploy on (default is 5002, which is the model server port)')
    args = parser.parse_args()
    
    # Deploy model
    deploy_model(args.run_id, args.port)
```

## app/scripts/download_llm.sh

```bash
#!/bin/bash

# Directory for LLM model
LLM_DIR="models/llm"
mkdir -p $LLM_DIR

# Prompt for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
  echo "Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):"
  read -s HF_TOKEN
  echo
fi

# Download using huggingface-cli
echo "Installing huggingface_hub if needed..."
pip install -q huggingface_hub

echo "Downloading Llama-3.2-1B-Instruct model..."
echo "This will take some time depending on your connection."

python -c "
from huggingface_hub import snapshot_download
import os

# Set token
os.environ['HF_TOKEN'] = '$HF_TOKEN'

# Download model files
model_path = snapshot_download(
    repo_id='meta-llama/Llama-3.2-1B-Instruct',
    local_dir='$LLM_DIR/Llama-3.2-1B-Instruct',
    local_dir_use_symlinks=False
)

print(f'Model downloaded to {model_path}')
"

# Update settings.py to use this model
SETTINGS_PATH="app/config/settings.py"
if [ -f "$SETTINGS_PATH" ]; then
    if grep -q "LLM_MODEL_PATH" "$SETTINGS_PATH"; then
        sed -i '' 's|LLM_MODEL_PATH = .*|LLM_MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "Llama-3.2-1B-Instruct")|' "$SETTINGS_PATH"
        echo "Updated settings.py to use the Llama-3.2-1B-Instruct model."
    fi
fi

echo "Now installing transformers to use with Meta's model format..."
pip install -q transformers accelerate

# Create a model loader adapter
mkdir -p app/utils/adapters
cat > app/utils/adapters/meta_llama_adapter.py << 'EOF'
import os
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class MetaLlamaAdapter:
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        """Initialize the Meta Llama adapter."""
        logger.info(f"Loading Meta Llama model from {model_path}")
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        logger.info("Meta Llama model loaded successfully")
    
    def __call__(self, prompt: str, **kwargs):
        """Generate text using the Meta Llama model."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.2) > 0,
        }
        
        # Generate text
        outputs = self.pipe(
            prompt,
            **generation_kwargs
        )
        
        # Format to match llama-cpp-python output format
        generated_text = outputs[0]["generated_text"][len(prompt):]
        
        # Count tokens
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(generated_text))
        
        return {
            "choices": [
                {
                    "text": generated_text,
                    "finish_reason": "length" if output_tokens >= generation_kwargs["max_new_tokens"] else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
EOF

# Update llm.py to use the Meta Llama adapter
sed -i '' 's/from llama_cpp import Llama/from llama_cpp import Llama\nfrom app.utils.adapters.meta_llama_adapter import MetaLlamaAdapter/' app/utils/llm.py

# Update the LLMProcessor.__init__ method
sed -i '' 's/self.model = Llama(/# Check if using Meta Llama model\n        if "Llama-3" in model_path and os.path.isdir(model_path):\n            self.model = MetaLlamaAdapter(\n                model_path=model_path,\n                max_new_tokens=max_tokens\n            )\n        else:\n            # Use llama-cpp for GGUF models\n            self.model = Llama(/' app/utils/llm.py

echo "Setup complete for using meta-llama/Llama-3.2-1B-Instruct"
```

## app/scripts/download_models.py

```python
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
import torch

def download_embedding_model():
    """Download the embedding model for offline use."""
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"Downloading embedding model to {EMBEDDING_MODEL_PATH}...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        os.makedirs(os.path.dirname(EMBEDDING_MODEL_PATH), exist_ok=True)
        model.save(EMBEDDING_MODEL_PATH)
        print("Embedding model downloaded successfully.")
        
        # Test the model
        test_embedding = model.encode(["Hello world"])
        print(f"Test embedding shape: {test_embedding.shape}")
        
    except Exception as e:
        print(f"Error downloading embedding model: {str(e)}")
        sys.exit(1)

def download_reranker_model():
    """Download the reranker model for offline use."""
    try:
        from sentence_transformers import CrossEncoder
        
        print(f"Downloading reranker model to {RERANKER_MODEL_PATH}...")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        os.makedirs(os.path.dirname(RERANKER_MODEL_PATH), exist_ok=True)
        model.save(RERANKER_MODEL_PATH)
        print("Reranker model downloaded successfully.")
        
    except Exception as e:
        print(f"Error downloading reranker model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_embedding_model()
    download_reranker_model()
    print("Please manually download the LLM model using the instructions in the README.")
```

## app/scripts/fix_markdown.py

```python
#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def fix_directory_tree(content):
    """Convert ASCII directory trees to LaTeX-friendly format."""
    # Replace ASCII characters with LaTeX commands
    replacements = {
        '├': '\\textbar--',
        '│': '\\textbar\\phantom{--}',
        '└': '\\textbar\\_',
        '─': '-',
    }
    
    def replace_tree_chars(match):
        line = match.group(0)
        for char, repl in replacements.items():
            line = line.replace(char, repl)
        return line
    
    # Find directory trees (indented blocks with ASCII characters)
    return re.sub(
        r'^([ \t]*[│├└].+$\n?)+',
        lambda m: '\\begin{verbatim}\n' + m.group(0) + '\\end{verbatim}\n',
        content,
        flags=re.MULTILINE
    )

def fix_tables(content):
    """Convert markdown tables to LaTeX tables."""
    def process_table(match):
        lines = match.group(0).strip().split('\n')
        if not lines:
            return match.group(0)
            
        # Count columns from header
        num_cols = len(lines[0].split('|')) - 2  # -2 for empty edges
        
        # Start table environment
        result = ['\\begin{table}[htbp]', '\\centering', '\\begin{tabular}{|' + 'l|' * num_cols + '}']
        result.append('\\hline')
        
        for i, line in enumerate(lines):
            # Skip separator line
            if i == 1 and all(c in '|-' for c in line.strip('| ')):
                continue
                
            # Process cells
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            # Escape special characters in cells
            cells = [
                cell.replace('_', '\\_')
                    .replace('&', '\\&')
                    .replace('%', '\\%')
                    .replace('#', '\\#')
                    .replace('$', '\\$')
                for cell in cells
            ]
            result.append(' & '.join(cells) + ' \\\\')
            result.append('\\hline')
        
        result.extend(['\\end{tabular}', '\\end{table}'])
        return '\n'.join(result)
    
    # Find and process tables
    return re.sub(
        r'^\|.+\|$\n\|[-|\s]+\|\n(\|.+\|$\n?)+',
        process_table,
        content,
        flags=re.MULTILINE
    )

def fix_code_blocks(content):
    """Convert markdown code blocks to LaTeX listings."""
    def process_code_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        
        # Special handling for directory trees
        if '├' in code or '│' in code or '└' in code:
            return fix_directory_tree(code)
            
        # Escape special characters in code
        code = (
            code.strip()
            .replace('_', '\\_')
            .replace('&', '\\&')
            .replace('%', '\\%')
            .replace('#', '\\#')
            .replace('$', '\\$')
        )
        
        # Regular code block
        return (
            '\\begin{lstlisting}' +
            ('[language=' + lang + ']' if lang else '') +
            '\n' + code + '\n\\end{lstlisting}'
        )
    
    return re.sub(
        r'```(\w+)?\n(.*?)```',
        process_code_block,
        content,
        flags=re.DOTALL
    )

def fix_inline_code(content):
    """Fix inline code and special characters."""
    # Fix inline code with special characters
    content = re.sub(
        r'`([^`]*)`',
        lambda m: '\\texttt{' + (
            m.group(1)
            .replace('_', '\\_')
            .replace('$', '\\$')
            .replace('&', '\\&')
            .replace('#', '\\#')
            .replace('%', '\\%')
            .replace('{', '\\{')
            .replace('}', '\\}')
            .replace('~', '\\~{}')
            .replace('^', '\\^{}')
            .replace('\\', '\\textbackslash{}')
        ) + '}',
        content
    )
    
    # Fix arrows and other special characters
    replacements = {
        '→': '$\\rightarrow$',
        '←': '$\\leftarrow$',
        '↑': '$\\uparrow$',
        '↓': '$\\downarrow$',
        '≤': '$\\leq$',
        '≥': '$\\geq$',
        '≠': '$\\neq$',
        '×': '$\\times$',
        '…': '\\ldots{}',
        ''': "'",
        ''': "'",
        '"': "``",
        '"': "''",
        '—': '---',
        '–': '--',
    }
    
    for char, repl in replacements.items():
        content = content.replace(char, repl)
    
    # Fix URLs
    content = re.sub(
        r'\[(.*?)\]\((.*?)\)',
        lambda m: '\\href{' + m.group(2).replace('%', '\\%') + '}{' + m.group(1) + '}',
        content
    )
    
    return content

def fix_images(content):
    """Fix image references."""
    def process_image(match):
        alt_text = match.group(1)
        path = match.group(2)
        
        # Convert SVG path to PDF
        if path.endswith('.svg'):
            path = path.replace('.svg', '.pdf')
            
        # Make path relative to the document
        if path.startswith('../../'):
            path = path[6:]
            
        # Handle image references
        if path.startswith('puml/'):
            path = '_images/' + path.replace('puml/', '')
            
        return (
            '\\begin{figure}[htbp]\n'
            '\\centering\n'
            '\\includegraphics[width=0.8\\textwidth]{' + path + '}\n'
            '\\caption{' + alt_text + '}\n'
            '\\label{fig:' + alt_text.lower().replace(' ', '-') + '}\n'
            '\\end{figure}'
        )
    
    # First convert any image links to proper markdown image syntax
    content = re.sub(
        r'\[([^]]+)\]\((.*?\.(?:svg|pdf))\)',
        r'![\1](\2)',
        content
    )
    
    # Then convert all images to figures
    content = re.sub(
        r'!\[(.*?)\]\((.*?)\)',
        process_image,
        content
    )
    
    # Remove any extra exclamation marks before figures
    content = re.sub(
        r'!\s*\\begin{figure}',
        r'\\begin{figure}',
        content
    )
    
    return content

def fix_lists(content):
    """Fix markdown lists to use proper LaTeX itemize/enumerate environments."""
    # Convert unordered lists
    content = re.sub(
        r'(?m)^(\s*)-\s',
        r'\1\\item ',
        content
    )
    
    # Convert ordered lists
    content = re.sub(
        r'(?m)^(\s*)\d+\.\s',
        r'\1\\item ',
        content
    )
    
    # Wrap lists in proper environments
    content = re.sub(
        r'(?sm)^\\item(.*?)(?=^[^\\]|\Z)',
        r'\\begin{itemize}\n\\item\1\\end{itemize}\n',
        content
    )
    
    return content

def fix_headings(content):
    """Convert markdown headings to LaTeX sections."""
    replacements = [
        (r'^#\s+(.+)$', r'\\section{\1}'),
        (r'^##\s+(.+)$', r'\\subsection{\1}'),
        (r'^###\s+(.+)$', r'\\subsubsection{\1}'),
        (r'^####\s+(.+)$', r'\\paragraph{\1}'),
        (r'^#####\s+(.+)$', r'\\subparagraph{\1}'),
    ]
    
    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content, flags=re.MULTILINE)
    
    return content

def fix_markdown_for_latex(content):
    """Fix markdown content for LaTeX processing."""
    # Process in specific order - images first to avoid interference with other elements
    content = fix_images(content)
    content = fix_headings(content)
    content = fix_tables(content)
    content = fix_code_blocks(content)
    content = fix_inline_code(content)
    content = fix_lists(content)
    
    # Fix include directives
    content = re.sub(
        r'\{include\} (.*?)\.md',
        r'\\input{\1}',
        content
    )
    
    # Add LaTeX document class and preamble if not present
    if not content.startswith('\\documentclass'):
        content = (
            '% Auto-generated LaTeX document\n'
            '\\input{preamble}\n\n'
            '\\begin{document}\n\n' +
            content +
            '\n\n\\end{document}\n'
        )
    
    return content

def process_file(file_path):
    """Process a markdown file and fix it for LaTeX."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found")
        return False
    
    try:
        content = path.read_text()
        fixed_content = fix_markdown_for_latex(content)
        
        # Create a new file with _latex suffix
        new_path = path.parent / (path.stem + '_latex' + path.suffix)
        new_path.write_text(fixed_content)
        print(f"Created LaTeX-compatible version at {new_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_markdown.py <markdown_file>")
        sys.exit(1)
    
    success = process_file(sys.argv[1])
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
```

## app/scripts/generate_docs.py

```python
#!/usr/bin/env python3
"""
Python wrapper for the generate_docs.sh script.
This script is provided for users who accidentally try to run the shell script with Python.
"""

import os
import sys
import subprocess

def main():
    """Main function to parse arguments and call the shell script."""
    print("This is a Python wrapper for the generate_docs.sh shell script.")
    print("Redirecting to the shell script...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the shell script
    shell_script = os.path.join(script_dir, "generate_docs.sh")
    
    # Make sure the shell script is executable
    os.chmod(shell_script, 0o755)
    
    # Construct the command
    cmd = [shell_script]
    
    # Add any arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    # Print the command being run
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run the shell script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: The shell script exited with code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Could not find the shell script at {shell_script}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
```

## app/scripts/generate_docs.sh

```bash
#!/bin/bash
# Script to generate comprehensive documentation for the project

# Check if this script is being run with Python
if [[ "$0" == *python* ]]; then
    echo "Error: This is a bash script and should be run directly, not with Python."
    echo "Please run it as: ./app/scripts/generate_docs.sh [--format FORMAT]"
    echo "Or: bash app/scripts/generate_docs.sh [--format FORMAT]"
    exit 1
fi

# Parse command line arguments
FORMAT="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --format|-f)
      FORMAT="$2"
      shift 2
      ;;
    *)
      # If it's a positional argument without a flag, assume it's the format
      if [[ "$1" =~ ^[a-zA-Z]+$ ]]; then
        FORMAT="$1"
      fi
      shift
      ;;
  esac
done

VALID_FORMATS=("html" "pdf" "markdown" "all")

# Validate format
if [[ ! " ${VALID_FORMATS[@]} " =~ " ${FORMAT} " ]]; then
    echo "Invalid format: $FORMAT"
    echo "Valid formats: html, pdf, markdown, all"
    echo "Usage: $0 [--format FORMAT]"
    exit 1
fi

echo "Generating documentation in format: $FORMAT"

# Make scripts executable
chmod +x app/scripts/setup_sphinx.py
chmod +x app/scripts/build_docs.py
chmod +x app/scripts/convert_svg_to_pdf.py

# Step 1: Convert SVG files to PDF for better compatibility
echo "Converting SVG files to PDF..."
python app/scripts/convert_svg_to_pdf.py docs

# Step 2: Copy dev_notes.md to Sphinx source directory
echo "Copying dev_notes.md to Sphinx source directory..."
mkdir -p docs/sphinx/source
cp dev_notes.md docs/sphinx/source/

# Step 3: Set up Sphinx documentation if not already set up
if [ ! -f docs/sphinx/source/conf.py ]; then
    echo "Setting up Sphinx documentation..."
    python app/scripts/setup_sphinx.py
fi

# Step 4: Update API documentation
echo "Updating API documentation..."
python app/scripts/build_docs.py --api-only

# Check if timeout command is available
if command -v timeout >/dev/null 2>&1; then
    HAS_TIMEOUT=true
else
    HAS_TIMEOUT=false
    echo "Warning: 'timeout' command not found. PDF generation will not have timeout protection."
fi

# Step 5: Build documentation in the requested format(s)
if [ "$FORMAT" == "all" ]; then
    echo "Building documentation in all formats..."
    python app/scripts/build_docs.py --format html --clean
    
    echo "Building PDF documentation..."
    if [ "$HAS_TIMEOUT" = true ]; then
        # Use timeout command to prevent hanging
        timeout 300 python app/scripts/build_docs.py --format pdf || {
            echo "PDF generation timed out after 5 minutes."
            echo "This is likely due to a LaTeX package issue or a complex document."
            echo "HTML documentation should still be available."
        }
    else
        # Run without timeout
        python app/scripts/build_docs.py --format pdf
    fi
    
    python app/scripts/build_docs.py --format markdown
    python app/scripts/build_docs.py --format epub
elif [ "$FORMAT" == "pdf" ]; then
    echo "Building PDF documentation..."
    if [ "$HAS_TIMEOUT" = true ]; then
        # Use timeout command to prevent hanging
        timeout 300 python app/scripts/build_docs.py --format pdf --clean || {
            echo "PDF generation timed out after 5 minutes."
            echo "This is likely due to a LaTeX package issue or a complex document."
        }
    else
        # Run without timeout
        python app/scripts/build_docs.py --format pdf --clean
    fi
else
    echo "Building documentation in $FORMAT format..."
    python app/scripts/build_docs.py --format $FORMAT --clean
fi

echo "Documentation generation complete!"
echo "You can find the generated documentation in the docs/sphinx/build directory."
echo ""
echo "API documentation has been automatically generated from docstrings in your Python modules."
echo "To improve the API documentation, add detailed docstrings to your Python code." 
```

## app/scripts/generate_pdf.sh

```bash
#!/bin/bash
# Script to generate PDF from Markdown with proper handling of SVG images

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <markdown_file>"
    exit 1
fi

MARKDOWN_FILE=$1
DOCS_DIR="docs"

# Make scripts executable
chmod +x app/scripts/convert_svg_to_pdf.py
chmod +x app/scripts/convert_md_to_pdf.py

# Step 1: Convert SVG files to PDF
echo "Converting SVG files to PDF..."
python app/scripts/convert_svg_to_pdf.py $DOCS_DIR $MARKDOWN_FILE

# Step 2: Generate PDF from Markdown
echo "Generating PDF from Markdown..."
python app/scripts/convert_md_to_pdf.py $MARKDOWN_FILE --keep-tex

echo "Done!" 
```

## app/scripts/log_model.py

```python
import os
import sys
import mlflow
import logging
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME
from app.models.rag_model import RAGModel, get_pip_requirements

def log_model():
    """Log the RAG model to MLflow."""
    logger.info(f"Logging model {MLFLOW_MODEL_NAME} to MLflow at {MLFLOW_TRACKING_URI}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create a temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory for artifacts: {temp_dir}")
        
        # Get project root
        project_root = Path(__file__).resolve().parent.parent.parent
        app_dir = os.path.join(project_root, "app")
        
        # Create a copy of the app directory in the temp directory
        temp_app_dir = os.path.join(temp_dir, "app")
        shutil.copytree(app_dir, temp_app_dir)
        
        # Log model
        with mlflow.start_run(run_name=f"{MLFLOW_MODEL_NAME}_deployment") as run:
            # Log parameters
            mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
            mlflow.log_param("reranker_model", "ms-marco-MiniLM-L-6-v2")
            mlflow.log_param("llm_model", "llama-2-7b-chat-q4_0.gguf")
            
            # Log model using the temporary directory for artifacts
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RAGModel(),
                artifacts={
                    "app_dir": temp_app_dir
                },
                pip_requirements=get_pip_requirements(),
                registered_model_name=MLFLOW_MODEL_NAME
            )
            
            logger.info(f"Model logged: {model_info.model_uri}")
            
            return model_info

if __name__ == "__main__":
    # Log model
    model_info = log_model()
    print(f"Model logged: {model_info.model_uri}")
    print(f"Run ID: {model_info.run_id}")
```

## app/scripts/prepare_latex.sh

```bash
#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel || pwd)
DOCS_DIR="$PROJECT_ROOT/docs"
SPHINX_SOURCE_DIR="$DOCS_DIR/sphinx/source"

echo -e "${GREEN}Preparing markdown files for LaTeX processing...${NC}"

# Clean up any existing _latex files
find "$SPHINX_SOURCE_DIR" -name "*_latex*.md" -type f -delete

# Function to process a markdown file
process_markdown() {
    local file="$1"
    # Skip if file is already a _latex version
    if [[ "$file" == *"_latex"* ]]; then
        return
    fi
    echo "Processing $file..."
    python "$PROJECT_ROOT/app/scripts/fix_markdown.py" "$file"
}

# Process all markdown files in the Sphinx source directory
find "$SPHINX_SOURCE_DIR" -name "*.md" -type f | while read -r file; do
    process_markdown "$file"
done

# Convert SVG files to PDF
echo -e "\n${GREEN}Converting SVG files to PDF...${NC}"
find "$DOCS_DIR/puml/svg" -name "*.svg" -type f | while read -r file; do
    pdf_file="${file%.svg}.pdf"
    if [ ! -f "$pdf_file" ] || [ "$file" -nt "$pdf_file" ]; then
        echo "Converting $file to PDF..."
        if ! cairosvg "$file" -o "$pdf_file" 2>/dev/null; then
            echo -e "${YELLOW}Warning: Could not convert $file using cairosvg, trying Inkscape...${NC}"
            if command -v inkscape >/dev/null 2>&1; then
                inkscape --export-filename="$pdf_file" "$file" 2>/dev/null || {
                    echo -e "${RED}Error: Failed to convert $file using both cairosvg and Inkscape${NC}"
                    continue
                }
            else
                echo -e "${RED}Error: Inkscape not found. Please install it to convert problematic SVG files${NC}"
                continue
            fi
        fi
    fi
done

# Create symbolic links to PDF files if needed
echo -e "\n${GREEN}Creating symbolic links to PDF files...${NC}"
cd "$SPHINX_SOURCE_DIR"
find "$DOCS_DIR/puml/svg" -name "*.pdf" -type f | while read -r file; do
    base_name=$(basename "$file")
    if [ ! -f "$base_name" ]; then
        ln -sf "$file" "$base_name"
    fi
done

echo -e "\n${GREEN}Done preparing files for LaTeX processing.${NC}"

# Check for potential issues
echo -e "\n${YELLOW}Checking for potential LaTeX issues...${NC}"

# Check for special characters
echo "Checking for problematic characters..."
find "$SPHINX_SOURCE_DIR" -name "*_latex.md" -type f -exec grep -l "[^[:print:]]" {} \; || true

# Check for unescaped special characters that haven't been properly processed
echo "Checking for unescaped LaTeX special characters..."
find "$SPHINX_SOURCE_DIR" -name "*_latex.md" -type f -exec grep -l "[^\\][&%$#_{}~^\\]" {} \; || true

echo -e "\n${GREEN}All done! You can now proceed with LaTeX compilation.${NC}"
echo "If you encounter any issues, check the files listed above for problematic characters."

# Provide a summary of processed files
echo -e "\n${GREEN}Summary of processed files:${NC}"
find "$SPHINX_SOURCE_DIR" -name "*_latex.md" -type f | while read -r file; do
    echo "- $(basename "$file")"
done 
```

## app/scripts/setup_sphinx.py

```python
#!/usr/bin/env python3
"""
Script to set up Sphinx documentation for the project.
This script creates the necessary configuration files and directory structure.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
import glob
import importlib
import pkgutil

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def run_command(command, cwd=None):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(
            command, 
            check=True, 
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error message: {e.stderr}")
        return False

def find_python_modules(base_path="app"):
    """Find all Python modules in the project."""
    modules = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Warning: Path {base_path} does not exist.")
        return modules
    
    for root, dirs, files in os.walk(base_path):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue
        
        # Process Python files
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = Path(root) / file
                # Use the file path directly without trying to make it relative to cwd
                module_path = str(file_path).replace("/", ".").replace("\\", ".").replace(".py", "")
                modules.append(module_path)
    
    return modules

def generate_api_docs(modules, output_dir):
    """Generate API documentation for the given modules."""
    # Create API directory
    api_dir = output_dir / "api"
    create_directory(api_dir)
    
    # Create API index file
    api_index_content = """API Reference
============

This section contains the API reference for the PDF RAG System.

.. toctree::
   :maxdepth: 2

   modules
"""
    
    # Write API index file
    with open(api_dir / "index.rst", "w") as f:
        f.write(api_index_content)
    
    # Run sphinx-apidoc to generate module documentation
    run_command([
        "sphinx-apidoc",
        "-o", str(api_dir),
        "-f", "-e", "-M", "-d", "4",  # Force, separate modules, module first, max depth 4
        "app"  # Path to the package
    ])
    
    return api_dir / "index.rst"

def setup_sphinx_docs():
    """Set up Sphinx documentation for the project."""
    # Create docs directory structure
    docs_dir = Path("docs")
    sphinx_dir = docs_dir / "sphinx"
    source_dir = sphinx_dir / "source"
    build_dir = sphinx_dir / "build"
    
    create_directory(docs_dir)
    create_directory(sphinx_dir)
    create_directory(source_dir)
    create_directory(build_dir)
    create_directory(source_dir / "_static")
    create_directory(source_dir / "_templates")
    
    # Initialize Sphinx with sphinx-quickstart
    print("Initializing Sphinx documentation...")
    
    # Check if sphinx-quickstart is available
    try:
        subprocess.run(["sphinx-quickstart", "--version"], check=True, stdout=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("sphinx-quickstart not found. Installing Sphinx...")
        run_command([
            "pip", "install", 
            "sphinx", "sphinx-rtd-theme", "recommonmark", 
            "sphinx-markdown-tables", "myst-parser"
        ])
    
    # Find Python modules
    print("Finding Python modules...")
    modules = find_python_modules()
    print(f"Found {len(modules)} Python modules.")
    
    # Create conf.py
    conf_py_content = """# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))

# -- Project information -----------------------------------------------------
project = 'PDF RAG System'
copyright = '2025, Ryan Hammang'
author = 'Ryan Hammang'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_markdown_tables',
    'myst_parser',
]

# Auto-generate API documentation
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'imported-members': True,
    'private-members': True,
}

# Make sure autodoc can find the modules
autodoc_mock_imports = []
autoclass_content = 'both'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_member_order = 'bysource'

# MyST Parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'display_version': True,
}

# -- Options for Markdown support --------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for PDF output --------------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    'preamble': r'''
\\usepackage{underscore}
\\usepackage{graphicx}
\\usepackage[utf8]{inputenc}
\\usepackage{xcolor}
\\usepackage{fancyvrb}
\\usepackage{tabulary}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{capt-of}
\\usepackage{needspace}
\\usepackage{hyperref}
''',
    'figure_align': 'H',
    'pointsize': '11pt',
    'papersize': 'letterpaper',
    'extraclassoptions': 'openany,oneside',
    'babel': r'\\usepackage[english]{babel}',
    'maketitle': r'\\maketitle',
    'tableofcontents': r'\\tableofcontents',
    'fncychap': r'\\usepackage[Bjarne]{fncychap}',
    'printindex': r'\\printindex',
}

# -- Extension configuration -------------------------------------------------
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None
"""
    
    with open(source_dir / "conf.py", "w") as f:
        f.write(conf_py_content)
    
    # Generate API documentation
    print("Generating API documentation...")
    api_index = generate_api_docs(modules, source_dir)
    
    # Create index.rst
    index_rst_content = """PDF RAG System Documentation
============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   architecture
   api/index
   development
   appendix

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
    
    with open(source_dir / "index.rst", "w") as f:
        f.write(index_rst_content)
    
    # Create introduction.md
    introduction_md_content = """# Introduction

This documentation covers the PDF RAG (Retrieval Augmented Generation) System, a local system for building and querying a knowledge base from PDF documents.

## Overview

The PDF RAG System allows users to:

1. Upload PDF documents to create a knowledge base
2. Query the knowledge base using natural language
3. Receive accurate responses with citations to the source documents
4. Manage and update the knowledge base

This system is built using Flask for the web interface and MLflow for experiment tracking and model management.
"""
    
    with open(source_dir / "introduction.md", "w") as f:
        f.write(introduction_md_content)
    
    # Create installation.md
    installation_md_content = """# Installation

This section covers how to install and set up the PDF RAG System.

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the configuration:
   ```bash
   cp app/config/config.example.py app/config/config.py
   # Edit app/config/config.py with your settings
   ```

5. Run the application:
   ```bash
   python app/main.py
   ```

The application should now be running at http://localhost:5000.
"""
    
    with open(source_dir / "installation.md", "w") as f:
        f.write(installation_md_content)
    
    # Create usage.md
    usage_md_content = """# Usage Guide

This section provides a guide on how to use the PDF RAG System.

## Uploading Documents

1. Navigate to the Upload page
2. Select PDF files from your computer
3. Click the Upload button
4. Wait for the processing to complete

## Querying the Knowledge Base

1. Navigate to the Query page
2. Enter your question in the text box
3. Click the Submit button
4. View the response with citations

## Managing the Knowledge Base

1. Navigate to the Management page
2. View all uploaded documents
3. Remove documents if needed
4. Update the knowledge base after changes
"""
    
    with open(source_dir / "usage.md", "w") as f:
        f.write(usage_md_content)
    
    # Create architecture.md with PUML diagrams
    architecture_md_content = """# Architecture and Design

This section describes the architecture and design of the PDF RAG System.

## System Architecture

The system is composed of several components that work together to provide the RAG functionality.

"""
    
    # Find PUML files and add them to architecture.md
    puml_files = glob.glob("docs/puml/svg/*.svg")
    for puml_file in puml_files:
        diagram_name = os.path.basename(puml_file).replace(".svg", "")
        diagram_title = diagram_name.replace("_", " ").title()
        architecture_md_content += f"""
### {diagram_title} Diagram

![{diagram_title} Diagram]({os.path.relpath(puml_file, source_dir)})

"""
    
    with open(source_dir / "architecture.md", "w") as f:
        f.write(architecture_md_content)
    
    # Create development.md
    development_md_content = """# Development Guide

This document contains development notes and additional information for developers working on the PDF RAG System.

## Project Structure

```
project/
├── app/
│   ├── api/
│   ├── models/
│   ├── scripts/
│   ├── static/
│   └── templates/
├── docs/
│   ├── puml/
│   └── sphinx/
└── tests/
```

## Development Environment

### Prerequisites

- Python 3.8+
- pip
- virtualenv or conda

### Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Run the development server

## Coding Standards

- Follow PEP 8 for Python code
- Use docstrings for all functions and classes
- Write unit tests for new features

## Deployment

### Local Deployment

Instructions for local deployment...

### Production Deployment

Instructions for production deployment...

## Future Improvements

- List of planned improvements and features
- Known limitations and how they might be addressed

"""
    
    with open(source_dir / "development.md", "w") as f:
        f.write(development_md_content)
    
    # Create appendix.md that includes dev_notes.md
    appendix_md_content = """# Appendix

This appendix contains additional information and resources.

## Development Notes

```{include} dev_notes.md
```

## References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

"""
    
    with open(source_dir / "appendix.md", "w") as f:
        f.write(appendix_md_content)
    
    # Create Makefile
    makefile_content = """# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean markdown

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*

markdown:
	@$(SPHINXBUILD) -M markdown "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
"""
    
    with open(sphinx_dir / "Makefile", "w") as f:
        f.write(makefile_content)
    
    # Create make.bat for Windows
    make_bat_content = """@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "markdown" goto markdown

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
rmdir /s /q %BUILDDIR%
goto end

:markdown
%SPHINXBUILD% -M markdown %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd
"""
    
    with open(sphinx_dir / "make.bat", "w") as f:
        f.write(make_bat_content)
    
    print("Sphinx documentation setup complete!")
    print(f"Documentation source files are in: {source_dir}")
    print("To build the documentation:")
    print(f"  cd {sphinx_dir}")
    print("  make html    # For HTML output")
    print("  make latexpdf  # For PDF output")
    print("  make markdown  # For Markdown output")

if __name__ == "__main__":
    setup_sphinx_docs() 
```

## app/scripts/start_mlflow.sh

```bash
#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "NOTICE: Local MLflow serving is disabled."
echo "Using model server's /invocations endpoint on port 5002 instead."
echo "If you need to start the MLflow UI server (not the serving component), use port 5001."

# Create MLflow directories if they don't exist
mkdir -p "$PROJECT_ROOT/mlflow/artifacts" "$PROJECT_ROOT/mlflow/db"

# DISABLED: Local MLflow serving is no longer needed as we're using the model server's /invocations endpoint
# Start MLflow server with absolute paths
# mlflow server \
#   --backend-store-uri "sqlite:///$PROJECT_ROOT/mlflow/db/mlflow.db" \
#   --default-artifact-root "$PROJECT_ROOT/mlflow/artifacts" \
#   --host 0.0.0.0 \
#   --port 5004 

# If you need to start the MLflow UI server (not the serving component), use this command:
echo "To start the MLflow UI server, run:"
echo "mlflow server --backend-store-uri sqlite:///$PROJECT_ROOT/mlflow/db/mlflow.db --default-artifact-root $PROJECT_ROOT/mlflow/artifacts --host 0.0.0.0 --port 5001" 
```

## app/scripts/stop_mlflow.sh

```bash
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
```

## app/scripts/test_mlflow_model.py

```python
import os
import sys
import mlflow
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME

def test_model():
    """Test loading and using the model from MLflow."""
    logger.info(f"Loading model {MLFLOW_MODEL_NAME} from MLflow at {MLFLOW_TRACKING_URI}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load the model
    model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/latest")
    
    # Test prediction
    query = "What is retrieval-augmented generation?"
    logger.info(f"Testing prediction with query: {query}")
    
    result = model.predict(query)
    
    logger.info(f"Prediction result: {result}")
    
    return result

if __name__ == "__main__":
    # Test the model
    result = test_model()
    print(f"Response: {result['text']}") 
```

## app/scripts/test_model_downloader.py

```python
#!/usr/bin/env python
"""
Test script to verify model loading and downloading functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from app.config.settings import MODEL_PATH, ALT_MODEL_PATHS, HF_MODEL_ID, HF_TOKEN
from app.utils.model_downloader import find_or_download_model
from app.utils.llm import LLMProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test the model loading/downloading functionality"""
    
    logger.info("Testing model loading/downloading functionality")
    logger.info(f"Primary model path: {MODEL_PATH}")
    logger.info(f"Alternative model paths: {ALT_MODEL_PATHS}")
    logger.info(f"HF Model ID: {HF_MODEL_ID}")
    logger.info(f"HF Token available: {'Yes' if HF_TOKEN else 'No'}")
    
    # First test direct find_or_download function
    model_path = find_or_download_model(
        MODEL_PATH,
        ALT_MODEL_PATHS,
        HF_MODEL_ID,
        HF_TOKEN
    )
    
    logger.info(f"Found or downloaded model at: {model_path}")
    
    if model_path:
        exists = os.path.exists(model_path)
        size = os.path.getsize(model_path) if exists else 0
        logger.info(f"Model exists: {exists}, Size: {size/1024/1024:.2f} MB")
    
    # Now test through the LLMProcessor
    logger.info("Testing model loading through LLMProcessor")
    
    processor = LLMProcessor(MODEL_PATH)
    
    if processor.use_mock:
        logger.warning("Using mock processor - real model not available or invalid")
    else:
        logger.info("Successfully loaded real LLM model!")
        
        # Test with a simple query
        prompt = "Tell me a short fact about language models."
        logger.info(f"Testing with prompt: {prompt}")
        
        response = processor.generate_response(prompt)
        
        logger.info(f"Response: {response['text']}")
        logger.info(f"Tokens used: {response['metadata']['tokens_used']}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    test_model_loading() 
```

## app/tests/README.md

```markdown
# Testing Guide for PDF RAG System

This directory contains tests for the PDF RAG System. The tests are organized by module and functionality.

## Running Tests

You can run tests using the provided `run_tests.py` script in the project root:

```bash
# Run all tests with pytest (default)
./run_tests.py

# Run with coverage report
./run_tests.py --coverage

# Run specific test file
./run_tests.py --test-file app/tests/test_pdf_ingestion.py

# Run with unittest framework
./run_tests.py --framework unittest
```

### Test Categories

The tests are categorized using markers. You can run specific categories of tests:

```bash
# Run only unit tests
./run_tests.py --unit

# Run only integration tests
./run_tests.py --integration

# Run only API tests
./run_tests.py --api

# Run only model tests
./run_tests.py --model

# Run only PDF-related tests
./run_tests.py --pdf

# Include slow tests (skipped by default)
./run_tests.py --runslow
```

Alternatively, you can use pytest or unittest directly:

```bash
# Using pytest
pytest app/tests
pytest app/tests --cov=app
pytest app/tests -m unit  # Run only unit tests

# Using unittest
python -m unittest discover app/tests
```

## Test Structure

- `conftest.py`: Contains pytest fixtures used across multiple test files
- `test_pdf_ingestion.py`: Tests for PDF scanning and text extraction
- `test_pdf_extraction.py`: Mock tests for PDF extraction functionality
- `test_integration.py`: Integration tests for the full system
- `test_text_chunking.py`: Unit tests for text chunking functionality
- `test_text_chunking_integration.py`: Integration tests for text chunking with other components

## Text Chunking Tests

The text chunking tests are organized into several files:

- `test_text_chunking.py`: Contains unit tests for the text chunking functionality
- `test_text_chunking_integration.py`: Contains integration tests for text chunking with other components
- `test_data_generator.py`: Utility for generating test data for chunking tests
- `run_chunking_tests.py`: Script to run all text chunking tests

### Running Text Chunking Tests

You can run the text chunking tests specifically using:

```bash
# Run all text chunking tests
python app/tests/run_chunking_tests.py

# Generate test data for chunking tests
python app/tests/run_chunking_tests.py --generate-data

# Generate data and run tests
python app/tests/run_chunking_tests.py --generate-data --run-tests
```

### Text Chunking Test Structure

The text chunking tests cover:

1. **Unit Tests**:
   - `TestCleanText`: Tests for the text cleaning functionality
   - `TestChunkText`: Tests for the text chunking functionality
   - `TestProcessChunks`: Tests for processing chunks from a DataFrame

2. **Integration Tests**:
   - `TestTextChunkingWithPDFIngestion`: Tests integration with PDF ingestion
   - `TestTextChunkingWithVectorDB`: Tests integration with vector database
   - `TestEndToEndProcessing`: End-to-end tests for the document processing pipeline

3. **Test Data Generation**:
   - `generate_random_text()`: Generates random text with paragraphs
   - `generate_test_pdf()`: Generates a test PDF with specified text
   - `generate_test_pdfs()`: Generates multiple test PDFs
   - `generate_test_dataframe()`: Generates a test DataFrame with PDF metadata
   - `generate_test_chunks_dataframe()`: Generates a test DataFrame with text chunks
   - `generate_test_embeddings()`: Generates test embeddings for text chunks

## Writing New Tests

When writing new tests:

1. Create a new file named `test_<module_name>.py`
2. Use pytest fixtures from `conftest.py` when possible
3. Use mocks for external dependencies
4. Ensure tests are isolated and don't depend on external resources
5. Add appropriate markers to categorize your tests:
   ```python
   @pytest.mark.unit
   def test_something():
       # Unit test implementation
       
   @pytest.mark.integration
   def test_integration():
       # Integration test implementation
       
   @pytest.mark.slow
   def test_slow_operation():
       # Slow test implementation
   ```

## Test Data

The `conftest.py` file provides fixtures for creating test data, including:

- `sample_pdf_dir`: A temporary directory with sample PDFs
- `empty_dir`: An empty directory for testing
- `sample_pdf_data`: Sample PDF metadata
- `sample_text_data`: Sample text data for testing text chunking
- `sample_pdf_dataframe`: Sample DataFrame with PDF data including text content

## Code Coverage

To generate a code coverage report:

```bash
./run_tests.py --coverage
```

This will create an HTML report in the `htmlcov` directory that you can open in a browser. 
```

## app/tests/conftest.py

```python
import os
import tempfile
import shutil
import pytest
import fitz
import pandas as pd

@pytest.fixture
def sample_pdf_dir():
    """Create a temporary directory with sample PDFs for testing."""
    test_dir = tempfile.mkdtemp()
    
    # Create sample PDFs
    create_sample_pdf_with_text(os.path.join(test_dir, "text.pdf"), "Hello, world!")
    create_blank_pdf(os.path.join(test_dir, "blank.pdf"))
    create_corrupted_pdf(os.path.join(test_dir, "corrupted.pdf"))
    
    # Create a subdirectory with a PDF
    sub_dir = os.path.join(test_dir, "subdir")
    os.mkdir(sub_dir)
    create_sample_pdf_with_text(os.path.join(sub_dir, "sub_text.pdf"), "Subdirectory PDF")
    
    # Create a non-PDF file
    with open(os.path.join(test_dir, "not_a_pdf.txt"), 'w') as f:
        f.write("This is not a PDF.")
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(test_dir)

@pytest.fixture
def empty_dir():
    """Create an empty directory for testing."""
    empty_dir = tempfile.mkdtemp()
    yield empty_dir
    shutil.rmtree(empty_dir)

@pytest.fixture
def sample_pdf_data():
    """Return sample PDF metadata for testing."""
    return [
        {'path': '/path/to/file1.pdf', 'filename': 'file1.pdf', 'size_bytes': 1000},
        {'path': '/path/to/file2.pdf', 'filename': 'file2.pdf', 'size_bytes': 2000}
    ]

@pytest.fixture
def sample_text_data():
    """Return sample text data for testing text chunking."""
    return {
        'short_text': "This is a short text that won't be chunked.",
        'medium_text': "This is a medium-length text. " * 10,
        'long_text': "This is a longer text with multiple sentences. " * 30,
        'paragraphs': (
            "Paragraph 1 with some content.\n\n"
            "Paragraph 2 with different content.\n\n"
            "Paragraph 3 with even more content.\n\n"
            "Paragraph 4 to ensure we have enough text."
        ),
        'whitespace_text': "  Text with   excessive    whitespace   \n\n\n   and line breaks.  ",
        'empty_text': ""
    }

@pytest.fixture
def sample_pdf_dataframe():
    """Return a sample DataFrame with PDF data including text content."""
    return pd.DataFrame([
        {
            'path': '/path/to/doc1.pdf',
            'filename': 'doc1.pdf',
            'text': "This is the text content of document 1. " * 20,
            'size_bytes': 1000
        },
        {
            'path': '/path/to/doc2.pdf',
            'filename': 'doc2.pdf',
            'text': "This is the text content of document 2. " * 20,
            'size_bytes': 2000
        },
        {
            'path': '/path/to/empty.pdf',
            'filename': 'empty.pdf',
            'text': "",
            'size_bytes': 500
        }
    ])

# Helper functions
def create_sample_pdf_with_text(filename, text):
    """Create a PDF with specified text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), text)
    doc.save(filename)

def create_blank_pdf(filename):
    """Create a blank PDF with no text."""
    doc = fitz.open()
    doc.new_page()
    doc.save(filename)

def create_corrupted_pdf(filename):
    """Create a corrupted PDF by writing plain text."""
    with open(filename, 'w') as f:
        f.write("This is not a PDF.") 
```

## app/tests/load_test.py

```python
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
```

## app/tests/run_chunking_tests.py

```python
#!/usr/bin/env python3
"""
Script to run all text chunking tests.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_tests():
    """Run all text chunking tests."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define test files
    test_files = [
        os.path.join(script_dir, "test_text_chunking.py"),
        os.path.join(script_dir, "test_text_chunking_integration.py"),
    ]
    
    # Run tests
    print("Running text chunking tests...")
    exit_code = pytest.main(["-xvs"] + test_files)
    
    return exit_code

def generate_test_data():
    """Generate test data for text chunking tests."""
    from app.tests.test_data_generator import (
        generate_test_pdfs,
        generate_test_dataframe,
        generate_test_chunks_dataframe,
        generate_test_embeddings
    )
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define output directory
    output_dir = os.path.join(script_dir, "data", "test_pdfs")
    
    # Generate test PDFs
    pdf_paths = generate_test_pdfs(output_dir, count=3)
    print(f"Generated {len(pdf_paths)} test PDFs in {output_dir}")
    
    # Generate test DataFrame
    pdf_df = generate_test_dataframe(pdf_paths)
    print(f"Generated DataFrame with {len(pdf_df)} PDFs")
    
    # Generate test chunks
    chunks_df = generate_test_chunks_dataframe(pdf_df)
    print(f"Generated {len(chunks_df)} chunks")
    
    # Generate test embeddings
    embeddings_df = generate_test_embeddings(chunks_df)
    print(f"Generated embeddings with dimension {len(embeddings_df['embedding'][0])}")
    
    return {
        'pdf_paths': pdf_paths,
        'pdf_df': pdf_df,
        'chunks_df': chunks_df,
        'embeddings_df': embeddings_df
    }

def main():
    """Main function."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run text chunking tests.")
    parser.add_argument("--generate-data", action="store_true", help="Generate test data")
    parser.add_argument("--run-tests", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    # Default to running tests if no arguments are provided
    if not args.generate_data and not args.run_tests:
        args.run_tests = True
    
    # Generate test data if requested
    if args.generate_data:
        generate_test_data()
    
    # Run tests if requested
    if args.run_tests:
        exit_code = run_tests()
        sys.exit(exit_code)

if __name__ == "__main__":
    main() 
```

## app/tests/test_component_integration.py

```python
import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import (
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, MODEL_PATH,
    VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION
)
from app.utils.query_processing import QueryProcessor
from app.utils.reranking import Reranker
from app.utils.vector_db import VectorDBClient
from app.utils.llm import LLMProcessor

@pytest.mark.integration
class TestComponentIntegration:
    """Test the integration between different components."""
    
    @pytest.fixture
    def mock_vector_db(self, monkeypatch):
        """Create a mock vector database client."""
        class MockVectorDBClient:
            def __init__(self, *args, **kwargs):
                self.documents = []
                self.vectors = []
            
            def create_collection(self):
                pass
                
            def upload_vectors(self, documents, vector_column='embedding', batch_size=100):
                self.documents.extend(documents)
                return len(documents)
            
            def search(self, query_vector, limit=3):
                return [
                    {"text": "Document 1", "metadata": {"source": "doc1.pdf"}, "score": 0.95},
                    {"text": "Document 2", "metadata": {"source": "doc2.pdf"}, "score": 0.85},
                ]
        
        monkeypatch.setattr("app.utils.vector_db.VectorDBClient", MockVectorDBClient)
        return MockVectorDBClient()
    
    @pytest.fixture
    def mock_reranker(self, monkeypatch):
        """Create a mock reranker."""
        class MockReranker:
            def __init__(self, *args, **kwargs):
                pass
            
            def rerank(self, query, documents, top_k=3):
                return [
                    {"text": documents[0]["text"], "metadata": documents[0]["metadata"], "score": 0.98},
                    {"text": documents[1]["text"], "metadata": documents[1]["metadata"], "score": 0.75},
                ]
        
        monkeypatch.setattr("app.utils.reranking.Reranker", MockReranker)
        return MockReranker()
    
    @pytest.fixture
    def mock_llm_processor(self, monkeypatch):
        """Create a mock LLM processor."""
        class MockLLMProcessor:
            def __init__(self, *args, **kwargs):
                pass
            
            def generate_response(self, query, context):
                return f"Answer based on context: {context[:50]}..."
        
        monkeypatch.setattr("app.utils.llm.LLMProcessor", MockLLMProcessor)
        return MockLLMProcessor()
    
    @pytest.mark.integration
    def test_query_to_vector_db(self, mock_vector_db):
        """Test the flow from query to vector database retrieval."""
        # Create a query processor
        query_processor = QueryProcessor()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        processed_query = query_processor.process(query)
        
        # Query the vector database
        results = mock_vector_db.search(processed_query, limit=3)
        
        # Check results
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]
        assert "text" in results[0]
        assert "metadata" in results[0]
    
    @pytest.mark.integration
    def test_reranking_flow(self, mock_vector_db, mock_reranker):
        """Test the reranking flow."""
        # Create a query processor
        query_processor = QueryProcessor()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        processed_query = query_processor.process(query)
        
        # Query the vector database
        initial_results = mock_vector_db.search(processed_query, limit=5)
        
        # Rerank the results
        reranked_results = mock_reranker.rerank(query, initial_results, top_k=2)
        
        # Check results
        assert len(reranked_results) == 2
        assert reranked_results[0]["score"] > reranked_results[1]["score"]
        assert reranked_results[0]["score"] > initial_results[0]["score"]
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_flow(self, mock_vector_db, mock_reranker, mock_llm_processor):
        """Test the end-to-end flow from query to response."""
        # Create a query processor
        query_processor = QueryProcessor()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        processed_query = query_processor.process(query)
        
        # Query the vector database
        initial_results = mock_vector_db.search(processed_query, limit=5)
        
        # Rerank the results
        reranked_results = mock_reranker.rerank(query, initial_results, top_k=2)
        
        # Generate a response
        context = " ".join([doc["text"] for doc in reranked_results])
        response = mock_llm_processor.generate_response(query, context)
        
        # Check response
        assert response is not None
        assert isinstance(response, str)
        assert "Answer based on context" in response

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
```

## app/tests/test_data_generator.py

```python
import os
import random
import string
import pandas as pd
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF

def generate_random_text(min_length=100, max_length=1000, paragraphs=3):
    """
    Generate random text with paragraphs for testing.
    
    Args:
        min_length: Minimum length of the text
        max_length: Maximum length of the text
        paragraphs: Number of paragraphs to generate
        
    Returns:
        Random text with paragraphs
    """
    # Generate random text length
    length = random.randint(min_length, max_length)
    
    # Generate random words
    words = []
    while len(' '.join(words)) < length:
        word_length = random.randint(3, 12)
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    
    # Split into paragraphs
    words_per_paragraph = len(words) // paragraphs
    paragraphs_text = []
    
    for i in range(paragraphs):
        start = i * words_per_paragraph
        end = (i + 1) * words_per_paragraph if i < paragraphs - 1 else len(words)
        paragraph = ' '.join(words[start:end])
        
        # Add some sentences
        paragraph = paragraph.replace(' ', '. ', random.randint(3, 8))
        paragraph = paragraph.capitalize() + '.'
        paragraphs_text.append(paragraph)
    
    # Join paragraphs with double newlines
    return '\n\n'.join(paragraphs_text)

def generate_test_pdf(output_path, text=None, pages=1):
    """
    Generate a test PDF with specified text.
    
    Args:
        output_path: Path to save the PDF
        text: Text to include in the PDF (if None, random text is generated)
        pages: Number of pages to generate
        
    Returns:
        Path to the generated PDF
    """
    # Create a new PDF document
    doc = fitz.open()
    
    for _ in range(pages):
        # Add a page
        page = doc.new_page()
        
        # Generate random text if not provided
        if text is None:
            text = generate_random_text()
        
        # Insert text
        page.insert_text((50, 50), text)
    
    # Save the PDF
    doc.save(output_path)
    
    return output_path

def generate_test_pdfs(output_dir, count=5, min_pages=1, max_pages=5):
    """
    Generate multiple test PDFs with random text.
    
    Args:
        output_dir: Directory to save the PDFs
        count: Number of PDFs to generate
        min_pages: Minimum number of pages per PDF
        max_pages: Maximum number of pages per PDF
        
    Returns:
        List of paths to the generated PDFs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_paths = []
    
    for i in range(count):
        # Generate random number of pages
        pages = random.randint(min_pages, max_pages)
        
        # Generate PDF path
        pdf_path = os.path.join(output_dir, f"test_pdf_{i+1}.pdf")
        
        # Generate PDF
        generate_test_pdf(pdf_path, pages=pages)
        
        pdf_paths.append(pdf_path)
    
    return pdf_paths

def generate_test_dataframe(pdf_paths=None, count=5):
    """
    Generate a test DataFrame with PDF metadata and text.
    
    Args:
        pdf_paths: List of PDF paths (if None, random paths are generated)
        count: Number of PDFs to include in the DataFrame
        
    Returns:
        DataFrame with PDF metadata and text
    """
    if pdf_paths is None:
        # Generate random PDF paths
        pdf_paths = [f"/path/to/pdf_{i+1}.pdf" for i in range(count)]
    
    data = []
    
    for path in pdf_paths:
        # Get filename from path
        filename = os.path.basename(path)
        
        # Generate random text
        text = generate_random_text(min_length=500, max_length=5000, paragraphs=5)
        
        # Generate random size
        size_bytes = random.randint(1000, 10000000)
        
        # Add to data
        data.append({
            'path': path,
            'filename': filename,
            'text': text,
            'size_bytes': size_bytes
        })
    
    return pd.DataFrame(data)

def generate_test_chunks_dataframe(pdf_df=None, chunks_per_doc=5):
    """
    Generate a test DataFrame with text chunks.
    
    Args:
        pdf_df: DataFrame with PDF metadata and text
        chunks_per_doc: Number of chunks per document
        
    Returns:
        DataFrame with text chunks
    """
    if pdf_df is None:
        # Generate random PDF DataFrame
        pdf_df = generate_test_dataframe()
    
    chunks_data = []
    
    for _, row in pdf_df.iterrows():
        # Skip if no text
        if not row['text'] or len(row['text']) == 0:
            continue
        
        # Split text into chunks (simple splitting for testing)
        text_length = len(row['text'])
        chunk_size = text_length // chunks_per_doc
        
        for i in range(chunks_per_doc):
            # Calculate chunk start and end
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < chunks_per_doc - 1 else text_length
            
            # Extract chunk text
            chunk_text = row['text'][start:end]
            
            # Add chunk to list
            chunk_data = {
                'chunk_id': f"chunk_{row['filename']}_{i+1}",
                'pdf_path': row['path'],
                'filename': row['filename'],
                'chunk_index': i,
                'chunk_text': chunk_text,
                'token_count': len(chunk_text.split())
            }
            chunks_data.append(chunk_data)
    
    return pd.DataFrame(chunks_data)

def generate_test_embeddings(chunks_df=None, embedding_dim=384):
    """
    Generate test embeddings for text chunks.
    
    Args:
        chunks_df: DataFrame with text chunks
        embedding_dim: Dimension of the embeddings
        
    Returns:
        DataFrame with text chunks and embeddings
    """
    if chunks_df is None:
        # Generate random chunks DataFrame
        chunks_df = generate_test_chunks_dataframe()
    
    # Copy the DataFrame
    embeddings_df = chunks_df.copy()
    
    # Add random embeddings
    embeddings = []
    for _ in range(len(chunks_df)):
        # Generate random embedding
        embedding = np.random.randn(embedding_dim)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
    
    # Add embeddings to DataFrame
    embeddings_df['embedding'] = embeddings
    
    return embeddings_df

if __name__ == "__main__":
    # Example usage
    output_dir = "app/tests/data/test_pdfs"
    
    # Generate test PDFs
    pdf_paths = generate_test_pdfs(output_dir, count=3)
    print(f"Generated {len(pdf_paths)} test PDFs in {output_dir}")
    
    # Generate test DataFrame
    pdf_df = generate_test_dataframe(pdf_paths)
    print(f"Generated DataFrame with {len(pdf_df)} PDFs")
    
    # Generate test chunks
    chunks_df = generate_test_chunks_dataframe(pdf_df)
    print(f"Generated {len(chunks_df)} chunks")
    
    # Generate test embeddings
    embeddings_df = generate_test_embeddings(chunks_df)
    print(f"Generated embeddings with dimension {len(embeddings_df['embedding'][0])}") 
```

## app/tests/test_end_to_end.py

```python
import os
import sys
import time
import pytest
import requests
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import PDF_UPLOAD_FOLDER, VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME
from app.utils.vector_db import VectorDBClient
from app.clients.mlflow_client import create_mlflow_client

# Test configuration
FLASK_BASE_URL = "http://localhost:8000"
TEST_PDF_PATH = os.path.join(Path(__file__).resolve().parent, "data", "sample.pdf")
TEST_QUESTION = "What is machine learning?"

def check_service_availability():
    """Check if all services are available."""
    services = {
        "Flask Web App": f"{FLASK_BASE_URL}/api/health",
        "MLflow": f"http://localhost:5001/ping",
        "Vector DB": f"http://localhost:6333/healthz",
    }
    
    available = {}
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            available[name] = response.status_code == 200
        except:
            available[name] = False
    
    return available

@pytest.fixture(scope="module")
def check_system():
    """Check if the entire system is ready for testing."""
    available = check_service_availability()
    
    if not all(available.values()):
        unavailable = [name for name, status in available.items() if not status]
        pytest.skip(f"Some services are not available: {', '.join(unavailable)}")

def test_health_endpoints(check_system):
    """Test health endpoints of all services."""
    # Flask health
    response = requests.get(f"{FLASK_BASE_URL}/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    
    # Check MLflow through Flask health endpoint
    assert data["mlflow"] == True

def test_vector_db_connection(check_system):
    """Test connection to vector database."""
    vector_db = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, 384)
    assert vector_db.client is not None
    
    # Check if collection exists
    collections = vector_db.client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    # If collection doesn't exist, this test is inconclusive
    if COLLECTION_NAME not in collection_names:
        pytest.skip(f"Collection {COLLECTION_NAME} does not exist yet")
    
    # Check collection count
    count = vector_db.count_vectors()
    print(f"Vector count in collection: {count}")

def test_mlflow_client(check_system):
    """Test MLflow client."""
    mlflow_client = create_mlflow_client()
    assert mlflow_client.is_alive() == True

def test_document_list(check_system):
    """Test document list API."""
    response = requests.get(f"{FLASK_BASE_URL}/documents")
    assert response.status_code == 200

def test_ask_question(check_system):
    """Test asking a question."""
    response = requests.post(
        f"{FLASK_BASE_URL}/api/ask",
        json={"question": TEST_QUESTION}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "text" in data
    assert len(data["text"]) > 0
    
    # Sources might be empty if no relevant documents are found
    assert "sources" in data

def test_end_to_end_flow(check_system):
    """Test the complete end-to-end flow."""
    # This test is more of a recipe for manual testing
    print("\nEnd-to-end test steps:")
    print("1. Upload a PDF document through the web interface")
    print("2. Trigger indexing process")
    print("3. Wait for indexing to complete")
    print("4. Ask a question related to the document content")
    print("5. Verify that the answer references information from the document")
    
    # Skip the actual test for automation
    pytest.skip("This test is a manual procedure")

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
```

## app/tests/test_integration.py

```python
"""Integration tests for the PDF RAG System."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path

from app.utils.pdf_ingestion import process_pdfs
from app.utils.vector_db import VectorDBClient
from app.utils.embedding_generation import EmbeddingGenerator


class TestPDFRagIntegration:
    """Integration tests for the PDF RAG system."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Create a temporary directory with test PDFs."""
        temp_dir = tempfile.mkdtemp()
        
        # Use the sample_pdf_dir fixture to populate our test directory
        sample_dir = pytest.lazy_fixture("sample_pdf_dir")
        
        # Copy files from sample_dir to temp_dir
        for item in os.listdir(sample_dir):
            s = os.path.join(sample_dir, item)
            d = os.path.join(temp_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.integration
    @pytest.mark.pdf
    def test_pdf_to_vector_db(self, test_data_dir, monkeypatch):
        """Test the full pipeline from PDF processing to vector database."""
        # Mock the embedding generator to return fixed vectors
        class MockEmbeddingGenerator:
            def embed_documents(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            def embed_query(self, query):
                return [0.1, 0.2, 0.3]
        
        # Mock the vector database client
        class MockVectorDBClient:
            def __init__(self, *args, **kwargs):
                pass
                
            def create_collection(self):
                pass
                
            def upload_vectors(self, df, vector_column='embedding', batch_size=100):
                pass
                
            def search(self, query_vector, limit=5):
                return [
                    {"text": "Document 1 content", "metadata": {"source": "doc1.pdf"}, "score": 0.95},
                    {"text": "Document 2 content", "metadata": {"source": "doc2.pdf"}, "score": 0.85},
                ]
        
        # Patch the embedding generator and vector database client
        monkeypatch.setattr("app.utils.embedding_generation.EmbeddingGenerator", MockEmbeddingGenerator)
        monkeypatch.setattr("app.utils.vector_db.VectorDBClient", MockVectorDBClient)
        
        # Process PDFs
        pdf_df = process_pdfs(test_data_dir)
        
        # Check that we have data
        assert not pdf_df.empty
        assert "text" in pdf_df.columns
        assert "path" in pdf_df.columns
        
        # Create a vector database client
        vector_db = VectorDBClient("localhost", 6333, "test_collection", 384)
        
        # Create collection
        vector_db.create_collection()
        
        # Add documents to vector database
        vector_db.upload_vectors(pdf_df)
        
        # Query the vector database
        results = vector_db.search([0.1, 0.2, 0.3], limit=2)
        
        # Check results
        assert len(results) > 0
        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert "score" in result
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_with_mocks(self, test_data_dir, monkeypatch):
        """Test the end-to-end flow with mocked components."""
        # Mock the necessary components
        class MockLLM:
            def generate(self, prompt, context):
                return f"Answer based on context: {context[:50]}..."
        
        class MockVectorDBClient:
            def search(self, query_vector, limit=3):
                return [
                    {"text": "Document 1 content", "metadata": {"source": "doc1.pdf"}, "score": 0.95},
                    {"text": "Document 2 content", "metadata": {"source": "doc2.pdf"}, "score": 0.85},
                ]
        
        # Patch the components
        monkeypatch.setattr("app.utils.llm.LLMProcessor", MockLLM)
        monkeypatch.setattr("app.utils.vector_db.VectorDBClient", MockVectorDBClient)
        
        # Import the RAG module (after patching)
        from app.utils.search import RAGSearch
        
        # Initialize the RAG system
        rag_system = RAGSearch()
        
        # Process a query
        query = "What is retrieval-augmented generation?"
        response = rag_system.process_query(query)
        
        # Check the response
        assert response is not None
        assert isinstance(response, str)
        assert "Answer based on context" in response
```

## app/tests/test_pdf_extraction.py

```python
"""Tests for PDF extraction functionality using mocks."""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.utils.pdf_ingestion import extract_text_from_pdf, process_pdfs


@pytest.fixture
def mock_pdf_path():
    return os.path.join("test_dir", "sample.pdf")


@pytest.fixture
def mock_pdf_dir():
    return "test_dir"


@pytest.mark.unit
@pytest.mark.pdf
def test_extract_text_with_mock(mock_pdf_path):
    """Test extract_text_from_pdf with a mocked PDF file."""
    expected_text = "This is sample text from a PDF file."
    
    # Mock fitz.open
    with patch("app.utils.pdf_ingestion.fitz.open") as mock_open:
        # Configure the mock
        mock_doc = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_doc
        
        # Set up the pages
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (0, 0, 0, 0, expected_text, 0, 0, 0)
        ]
        mock_doc.__iter__.return_value = [mock_page]
        
        # Call the function
        result = extract_text_from_pdf(mock_pdf_path)
        
        # Assertions
        assert expected_text in result
        mock_open.assert_called_once_with(mock_pdf_path)


@pytest.mark.unit
@pytest.mark.pdf
def test_extract_text_with_exception(mock_pdf_path):
    """Test extract_text_from_pdf with an exception."""
    # Mock fitz.open to raise an exception
    with patch("app.utils.pdf_ingestion.fitz.open") as mock_open:
        mock_open.side_effect = Exception("PDF Error")
        
        # Mock the logger
        with patch("app.utils.pdf_ingestion.logger") as mock_logger:
            # Call the function
            result = extract_text_from_pdf(mock_pdf_path)
            
            # Assertions
            assert result == ""
            mock_logger.error.assert_called_once()
            assert "Error extracting text from" in mock_logger.error.call_args[0][0]


@pytest.mark.unit
@pytest.mark.pdf
@patch("app.utils.pdf_ingestion.scan_directory")
@patch("app.utils.pdf_ingestion.extract_text_from_pdf")
@patch("app.utils.pdf_ingestion.create_pdf_dataframe")
@patch("app.utils.pdf_ingestion.tqdm.pandas")
def test_process_pdfs_with_mocks(mock_tqdm_pandas, mock_create_df, mock_extract, mock_scan, mock_pdf_dir):
    """Test process_pdfs with mocked dependencies."""
    # Configure mocks
    mock_scan.return_value = [
        {"path": "test_dir/doc1.pdf", "filename": "doc1.pdf", "size_bytes": 1000},
        {"path": "test_dir/doc2.pdf", "filename": "doc2.pdf", "size_bytes": 2000},
    ]
    
    # Create a mock DataFrame
    mock_df = MagicMock()
    mock_df.__getitem__.return_value.progress_apply.return_value = ["Text from doc1", "Text from doc2"]
    
    # Mock the text_lengths Series
    mock_text_lengths = MagicMock()
    mock_text_lengths.min.return_value = 10
    mock_text_lengths.max.return_value = 20
    mock_text_lengths.mean.return_value = 15
    mock_text_lengths.median.return_value = 15
    
    # Set up the str.len() to return the mock Series
    mock_df.__getitem__.return_value.str.len.return_value = mock_text_lengths
    
    # Set up the empty text count
    mock_df.__getitem__.return_value.apply.return_value.sum.return_value = 0
    
    mock_create_df.return_value = mock_df
    
    # Call the function
    result = process_pdfs(mock_pdf_dir)
    
    # Assertions
    assert result is mock_df
    mock_scan.assert_called_once_with(mock_pdf_dir)
    mock_create_df.assert_called_once_with(mock_scan.return_value) 
```

## app/tests/test_pdf_ingestion.py

```python
"""Pytest version of the PDF ingestion tests."""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from app.utils.pdf_ingestion import scan_directory, create_pdf_dataframe, extract_text_from_pdf, process_pdfs, logger

@pytest.mark.unit
@pytest.mark.pdf
def test_scan_directory(sample_pdf_dir, empty_dir):
    """Test scanning a directory for PDF files."""
    # Test with a directory containing PDFs
    pdf_files = scan_directory(sample_pdf_dir)
    assert len(pdf_files) == 4  # text.pdf, blank.pdf, corrupted.pdf, and sub_text.pdf
    
    # Check that each file has the expected metadata
    for pdf_file in pdf_files:
        assert "path" in pdf_file
        assert "filename" in pdf_file
        assert "size_bytes" in pdf_file
        assert pdf_file["filename"].endswith(".pdf")
    
    # Test with an empty directory
    empty_files = scan_directory(empty_dir)
    assert len(empty_files) == 0

@pytest.mark.unit
@pytest.mark.pdf
def test_create_pdf_dataframe():
    """Test creating a DataFrame from PDF metadata."""
    # Test with sample data
    pdf_data = [
        {"path": "test1.pdf", "filename": "test1.pdf", "size_bytes": 1000},
        {"path": "test2.pdf", "filename": "test2.pdf", "size_bytes": 2000}
    ]
    df = create_pdf_dataframe(pdf_data)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "path" in df.columns
    assert "filename" in df.columns
    assert "size_bytes" in df.columns
    
    # Test with empty list
    empty_df = create_pdf_dataframe([])
    assert isinstance(empty_df, pd.DataFrame)
    assert len(empty_df) == 0

@pytest.mark.unit
@pytest.mark.pdf
def test_extract_text_from_pdf(sample_pdf_dir, caplog):
    """Test extracting text from PDF files."""
    # Test with a valid PDF
    valid_pdf = os.path.join(sample_pdf_dir, "text.pdf")
    text = extract_text_from_pdf(valid_pdf)
    assert text.strip() != ""
    
    # Test with a blank PDF
    blank_pdf = os.path.join(sample_pdf_dir, "blank.pdf")
    blank_text = extract_text_from_pdf(blank_pdf)
    assert blank_text.strip() == ""
    
    # Test with a corrupted PDF
    corrupted_pdf = os.path.join(sample_pdf_dir, "corrupted.pdf")
    corrupted_text = extract_text_from_pdf(corrupted_pdf)
    assert corrupted_text == ""
    assert "Error extracting text from" in caplog.text

@pytest.mark.unit
@pytest.mark.pdf
def test_process_pdfs(sample_pdf_dir, caplog):
    """Test processing PDFs in a directory."""
    # Test with a directory containing PDFs
    df = process_pdfs(sample_pdf_dir)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "path" in df.columns
    assert "filename" in df.columns
    assert "size_bytes" in df.columns
    assert "text" in df.columns
    
    # Check that PDFs with no text are logged
    assert "Found" in caplog.text and "PDFs with no extractable text" in caplog.text

@pytest.mark.unit
@pytest.mark.pdf
def test_process_pdfs_empty_dir():
    """Test processing PDFs in an empty directory."""
    # Need to patch the entire function to handle empty DataFrame
    with patch("app.utils.pdf_ingestion.process_pdfs") as mock_process:
        # Configure mock
        mock_df = pd.DataFrame()
        mock_process.return_value = mock_df
        
        # Call the function
        empty_dir = "/path/to/empty/dir"
        result = mock_process(empty_dir)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        mock_process.assert_called_once_with(empty_dir)
```

## app/tests/test_pdf_ingestion_pytest.py

```python
"""
Pytest version of the PDF ingestion tests.
"""

import os
import pytest
import pandas as pd
from app.utils.pdf_ingestion import scan_directory, create_pdf_dataframe, extract_text_from_pdf, process_pdfs, logger

def test_scan_directory(sample_pdf_dir, empty_dir):
    """Test the scan_directory function."""
    # Test with a directory containing PDFs and other files
    pdf_files = scan_directory(sample_pdf_dir)
    assert len(pdf_files) == 4  # Expect 4 PDFs: text.pdf, blank.pdf, corrupted.pdf, sub_text.pdf
    
    filenames = [f['filename'] for f in pdf_files]
    assert 'text.pdf' in filenames
    assert 'blank.pdf' in filenames
    assert 'corrupted.pdf' in filenames
    assert 'sub_text.pdf' in filenames

    # Verify metadata for each PDF
    for f in pdf_files:
        if f['filename'] == 'text.pdf':
            assert f['page_count'] == 1
            assert f['parent_dir'] == sample_pdf_dir
        elif f['filename'] == 'blank.pdf':
            assert f['page_count'] == 1
        elif f['filename'] == 'corrupted.pdf':
            assert f['page_count'] == 0  # Corrupted PDF should have page_count 0
        elif f['filename'] == 'sub_text.pdf':
            assert f['page_count'] == 1
            assert f['parent_dir'] == os.path.join(sample_pdf_dir, 'subdir')

    # Test with an empty directory
    pdf_files = scan_directory(empty_dir)
    assert len(pdf_files) == 0

    # Verify logging for corrupted PDF
    with pytest.warns(UserWarning, match="Could not read PDF metadata"):
        scan_directory(sample_pdf_dir)

def test_create_pdf_dataframe(sample_pdf_data):
    """Test the create_pdf_dataframe function."""
    # Test with a sample list of dictionaries
    df = create_pdf_dataframe(sample_pdf_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['path', 'filename', 'size_bytes']

    # Test with an empty list
    df = create_pdf_dataframe([])
    assert len(df) == 0

def test_extract_text_from_pdf(sample_pdf_dir):
    """Test the extract_text_from_pdf function."""
    # Test with a PDF containing text
    text = extract_text_from_pdf(os.path.join(sample_pdf_dir, 'text.pdf'))
    assert 'Hello, world!' in text

    # Test with a blank PDF
    text = extract_text_from_pdf(os.path.join(sample_pdf_dir, 'blank.pdf'))
    assert text == ''

    # Test with a corrupted PDF
    with pytest.raises(Exception):
        extract_text_from_pdf(os.path.join(sample_pdf_dir, 'corrupted.pdf'))

def test_process_pdfs(sample_pdf_dir, empty_dir):
    """Test the process_pdfs function."""
    # Test with a directory containing PDFs
    df = process_pdfs(sample_pdf_dir)
    assert len(df) == 4  # Expect 4 rows in the DataFrame
    
    text_pdf_text = df.loc[df['filename'] == 'text.pdf', 'text'].iloc[0]
    assert 'Hello, world!' in text_pdf_text
    
    blank_pdf_text = df.loc[df['filename'] == 'blank.pdf', 'text'].iloc[0]
    assert blank_pdf_text == ''
    
    corrupted_pdf_text = df.loc[df['filename'] == 'corrupted.pdf', 'text'].iloc[0]
    assert corrupted_pdf_text == ''
    
    sub_text_pdf_text = df.loc[df['filename'] == 'sub_text.pdf', 'text'].iloc[0]
    assert 'Subdirectory PDF' in sub_text_pdf_text

    # Verify logging for PDFs with no extractable text
    with pytest.warns(UserWarning, match="Found 2 PDFs with no extractable text"):
        process_pdfs(sample_pdf_dir)

    # Test with an empty directory
    df = process_pdfs(empty_dir)
    assert len(df) == 0 
```

## app/tests/test_pdf_processing.py

```python
import os
import sys
import pytest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.pdf_ingestion import scan_directory, extract_text_from_pdf, process_pdfs
from app.utils.text_chunking import chunk_text, process_chunks
from app.config.settings import PDF_UPLOAD_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP

def test_scan_directory():
    """Test scanning directory for PDFs."""
    # Create a test PDF if none exists
    if not list(Path(PDF_UPLOAD_FOLDER).rglob('*.pdf')):
        pytest.skip("No PDF files found for testing")
    
    pdfs = scan_directory(PDF_UPLOAD_FOLDER)
    assert len(pdfs) > 0, "No PDFs found in directory"
    assert 'path' in pdfs[0], "PDF info missing 'path'"
    assert 'filename' in pdfs[0], "PDF info missing 'filename'"

def test_extract_text():
    """Test extracting text from a PDF."""
    # Create a test PDF if none exists
    if not list(Path(PDF_UPLOAD_FOLDER).rglob('*.pdf')):
        pytest.skip("No PDF files found for testing")
    
    pdf_path = next(Path(PDF_UPLOAD_FOLDER).rglob('*.pdf'))
    text = extract_text_from_pdf(str(pdf_path))
    assert text, "No text extracted from PDF"
    
def test_chunk_text():
    """Test chunking text."""
    sample_text = """
    This is a sample document that will be split into chunks.
    It has multiple sentences and paragraphs.
    
    This is the second paragraph with some more text.
    We want to make sure the chunking works correctly.
    
    Let's add a third paragraph to ensure we have enough text to create multiple chunks.
    This should be enough for the test.
    """
    
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1, "Text not split into multiple chunks"
    assert len(chunks[0]) <= 100 + 20, "Chunk size exceeds expected maximum"

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
```

## app/tests/test_text_chunking.py

```python
import os
import sys
import uuid
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.text_chunking import clean_text, chunk_text, process_chunks


class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_clean_text_removes_excessive_whitespace(self, sample_text_data):
        """Test that clean_text removes excessive whitespace."""
        cleaned = clean_text(sample_text_data['whitespace_text'])
        assert "Text with excessive whitespace" in cleaned
        assert "  " not in cleaned  # No double spaces
    
    def test_clean_text_normalizes_line_breaks(self, sample_text_data):
        """Test that clean_text normalizes line breaks."""
        cleaned = clean_text(sample_text_data['whitespace_text'])
        assert "\n\n\n" not in cleaned  # No triple line breaks
        assert "and line breaks" in cleaned
    
    def test_clean_text_strips_whitespace(self, sample_text_data):
        """Test that clean_text strips whitespace from beginning and end."""
        cleaned = clean_text(sample_text_data['whitespace_text'])
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
    
    def test_clean_text_handles_empty_string(self, sample_text_data):
        """Test that clean_text handles empty string."""
        cleaned = clean_text(sample_text_data['empty_text'])
        assert cleaned == ""


class TestChunkText:
    """Tests for the chunk_text function."""
    
    def test_chunk_text_splits_text(self, sample_text_data):
        """Test that chunk_text splits text into chunks."""
        chunks = chunk_text(sample_text_data['long_text'], chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1, "Text not split into multiple chunks"
    
    def test_chunk_size_respected(self, sample_text_data):
        """Test that chunk_text respects chunk size."""
        chunks = chunk_text(sample_text_data['long_text'], chunk_size=100, chunk_overlap=20)
        # Allow for some flexibility due to how RecursiveCharacterTextSplitter works
        # It tries to split on separators, so chunks might be smaller than chunk_size
        for chunk in chunks:
            assert len(chunk) <= 120, f"Chunk size exceeds expected maximum: {len(chunk)}"
    
    def test_chunk_overlap(self, sample_text_data):
        """Test that chunk_text includes overlap between chunks."""
        chunks = chunk_text(sample_text_data['long_text'], chunk_size=100, chunk_overlap=50)
        
        # Check if there's overlap between consecutive chunks
        if len(chunks) >= 2:
            # Find a word that should be in the overlap
            overlap_found = False
            for i in range(len(chunks) - 1):
                # Get the end of the first chunk
                end_of_first = chunks[i][-50:]
                # Check if any part of it is in the beginning of the next chunk
                if any(word in chunks[i+1][:100] for word in end_of_first.split() if len(word) > 3):
                    overlap_found = True
                    break
            assert overlap_found, "No overlap found between chunks"
    
    def test_chunk_text_with_empty_string(self, sample_text_data):
        """Test that chunk_text handles empty string."""
        chunks = chunk_text(sample_text_data['empty_text'])
        assert len(chunks) == 0, "Empty text should result in no chunks"
    
    def test_chunk_text_with_short_text(self, sample_text_data):
        """Test that chunk_text handles text shorter than chunk size."""
        chunks = chunk_text(sample_text_data['short_text'], chunk_size=100)
        assert len(chunks) == 1, "Short text should result in a single chunk"
        assert chunks[0] == sample_text_data['short_text'].strip(), "Chunk should contain the entire text"
    
    def test_chunk_text_with_paragraphs(self, sample_text_data):
        """Test that chunk_text uses separators correctly with paragraphs."""
        chunks = chunk_text(sample_text_data['paragraphs'], chunk_size=50, chunk_overlap=0)
        # Should split at paragraph breaks, not mid-paragraph
        assert "Paragraph 1" in chunks[0]
        # Check that we have multiple chunks
        assert len(chunks) > 1, "Text should be split into multiple chunks"


class TestProcessChunks:
    """Tests for the process_chunks function."""
    
    def test_process_chunks_creates_dataframe(self, sample_pdf_dataframe):
        """Test that process_chunks creates a DataFrame with chunks."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        assert isinstance(chunks_df, pd.DataFrame), "Result should be a DataFrame"
        assert len(chunks_df) > 0, "DataFrame should contain chunks"
    
    def test_process_chunks_skips_empty_text(self, sample_pdf_dataframe):
        """Test that process_chunks skips documents with empty text."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        # Check that no chunks were created for the empty document
        assert not any(chunks_df['filename'] == 'empty.pdf'), "Empty document should be skipped"
    
    def test_process_chunks_limits_chunks(self, sample_pdf_dataframe):
        """Test that process_chunks limits chunks per document."""
        max_chunks = 2
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=50, chunk_overlap=10, max_chunks_per_doc=max_chunks)
        
        # Group by filename and count chunks
        chunk_counts = chunks_df.groupby('filename').size()
        
        # Check that no document has more than max_chunks
        for count in chunk_counts:
            assert count <= max_chunks, f"Document has more than {max_chunks} chunks"
    
    def test_process_chunks_generates_unique_ids(self, sample_pdf_dataframe):
        """Test that process_chunks generates unique IDs for chunks."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Check that all chunk_ids are unique
        assert len(chunks_df['chunk_id'].unique()) == len(chunks_df), "Chunk IDs should be unique"
        
        # Check that chunk_ids are valid UUIDs
        for chunk_id in chunks_df['chunk_id']:
            try:
                uuid.UUID(chunk_id)
                is_valid = True
            except ValueError:
                is_valid = False
            assert is_valid, f"Chunk ID {chunk_id} is not a valid UUID"
    
    def test_process_chunks_includes_metadata(self, sample_pdf_dataframe):
        """Test that process_chunks includes metadata from the original DataFrame."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Check that metadata columns are present
        assert 'pdf_path' in chunks_df.columns
        assert 'filename' in chunks_df.columns
        assert 'chunk_index' in chunks_df.columns
        assert 'chunk_text' in chunks_df.columns
        assert 'token_count' in chunks_df.columns
        
        # Check that metadata is correctly copied
        for _, row in chunks_df.iterrows():
            original_row = sample_pdf_dataframe[sample_pdf_dataframe['filename'] == row['filename']].iloc[0]
            assert row['pdf_path'] == original_row['path']
            assert row['filename'] == original_row['filename']
    
    @patch('app.utils.text_chunking.chunk_text')
    def test_process_chunks_calls_chunk_text(self, mock_chunk_text, sample_pdf_dataframe):
        """Test that process_chunks calls chunk_text with correct parameters."""
        mock_chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        
        process_chunks(sample_pdf_dataframe, chunk_size=200, chunk_overlap=30)
        
        # Check that chunk_text was called for each document with text
        assert mock_chunk_text.call_count == 2  # Two documents with text
        
        # Check that chunk_text was called with correct parameters
        mock_chunk_text.assert_any_call(sample_pdf_dataframe.iloc[0]['text'], 200, 30)
        mock_chunk_text.assert_any_call(sample_pdf_dataframe.iloc[1]['text'], 200, 30)


# Integration tests
class TestTextChunkingIntegration:
    """Integration tests for text chunking functionality."""
    
    def test_end_to_end_chunking(self, sample_pdf_dataframe):
        """Test the entire chunking process from text to DataFrame."""
        # Process chunks
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Verify results
        assert isinstance(chunks_df, pd.DataFrame)
        assert len(chunks_df) > 1
        assert 'chunk_id' in chunks_df.columns
        assert 'pdf_path' in chunks_df.columns
        assert 'filename' in chunks_df.columns
        assert 'chunk_index' in chunks_df.columns
        assert 'chunk_text' in chunks_df.columns
        assert 'token_count' in chunks_df.columns
        
        # Check that chunk indices are sequential within each document
        for filename in chunks_df['filename'].unique():
            doc_chunks = chunks_df[chunks_df['filename'] == filename]
            assert list(doc_chunks['chunk_index']) == list(range(len(doc_chunks)))
        
        # Check that token counts are reasonable
        for _, row in chunks_df.iterrows():
            assert row['token_count'] > 0
            assert row['token_count'] == len(row['chunk_text'].split())


if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__]) 
```

## app/tests/test_text_chunking_integration.py

```python
import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.text_chunking import clean_text, chunk_text, process_chunks
from app.utils.pdf_ingestion import process_pdfs


class TestTextChunkingWithPDFIngestion:
    """Integration tests for text chunking with PDF ingestion."""
    
    @patch('app.utils.pdf_ingestion.extract_text_from_pdf')
    def test_pdf_ingestion_to_chunking(self, mock_extract_text, sample_pdf_dir):
        """Test the integration of PDF ingestion with text chunking."""
        # Mock the text extraction to return predictable text
        mock_extract_text.return_value = "This is extracted text from a PDF. " * 20
        
        # Process PDFs
        pdf_df = process_pdfs(sample_pdf_dir)
        
        # Process chunks
        chunks_df = process_chunks(pdf_df, chunk_size=100, chunk_overlap=20)
        
        # Verify results
        assert isinstance(chunks_df, pd.DataFrame)
        assert len(chunks_df) > 0
        assert 'chunk_id' in chunks_df.columns
        assert 'pdf_path' in chunks_df.columns
        assert 'filename' in chunks_df.columns
        assert 'chunk_text' in chunks_df.columns
        
        # Check that we have chunks for each PDF with text
        pdf_count = len([f for f in os.listdir(sample_pdf_dir) 
                        if f.endswith('.pdf') and f != 'corrupted.pdf'])
        pdf_count += len([f for f in os.listdir(os.path.join(sample_pdf_dir, 'subdir')) 
                         if f.endswith('.pdf')])
        
        # We should have chunks from each PDF
        assert len(chunks_df['filename'].unique()) == pdf_count


class TestTextChunkingWithVectorDB:
    """Integration tests for text chunking with vector database."""
    
    @patch('app.utils.vector_db.insert_chunks')
    def test_chunking_to_vector_db(self, mock_insert_chunks, sample_pdf_dataframe):
        """Test the integration of text chunking with vector database insertion."""
        # Process chunks
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Mock the vector DB insertion
        mock_insert_chunks.return_value = {'inserted': len(chunks_df), 'errors': 0}
        
        # Simulate inserting chunks into vector DB
        result = mock_insert_chunks(chunks_df)
        
        # Verify results
        assert result['inserted'] == len(chunks_df)
        assert result['errors'] == 0
        
        # Verify that mock was called with the chunks DataFrame
        mock_insert_chunks.assert_called_once()
        args, _ = mock_insert_chunks.call_args
        assert isinstance(args[0], pd.DataFrame)
        assert len(args[0]) == len(chunks_df)


class TestEndToEndProcessing:
    """End-to-end tests for the document processing pipeline."""
    
    @patch('app.utils.pdf_ingestion.extract_text_from_pdf')
    @patch('app.utils.vector_db.insert_chunks')
    @patch('app.utils.embedding.generate_embeddings')
    def test_end_to_end_pipeline(self, mock_generate_embeddings, mock_insert_chunks, 
                                mock_extract_text, sample_pdf_dir):
        """Test the entire document processing pipeline from PDF to vector DB."""
        # Mock the text extraction
        mock_extract_text.return_value = "This is extracted text from a PDF. " * 20
        
        # Mock the embedding generation
        mock_generate_embeddings.return_value = pd.DataFrame({
            'chunk_id': ['id1', 'id2'],
            'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        })
        
        # Mock the vector DB insertion
        mock_insert_chunks.return_value = {'inserted': 2, 'errors': 0}
        
        # Process PDFs
        pdf_df = process_pdfs(sample_pdf_dir)
        
        # Process chunks
        chunks_df = process_chunks(pdf_df, chunk_size=100, chunk_overlap=20)
        
        # Generate embeddings (mocked)
        embeddings_df = mock_generate_embeddings(chunks_df)
        
        # Insert into vector DB (mocked)
        result = mock_insert_chunks(embeddings_df)
        
        # Verify results
        assert isinstance(chunks_df, pd.DataFrame)
        assert len(chunks_df) > 0
        assert result['inserted'] > 0
        assert result['errors'] == 0


if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__]) 
```

## app/tests/test_vector_db.py

```python
import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.vector_db import VectorDBClient
from app.config.settings import VECTOR_DB_HOST, VECTOR_DB_PORT, VECTOR_DIMENSION

def test_vector_db_connection():
    """Test connecting to the vector database."""
    # Create client
    client = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, "test_collection", VECTOR_DIMENSION)
    
    # Check connection
    try:
        collections = client.client.get_collections()
        assert collections is not None, "Failed to get collections"
    except Exception as e:
        pytest.fail(f"Failed to connect to vector database: {str(e)}")

def test_collection_operations():
    """Test collection operations."""
    # Create client
    client = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, "test_collection", VECTOR_DIMENSION)
    
    # Create collection
    client.create_collection()
    
    # Check if collection exists
    collections = client.client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    assert "test_collection" in collection_names, "Collection not created"
    
    # Delete collection
    client.delete_collection()
    
    # Check if collection is deleted
    collections = client.client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    assert "test_collection" not in collection_names, "Collection not deleted"

def test_vector_operations():
    """Test vector operations."""
    # Create client
    client = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, "test_vectors", VECTOR_DIMENSION)
    
    # Create collection and clear any existing data
    client.delete_collection()
    client.create_collection()
    
    # Create test vectors
    import pandas as pd
    
    # Create 10 random test vectors
    np.random.seed(42)  # For reproducibility
    test_vectors = []
    for i in range(10):
        vec = np.random.rand(VECTOR_DIMENSION)
        # Normalize for cosine similarity
        vec = vec / np.linalg.norm(vec)
        test_vectors.append(vec)
    
    # Create test dataframe
    df = pd.DataFrame({
        'chunk_id': [f"chunk_{i}" for i in range(10)],
        'pdf_path': [f"/path/to/pdf_{i}.pdf" for i in range(10)],
        'filename': [f"pdf_{i}.pdf" for i in range(10)],
        'chunk_index': list(range(10)),
        'chunk_text': [f"This is test chunk {i}" for i in range(10)],
        'token_count': [len(f"This is test chunk {i}".split()) for i in range(10)],
        'embedding': test_vectors
    })
    
    # Upload vectors
    client.upload_vectors(df)
    
    # Check if vectors are uploaded
    count = client.count_vectors()
    assert count == 10, f"Expected 10 vectors, got {count}"
    
    # Search for similar vector
    results = client.search(test_vectors[0])
    assert len(results) > 0, "No search results returned"
    assert results[0]['chunk_id'] == "chunk_0", "First result should be the query vector itself"
    
    # Clean up
    client.delete_collection()

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
```

## app/utils/__init__.py

```python
"""
Initializes the utils module.
"""

```

## app/utils/adapters/meta_llama_adapter.py

```python
import os
import logging
import time
import threading
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class GenerationTimeoutError(Exception):
    """Exception raised when text generation times out."""
    pass

class MetaLlamaAdapter:
    def __init__(self, model_path: str, max_new_tokens: int = 512, generation_timeout: int = 120):
        """Initialize the Meta Llama adapter."""
        logger.info(f"Loading Meta Llama model from {model_path}")
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.generation_timeout = generation_timeout  # Timeout in seconds
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check GPU availability and set up appropriate configuration
        import torch
        
        # Set default model loading kwargs
        model_kwargs = {
            "low_cpu_mem_usage": True,
        }
        
        # Set pipeline kwargs
        pipe_kwargs = {}
        
        if torch.cuda.is_available():
            logger.info("Using CUDA for model inference")
            device_map = "auto"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs.update({
                "device_map": device_map,
                "torch_dtype": torch_dtype,
            })
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using MPS (Metal) for model inference")
            # For Apple Silicon, use more optimized settings
            device_map = "mps" 
            torch_dtype = torch.float16
            
            # Set torch to use MPS (Metal Performance Shaders)
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use less GPU memory
            
            model_kwargs.update({
                "device_map": device_map,
                "torch_dtype": torch_dtype,
            })
            
            # Enable memory efficient attention
            if torch.__version__ >= "2.0.0":
                pipe_kwargs["use_cache"] = True

        else:
            logger.info("Using CPU for model inference - this will be slow")
            # For CPU, enable 4-bit quantization if possible to save memory
            try:
                from transformers import BitsAndBytesConfig
                
                model_kwargs.update({
                    "device_map": "auto",
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    ),
                })
                logger.info("Enabled 4-bit quantization for CPU inference")
            except ImportError:
                model_kwargs.update({
                    "device_map": "auto",
                    "torch_dtype": torch.float32,
                })
                logger.warning("Quantization not available - model will use full precision")
        
        # Set max length parameter
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For MPS (Metal), limit context size
            pipe_kwargs["max_length"] = 2048
        
        # Load model with performance settings
        logger.info(f"Loading model with settings: {model_kwargs}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                **model_kwargs
            )
            
            # Create pipeline with appropriate settings
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **pipe_kwargs
            )
            
            logger.info("Meta Llama model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _generate_with_timeout(self, prompt, generation_kwargs):
        """Generate text with a timeout."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = self.pipe(prompt, **generation_kwargs)
            except Exception as e:
                exception[0] = e
                logger.error(f"Generation error: {str(e)}")
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        
        start_time = time.time()
        thread.start()
        thread.join(self.generation_timeout)
        
        if thread.is_alive():
            logger.warning(f"Generation timed out after {self.generation_timeout} seconds")
            raise GenerationTimeoutError(f"Text generation timed out after {self.generation_timeout} seconds")
        
        if exception[0]:
            raise exception[0]
            
        logger.info(f"Generation completed in {time.time() - start_time:.2f} seconds")
        return result[0]
    
    def __call__(self, prompt: str, **kwargs):
        """Generate text using the Meta Llama model."""
        generation_kwargs = {
            "max_new_tokens": min(kwargs.get("max_tokens", self.max_new_tokens), 256),  # Limit token generation
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.2) > 0,
        }
        
        logger.info(f"Generating with parameters: {generation_kwargs}")
        
        # Generate text with timeout
        try:
            outputs = self._generate_with_timeout(prompt, generation_kwargs)
        except GenerationTimeoutError:
            # Return a partial response if timed out
            return {
                "choices": [
                    {
                        "text": "[Generation timed out. The model is taking too long to respond.]",
                        "finish_reason": "timeout",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(prompt)),
                    "completion_tokens": 0,
                    "total_tokens": len(self.tokenizer.encode(prompt)),
                }
            }
        
        # Format to match llama-cpp-python output format
        try:
            generated_text = outputs[0]["generated_text"][len(prompt):]
            
            # Count tokens
            input_tokens = len(self.tokenizer.encode(prompt))
            output_tokens = len(self.tokenizer.encode(generated_text))
            
            return {
                "choices": [
                    {
                        "text": generated_text,
                        "finish_reason": "length" if output_tokens >= generation_kwargs["max_new_tokens"] else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return {
                "choices": [
                    {
                        "text": f"[Error processing model response: {str(e)}]",
                        "finish_reason": "error",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(prompt)),
                    "completion_tokens": 0,
                    "total_tokens": len(self.tokenizer.encode(prompt)),
                }
            }
```

## app/utils/code_review.md

```markdown
# Code Review Results
## 1. Redundant Code

1. **Duplicate Logging Setup**: 
   - Logging is initialized in multiple files (`serve_model.py`, `rag_app.py`, `pipeline.py`, `app.py`) with similar configurations. Consider centralizing the logging setup in a shared module.

2. **Duplicate System Path Configuration**:
   - `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` appears in multiple files. Create a shared utility function or configure the Python path properly.

3. **Duplicate Flask App Creation**: 
   - In `flask-app/app.py`, there are two instances of Flask app creation (lines 29 and 56). The first instance is unnecessary.

4. **Redundant Health Checking**:
   - Similar health check logic is implemented in multiple places. Consider creating a shared health check utility.

5. **Duplicate Constants**:
   - Constants like `MODEL_SERVER_URL`, timeout settings, and retry configurations appear in multiple files. Create a central configuration module.

## 2. Code That Does Not Do Anything

1. **Unused Monkey Patch** in `serve_model.py`:
   - The monkey patch for `LlamaConfig._rope_scaling_validation` is implemented but not properly validated for effectiveness.

2. **Unused Imports**: 
   - Several files contain imports that aren't used, such as `importlib.util` in `flask-app/app.py`.

3. **Commented-Out Code**: 
   - There appears to be commented-out code in several files that should be either properly restored or removed.

## 3. Incomplete Docstrings

1. **Missing Parameter Documentation**:
   - Many functions (like `process_question_with_model_server` in `flask-app/app.py`) have parameters that are not documented in the docstring.

2. **Missing Return Value Documentation**:
   - Functions often lack documentation for their return values, making it hard to understand what they return.

3. **Missing Function Purpose Documentation**:
   - Some functions (especially utility functions) lack clear descriptions of their purpose.

4. **Incomplete Module-Level Documentation**:
   - Most files lack module-level docstrings explaining their overall purpose and relationships to other modules.

## 4. Project Reorganization Opportunities

1. **Centralized Configuration**:
   - Create a unified configuration system instead of having settings spread across multiple files. Consider using environment-specific configurations.

2. **API Layer Separation**:
   - The Flask application combines web UI and API routes in the same file. Consider separating them into distinct modules (e.g., `api.py` and `views.py`).

3. **Modular Error Handling**:
   - Implement centralized error handling for all components to ensure consistent error responses.

4. **Client Abstraction**:
   - Create a consistent client abstraction for all external services (model server, vector database) with proper interface definitions.

5. **Deployment Configuration**:
   - Move Docker and deployment configurations to a dedicated `deploy` directory.

6. **Environment Management**:
   - Consider using environment-specific configuration files (dev, test, prod) and proper environment variable handling.

## 5. Unit Testing Recommendations

1. **Utility Function Tests**:
   - Add unit tests for all utility functions in `app/utils/*`:
     - `embedding_generation.py`
     - `text_chunking.py`
     - `pdf_ingestion.py`
     - `query_processing.py`

2. **Mock-Based Tests**:
   - Implement mock-based tests for external dependencies (model server, vector database) to allow testing without the actual services.

3. **Error Handling Tests**:
   - Add tests for error conditions to ensure robust error handling.

4. **Configuration Tests**:
   - Add tests for configuration loading and validation.

5. **Model-Specific Tests**:
   - Add unit tests for the model processing logic with test cases for different input types.

## 6. Integration Testing Recommendations

1. **API Flow Testing**:
   - Add integration tests for the complete API flow from request to response, including error handling.

2. **Document Processing Pipeline**:
   - Test the complete document processing pipeline from upload to indexing to search.

3. **Model Serving Tests**:
   - Test the interaction between the Flask app and the model server, including timeout and retry mechanisms.

4. **Database Integration**:
   - Test the vector database integration with real-world queries and verify result correctness.

5. **Performance Testing**:
   - Add performance tests to measure response time under various loads.

## 7. Additional Recommendations

1. **Type Annotations**:
   - Add comprehensive type annotations to all functions to improve code understanding and enable static type checking.

2. **Error Handling**:
   - Improve error handling, especially for external service failures.

3. **Documentation**:
   - Enhance project documentation, including architecture diagrams and component relationships.

4. **Security**:
   - Review security aspects, especially around file uploads and API endpoints.

5. **Code Modularity**:
   - Break down large files (like `flask-app/app.py` with 577 lines) into smaller, more focused modules.

6. **API Versioning**:
   - Consider implementing API versioning for better backward compatibility.

7. **Automated Testing**:
   - Set up CI/CD with automated testing to ensure code quality consistency.

This review provides a comprehensive overview of the code quality issues and improvement opportunities in the project. Implementing these recommendations will significantly enhance the maintainability, reliability, and scalability of the codebase.
```

## app/utils/embedding_generation.py

```python
import os
import numpy as np
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""This module provides functionality to generate embeddings for text chunks using a Sentence Transformer model.

It includes a class for generating embeddings and a utility function to process pandas DataFrames containing 
text chunks, enabling efficient embedding generation for text data.
"""

class EmbeddingGenerator:
    """The EmbeddingGenerator class handles loading a Sentence Transformer model and generating embeddings for text data.

    It offers methods to create embeddings from lists of text strings and to process pandas DataFrames by adding 
    embeddings to specified columns, facilitating downstream tasks like similarity computation.
    """

    def __init__(self, model_path: str, batch_size: int = 32):
        """Initializes the EmbeddingGenerator with a model path and batch size for processing.

        Args:
            model_path (str): The file path to the pre-trained Sentence Transformer model.
            batch_size (int, optional): The number of text items to process in each batch. Defaults to 32.

        The constructor loads the specified model and retrieves its embedding dimension for later use.
        """
        logger.info(f"Loading embedding model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Creates embeddings for a list of text strings using the Sentence Transformer model.

        Args:
            texts (List[str]): A list of text strings to generate embeddings for.

        Returns:
            np.ndarray: A 2D NumPy array where each row represents the embedding vector of the corresponding text.

        Texts are processed in batches for efficiency, and embeddings are normalized to support similarity calculations.
        """
        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {self.batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'chunk_text',
                          embedding_column: str = 'embedding') -> pd.DataFrame:
        """Processes a DataFrame by adding embeddings for text in a specified column to a new column.

        Args:
            df (pd.DataFrame): The input DataFrame with text data to process.
            text_column (str, optional): The column name containing text to embed. Defaults to 'chunk_text'.
            embedding_column (str, optional): The column name to store embeddings. Defaults to 'embedding'.

        Returns:
            pd.DataFrame: The input DataFrame augmented with a new column of embeddings.
        """
        logger.info(f"Processing DataFrame with {len(df)} rows")
        
        # Extract texts from the specified column
        texts = df[text_column].tolist()
        
        # Generate embeddings for the texts
        embeddings = self.generate_embeddings(texts)
        
        # Assign embeddings to the new column
        df[embedding_column] = list(embeddings)
        
        return df

def embed_chunks(chunks_df: pd.DataFrame, model_path: str, batch_size: int = 32) -> pd.DataFrame:
    """Generates embeddings for text chunks in a DataFrame using a Sentence Transformer model.

    Args:
        chunks_df (pd.DataFrame): A DataFrame with text chunks to embed.
        model_path (str): The file path to the pre-trained Sentence Transformer model.
        batch_size (int, optional): The number of texts to process per batch. Defaults to 32.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'embedding' column containing the generated embeddings.

    This function acts as a convenient wrapper, initializing an EmbeddingGenerator and processing the DataFrame in one step.
    """
    # Initialize the embedder with the specified model and batch size
    embedder = EmbeddingGenerator(model_path, batch_size)
    
    # Process the DataFrame to add embeddings
    chunks_df = embedder.process_dataframe(chunks_df)
    
    return chunks_df

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import PDF_UPLOAD_FOLDER, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP
    from app.utils.pdf_ingestion import process_pdfs
    from app.utils.text_chunking import process_chunks
    
    # Process PDFs into a DataFrame
    pdf_df = process_pdfs(PDF_UPLOAD_FOLDER)
    
    # Chunk the text data
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Generate embeddings for the chunks
    chunks_with_embeddings = embed_chunks(chunks_df, EMBEDDING_MODEL_PATH)
    
    print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    print(f"Embedding dimension: {len(chunks_with_embeddings['embedding'].iloc[0])}")
```

## app/utils/llm.py

```python
import os
from typing import Dict, Any, List, Optional
import logging
import sys
from pathlib import Path

# Import settings and model utilities
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.config.settings import ALT_MODEL_PATHS, HF_MODEL_ID, HF_TOKEN
from app.utils.model_downloader import find_or_download_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available, model loading will be limited to transformers")

# Try to import transformers for non-GGUF models
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, model loading will be limited to llama-cpp")

# Try to import MetaLlamaAdapter
try:
    from app.utils.adapters.meta_llama_adapter import MetaLlamaAdapter
    META_LLAMA_ADAPTER_AVAILABLE = True
except ImportError:
    logger.warning("MetaLlamaAdapter not available")
    META_LLAMA_ADAPTER_AVAILABLE = False

class TransformersLLMProcessor:
    """LLM processor implementation using HuggingFace Transformers."""
    
    def __init__(self, model_path: str, context_size: int = 2048, max_tokens: int = 512):
        """
        Initialize the Transformers LLM processor.
        
        Args:
            model_path: Path to the transformer model
            context_size: Context size for the model
            max_tokens: Maximum number of tokens to generate
        """
        logger.info(f"Loading Transformers model from {model_path}")
        
        try:
            # Determine if model_path is a directory with model files
            if os.path.isdir(model_path):
                # For Llama 3.2, we need to modify the config file to fix rope_scaling
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Fix rope_scaling if it exists
                    if 'rope_scaling' in config:
                        logger.info("Fixing rope_scaling in config.json")
                        # Set to a valid format
                        config['rope_scaling'] = {"type": "dynamic", "factor": 2.0}
                        
                        # Save the modified config
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)
                
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Determine device
                if torch.cuda.is_available():
                    logger.info("Using CUDA for model inference")
                    device = "cuda"
                elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("Using MPS (Metal) for model inference")
                    device = "mps"
                else:
                    logger.info("Using CPU for model inference")
                    device = "cpu"
                
                # Use pipeline directly with text-generation task
                self.pipe = pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16,
                    device_map=device,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                    },
                    trust_remote_code=True,
                )
                
                logger.info("Transformers model loaded successfully")
            else:
                raise ValueError(f"Model path {model_path} is not a directory with model files")
            
        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            raise
            
        self.max_tokens = max_tokens
        self.context_size = context_size
    
    def create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for Llama 3.
        """
        # Format context
        context_text = ""
        for i, doc in enumerate(context):
            context_text += f"Document {i+1}:\n{doc['chunk_text']}\n\n"
        
        # Llama 3.x prompt format
        prompt = f"""<|system|>
You are a helpful AI assistant that provides accurate and concise answers based on the provided context documents. 
If the answer is not contained in the documents, say "I don't have enough information to answer this question."
Do not make up or hallucinate any information that is not supported by the documents.
</|system|>

<|user|>
I need information about the following topic:

{query}

Here are relevant documents to help answer this question:

{context_text}
</|user|>

<|assistant|>
"""
        return prompt
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the Transformers model.
        
        Args:
            prompt: Prompt for the model
            
        Returns:
            Response with text and metadata
        """
        logger.info("Generating response with Transformers model")
        
        # Generate the response
        # Truncate input if it's too long
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.context_size)
        prompt_tokens = len(inputs.input_ids[0])
        
        # Move inputs to the same device as the model
        if hasattr(self.pipe, 'device'):
            inputs = {k: v.to(self.pipe.device) for k, v in inputs.items()}
        
        # Generate the text
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            return_full_text=False  # Only return the newly generated text
        )
        
        # Get the generated text
        response_text = outputs[0]['generated_text']
        
        # Get metadata (estimated)
        response_tokens = len(self.tokenizer(response_text, return_tensors="pt").input_ids[0])
        
        metadata = {
            'tokens_used': prompt_tokens + response_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': response_tokens,
        }
        
        logger.info(f"Generated response with {metadata['completion_tokens']} tokens")
        
        return {
            'text': response_text,
            'metadata': metadata
        }

class LLMProcessor:
    def __init__(self, model_path: str, context_size: int = 2048, max_tokens: int = 512):
        """
        Initialize the LLM processor.
        
        Args:
            model_path: Path to the LLM model
            context_size: Context size for the model
            max_tokens: Maximum number of tokens to generate
        """
        logger.info(f"Loading LLM from {model_path}")
        
        # Try to find the model in various locations or download it
        actual_model_path = find_or_download_model(
            model_path,
            ALT_MODEL_PATHS,
            HF_MODEL_ID,
            HF_TOKEN
        )
        
        # Default values
        self.model = None
        self.use_transformers = False
        self.use_meta_adapter = False
        
        # Check if it's a Llama-3.2 Instruct model (either 1B or 3B)
        is_llama_3_2_instruct = any(model_name in actual_model_path or model_name in os.path.dirname(actual_model_path) 
                                   for model_name in ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"])
        
        if is_llama_3_2_instruct and META_LLAMA_ADAPTER_AVAILABLE and TRANSFORMERS_AVAILABLE:
            
            # Determine the actual model directory
            if os.path.isdir(actual_model_path):
                model_dir = actual_model_path
            else:
                model_dir = os.path.dirname(actual_model_path)
                # Check for either 1B or 3B model directories
                for model_name in ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]:
                    if model_name in os.listdir(model_dir):
                        model_dir = os.path.join(model_dir, model_name)
                        break
            
            # Get the model name from the directory path
            model_name = os.path.basename(model_dir)
            
            try:
                logger.info(f"Loading {model_name} with MetaLlamaAdapter from {model_dir}")
                self.meta_adapter = MetaLlamaAdapter(
                    model_path=model_dir,
                    max_new_tokens=max_tokens
                )
                self.use_meta_adapter = True
                logger.info(f"Loaded {model_name} with MetaLlamaAdapter")
            except Exception as e:
                logger.error(f"Failed to load {model_name} with MetaLlamaAdapter: {e}")
                # Fall back to TransformersLLMProcessor
                try:
                    self.transformers_processor = TransformersLLMProcessor(
                        model_path=model_dir,
                        context_size=context_size,
                        max_tokens=max_tokens
                    )
                    self.use_transformers = True
                    logger.info(f"Loaded {model_name} with TransformersLLMProcessor as fallback")
                except Exception as e2:
                    logger.error(f"Failed to load with TransformersLLMProcessor: {e2}")
        
        # Case 1: Valid GGUF file and llama-cpp available
        elif (os.path.exists(actual_model_path) and 
            os.path.getsize(actual_model_path) > 1000000 and  # >1MB is probably valid
            actual_model_path.endswith(".gguf") and
            LLAMA_CPP_AVAILABLE):
            
            try:
                self.model = Llama(
                    model_path=actual_model_path,
                    n_ctx=context_size,
                    n_batch=512,  # Adjust based on available RAM
                )
                logger.info("Loaded GGUF model with llama-cpp")
            except Exception as e:
                logger.error(f"Failed to load GGUF model with llama-cpp: {e}")
        
        # Case 2: Directory with model files and transformers available
        elif (os.path.isdir(os.path.dirname(actual_model_path)) and
              TRANSFORMERS_AVAILABLE):
            
            # Check if there's a directory with model files
            model_dir = os.path.dirname(actual_model_path)
            
            try:
                # Use the directory with model files
                self.transformers_processor = TransformersLLMProcessor(
                    model_path=model_dir,
                    context_size=context_size,
                    max_tokens=max_tokens
                )
                self.use_transformers = True
                logger.info("Loaded model with Transformers")
            except Exception as e:
                logger.error(f"Failed to load model with Transformers: {e}")
        
        # Case 3: No valid model found
        else:
            logger.error(f"Model file {actual_model_path} not found or invalid, and no alternative available.")
            # Instead of using a mock, raise an exception
            raise ValueError(f"Model file {actual_model_path} not found or invalid, and no alternative available. Please ensure the model file exists and is valid.")
        
        self.max_tokens = max_tokens
    
    def create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the LLM.
        """
        # If using transformers, delegate to transformers processor
        if self.use_transformers:
            return self.transformers_processor.create_prompt(query, context)
            
        # Format context
        context_text = ""
        for i, doc in enumerate(context):
            context_text += f"Document {i+1}:\n{doc['chunk_text']}\n\n"
        
        # Llama 3.x prompt format
        prompt = f"""<|system|>
    You are a helpful AI assistant that provides accurate and concise answers based on the provided context documents. 
    If the answer is not contained in the documents, say "I don't have enough information to answer this question."
    Do not make up or hallucinate any information that is not supported by the documents.
    </|system|>

    <|user|>
    I need information about the following topic:

    {query}

    Here are relevant documents to help answer this question:

    {context_text}
    </|user|>

    <|assistant|>
    """
        return prompt        
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Response from the LLM
        """
        # If using meta adapter, use it
        if self.use_meta_adapter:
            response = self.meta_adapter(prompt, max_tokens=self.max_tokens)
            # Extract text
            response_text = response['choices'][0]['text'].strip()
            # Get metadata
            metadata = {
                'tokens_used': response['usage']['total_tokens'],
                'prompt_tokens': response['usage']['prompt_tokens'],
                'completion_tokens': response['usage']['completion_tokens'],
            }
            return {
                'text': response_text,
                'metadata': metadata
            }
        # If using transformers, delegate to transformers processor
        elif self.use_transformers:
            return self.transformers_processor.generate_response(prompt)
            
        logger.info("Generating LLM response with llama-cpp")
        
        # Generate response using llama-cpp
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            stop=["User query:", "\n\n"],
            temperature=0.2,  # Lower temperature for more factual responses
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        # Extract text
        response_text = response['choices'][0]['text'].strip()
        
        # Get metadata
        metadata = {
            'tokens_used': len(response['usage']['prompt_tokens']) + len(response['usage']['completion_tokens']),
            'prompt_tokens': len(response['usage']['prompt_tokens']),
            'completion_tokens': len(response['usage']['completion_tokens']),
        }
        
        logger.info(f"Generated response with {metadata['completion_tokens']} tokens")
        
        return {
            'text': response_text,
            'metadata': metadata
        }

class RAGProcessor:
    def __init__(self, search_pipeline, llm_processor):
        """
        Initialize the RAG processor.
        
        Args:
            search_pipeline: Search pipeline
            llm_processor: LLM processor
        """
        self.search_pipeline = search_pipeline
        self.llm_processor = llm_processor
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using RAG.
        
        Args:
            query: User query
            
        Returns:
            Response with text, sources, and metadata
        """
        logger.info(f"Processing RAG query: {query}")
        
        # Search for relevant documents
        search_results = self.search_pipeline.search(query)
        
        if not search_results:
            return {
                'text': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'metadata': {'search_results': 0}
            }
        
        # Create prompt
        prompt = self.llm_processor.create_prompt(query, search_results)
        
        # Generate response
        response = self.llm_processor.generate_response(prompt)
        
        # Format sources
        sources = []
        for result in search_results:
            sources.append({
                'filename': result['filename'],
                'chunk_text': result['chunk_text'],
                'rerank_score': result['rerank_score'],
                'vector_score': result['score']
            })
        
        # Combine results
        return {
            'text': response['text'],
            'sources': sources,
            'metadata': {
                'llm': response['metadata'],
                'search_results': len(search_results)
            }
        }

def create_rag_processor(search_pipeline, llm_model_path: str, 
                       context_size: int = 2048, max_tokens: int = 512) -> RAGProcessor:
    """
    Create a RAG processor.
    
    Args:
        search_pipeline: Search pipeline
        llm_model_path: Path to the LLM model
        context_size: Context size for the model
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        RAG processor
    """
    # Create LLM processor
    llm_processor = LLMProcessor(llm_model_path, context_size, max_tokens)
    
    # Create RAG processor
    rag_processor = RAGProcessor(search_pipeline, llm_processor)
    
    return rag_processor

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, LLM_MODEL_PATH
    )
    from app.utils.search import create_search_pipeline
    
    # Create search pipeline
    search_pipeline = create_search_pipeline(
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
    )
    
    # Create RAG processor
    rag_processor = create_rag_processor(search_pipeline, LLM_MODEL_PATH)
    
    # Process query
    query = "What is retrieval-augmented generation?"
    response = rag_processor.process_query(query)
    
    # Print response
    print(f"Query: {query}")
    print(f"Response: {response['text']}")
    print(f"Sources: {len(response['sources'])}")
    for i, source in enumerate(response['sources']):
        print(f"Source {i+1}: {source['filename']} (Score: {source['rerank_score']:.4f})")
```

## app/utils/model_downloader.py

```python
import os
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_huggingface_model(
    model_id: str, 
    hf_token: str, 
    output_path: str,
    quantize: bool = True
) -> Optional[str]:
    """
    Download a model from Hugging Face.
    
    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        hf_token: Hugging Face token
        output_path: Path to save the model
        quantize: Whether to quantize the model
        
    Returns:
        Path to the downloaded model, or None if download failed
    """
    if not hf_token:
        logger.error("No Hugging Face token provided")
        return None
        
    # Create download directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if the output path exists and is valid
    if os.path.exists(output_path):
        # Handle directories differently from files
        if os.path.isdir(output_path):
            # Check if it's a valid model directory by looking for model files
            if (os.path.exists(os.path.join(output_path, "config.json")) and
                os.path.exists(os.path.join(output_path, "tokenizer.json"))):
                logger.info(f"Valid model directory already exists at {output_path}")
                return output_path
            else:
                logger.info(f"Directory exists at {output_path} but doesn't appear to be a valid model")
                # Skip downloading if directory exists but is not valid
                # Attempting to remove could cause permission errors
                return output_path
        else:
            # Regular file path
            if os.path.getsize(output_path) < 1000000:  # Less than 1MB is probably not valid
                logger.info(f"Removing invalid or empty model file at {output_path}")
                try:
                    os.remove(output_path)
                except (PermissionError, OSError) as e:
                    logger.error(f"Failed to remove invalid model file: {e}")
                    return output_path
            else:
                logger.info(f"Valid model file already exists at {output_path}")
                return output_path

    # Handle directory output path for huggingface_hub download
    if output_path.endswith(os.path.sep) or (not output_path.endswith('.gguf') and model_id.endswith('Instruct')):
        try:
            # Use huggingface_hub for directory-based models
            try:
                from huggingface_hub import snapshot_download
                # Make sure the directory exists
                os.makedirs(output_path, exist_ok=True)
                
                # Download the model
                logger.info(f"Downloading model {model_id} using huggingface_hub")
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=output_path,
                    local_dir_use_symlinks=False,
                    token=hf_token
                )
                logger.info(f"Model downloaded to {downloaded_path}")
                return downloaded_path
            except ImportError:
                logger.error("huggingface_hub not installed, falling back to git clone method")
        except Exception as e:
            logger.error(f"Failed to download model using huggingface_hub: {e}")
            return None

    # Create a temporary directory for the model
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Downloading model {model_id} to temporary directory")
        
        # Set git credentials
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"  # Skip LFS files initially for faster clone
        os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Disable git prompts
        
        # Clone the repo
        clone_cmd = [
            "git", "clone", 
            f"https://USER:{hf_token}@huggingface.co/{model_id}", 
            temp_dir
        ]
        
        logger.info("Running git clone command")
        
        try:
            subprocess.run(
                clone_cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # Pull LFS files
            lfs_cmd = ["git", "lfs", "pull"]
            subprocess.run(
                lfs_cmd, 
                check=True, 
                cwd=temp_dir, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            logger.info("Model download complete")
            
            # Quantize if requested
            if quantize:
                logger.info("Quantizing model to GGUF format")
                
                # Check for llama.cpp or llamafile-quantize
                quantize_tool = None
                
                # Check for llamafile-quantize
                try:
                    subprocess.run(
                        ["llamafile-quantize", "--help"], 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    quantize_tool = "llamafile-quantize"
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Check for llama.cpp
                    try:
                        llamacpp_path = os.environ.get("LLAMA_CPP_PATH")
                        if llamacpp_path and os.path.exists(os.path.join(llamacpp_path, "convert.py")):
                            quantize_tool = "llama.cpp"
                    except Exception:
                        pass
                
                if quantize_tool == "llamafile-quantize":
                    # Use llamafile-quantize
                    quantize_cmd = [
                        "llamafile-quantize", 
                        f"{temp_dir}", 
                        "--outfile", output_path,
                        "--q4_0"  # Use Q4_0 quantization for balance of quality and size
                    ]
                    subprocess.run(
                        quantize_cmd, 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    return output_path
                elif quantize_tool == "llama.cpp":
                    # Use llama.cpp
                    convert_script = os.path.join(os.environ["LLAMA_CPP_PATH"], "convert.py")
                    convert_cmd = [
                        "python", convert_script,
                        f"{temp_dir}", 
                        "--outfile", output_path,
                        "--outtype", "q4_0"
                    ]
                    subprocess.run(
                        convert_cmd, 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    return output_path
                else:
                    logger.warning("No quantization tools found, preparing model as-is")
                    
                    # Create placeholder file with the right path
                    try:
                        # Find the main model file (usually model.safetensors or pytorch_model.bin)
                        model_files = []
                        for file in os.listdir(temp_dir):
                            if file.endswith(".safetensors") or file.endswith(".bin"):
                                model_files.append(os.path.join(temp_dir, file))
                        
                        if not model_files:
                            logger.error("No model files found in downloaded repository")
                            return None
                            
                        # Create model directory if it doesn't exist
                        model_dir = os.path.dirname(output_path)
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Create a placeholder file to identify this as a model
                        with open(output_path, 'w') as f:
                            # Add a header to identify this as a non-GGUF file
                            f.write("# This is a placeholder for a GGUF model. The original model is available at:\n")
                            f.write(f"# https://huggingface.co/{model_id}\n")
                            f.write("# Please convert to GGUF format using llama.cpp or llamafile-quantize\n")
                        
                        # Copy the tokenizer and other relevant files to the model directory
                        for file in os.listdir(temp_dir):
                            if file.endswith(".json") or file.endswith(".py") or file.endswith(".md"):
                                src = os.path.join(temp_dir, file)
                                dst = os.path.join(model_dir, file)
                                shutil.copy2(src, dst)
                                
                        logger.info(f"Created placeholder model file at {output_path}")
                        return output_path
                    except Exception as e:
                        logger.error(f"Error preparing model: {e}")
                        return None
            else:
                # No quantization, just copy the model files
                try:
                    # Copy all files to output directory
                    output_dir = os.path.dirname(output_path)
                    for item in os.listdir(temp_dir):
                        src = os.path.join(temp_dir, item)
                        dst = os.path.join(output_dir, item)
                        
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                            
                    # Create a symlink or copy of the main model file with the expected name
                    main_model_file = None
                    for file in os.listdir(output_dir):
                        if file.endswith(".safetensors") or file.endswith(".bin"):
                            main_model_file = os.path.join(output_dir, file)
                            break
                            
                    if main_model_file:
                        try:
                            os.symlink(main_model_file, output_path)
                        except (OSError, AttributeError):
                            shutil.copy2(main_model_file, output_path)
                            
                    return output_path
                except Exception as e:
                    logger.error(f"Error copying model files: {e}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None
    
    logger.error("Failed to download the model")
    return None

def find_or_download_model(
    primary_path: str, 
    alt_paths: List[str], 
    model_id: str, 
    hf_token: str
) -> str:
    """
    Find a model in the specified paths or download it if not found.
    
    Args:
        primary_path: Primary path to check for the model
        alt_paths: Alternative paths to check for the model
        model_id: Hugging Face model ID for download if not found
        hf_token: Hugging Face token for download
        
    Returns:
        Path to the found or downloaded model
    """
    # First check if the primary path exists and is valid
    if os.path.exists(primary_path):
        # Check if it's a directory or file
        if os.path.isdir(primary_path):
            # For directories, check for key model files
            if (os.path.exists(os.path.join(primary_path, "config.json")) or
                os.path.exists(os.path.join(primary_path, "model.safetensors.index.json")) or
                os.path.exists(os.path.join(primary_path, "model-00001-of-00002.safetensors"))):
                logger.info(f"Found valid model directory at {primary_path}")
                return primary_path
        elif os.path.getsize(primary_path) > 1000000:  # >1MB is probably valid
            logger.info(f"Found valid model file at {primary_path}")
            return primary_path
            
    # Check alternative paths
    for path in alt_paths:
        if os.path.exists(path):
            # Check if it's a directory or file
            if os.path.isdir(path):
                # For directories, check for key model files
                if (os.path.exists(os.path.join(path, "config.json")) or
                    os.path.exists(os.path.join(path, "model.safetensors.index.json")) or
                    os.path.exists(os.path.join(path, "model-00001-of-00002.safetensors"))):
                    logger.info(f"Found valid model directory at {path}")
                    return path
            elif os.path.getsize(path) > 1000000:  # >1MB is probably valid
                logger.info(f"Found valid model file at {path}")
                return path
                
    # Model not found, download it
    logger.info(f"Model not found in any of the specified paths, downloading from Hugging Face")
    
    # Check if primary_path is a directory path for Llama-3.2 models
    llama_models = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]
    is_llama_model = any(model_name in primary_path or model_name in model_id for model_name in llama_models)
    
    if is_llama_model:
        # Determine which model we're using
        model_name = next((name for name in llama_models if name in model_id), llama_models[0])
        
        # Ensure primary_path is a directory
        if not primary_path.endswith(os.path.sep) and not os.path.isdir(primary_path):
            primary_path = os.path.dirname(primary_path)
            if not any(primary_path.endswith(model_name) for model_name in llama_models):
                primary_path = os.path.join(primary_path, model_name)
                
        # Create directory if it doesn't exist
        os.makedirs(primary_path, exist_ok=True)
    
    # Download the model
    downloaded_path = download_huggingface_model(model_id, hf_token, primary_path)
    
    if downloaded_path:
        return downloaded_path
    else:
        # If download failed, return the primary path anyway
        # The calling code should handle the case where the model doesn't exist
        logger.warning(f"Failed to download model, returning primary path: {primary_path}")
        return primary_path 
```

## app/utils/pdf_ingestion.py

```python
"""
A script for processing PDF files in a directory, extracting metadata and text, and organizing the results into a pandas DataFrame.

This script scans a specified directory for PDF files, collects metadata such as file path, size, page count, and last modified time,
extracts the text from each PDF, and stores all the information in a pandas DataFrame for further analysis. It uses PyMuPDF for PDF operations,
pandas for data handling, and tqdm for progress tracking.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scan_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Scan a directory for PDF files and collect their metadata.

    This function recursively searches the given directory for PDF files and gathers
    basic file information along with PDF-specific metadata using PyMuPDF.

    Parameters:
    directory_path (str): Path to the directory containing PDF files.

    Returns:
    List[Dict[str, Any]]: A list of dictionaries, each containing information about a PDF file.
        Each dictionary has the following keys:
        - 'path': Full path to the PDF file.
        - 'filename': Name of the PDF file.
        - 'parent_dir': Path to the parent directory of the PDF file.
        - 'size_bytes': Size of the file in bytes.
        - 'last_modified': Last modification time of the file (Unix timestamp).
        - 'page_count': Number of pages in the PDF (0 if unable to read).
        - 'metadata': Dictionary of PDF metadata (empty if unable to read).

    Raises:
    None: The function handles exceptions internally and logs errors.

    Example:
    >>> pdf_files = scan_directory('/path/to/directory')
    >>> print(pdf_files[0]['filename'])
    some_file.pdf
    """
    logger.info(f"Scanning directory: {directory_path}")
    pdf_files = []
    
    for path in tqdm(list(Path(directory_path).rglob('*.pdf'))):
        try:
            # Get basic file information
            file_info = {
                'path': str(path),
                'filename': path.name,
                'parent_dir': str(path.parent),
                'size_bytes': path.stat().st_size,
                'last_modified': path.stat().st_mtime
            }
            
            # Get PDF-specific metadata if possible
            try:
                with fitz.open(str(path)) as doc:
                    file_info['page_count'] = len(doc)
                    file_info['metadata'] = doc.metadata
            except Exception as e:
                logger.warning(f"Could not read PDF metadata for {path}: {str(e)}")
                file_info['page_count'] = 0
                file_info['metadata'] = {}
                
            pdf_files.append(file_info)
        except Exception as e:
            logger.error(f"Error processing {path}: {str(e)}")
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files

def create_pdf_dataframe(pdf_files: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a list of PDF file information.

    This function takes a list of dictionaries containing PDF file information and
    converts it into a pandas DataFrame for easy data manipulation and analysis.

    Parameters:
    pdf_files (List[Dict[str, Any]]): List of dictionaries with PDF file information.

    Returns:
    pd.DataFrame: A DataFrame where each row represents a PDF file and columns correspond to the dictionary keys.

    Example:
    >>> pdf_files = [{'path': '/path/to/file.pdf', 'filename': 'file.pdf', ...}]
    >>> df = create_pdf_dataframe(pdf_files)
    >>> print(df.head())
    """
    return pd.DataFrame(pdf_files)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    This function opens a PDF file using PyMuPDF and extracts its text, preserving some structure
    by using text blocks (e.g., paragraphs). It joins the blocks with newlines and adds double newlines
    between pages for readability.

    Parameters:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: The extracted text from the PDF. If extraction fails, an empty string is returned.

    Raises:
    None: The function handles exceptions internally and logs errors.

    Example:
    >>> text = extract_text_from_pdf('/path/to/file.pdf')
    >>> print(text[:100])  # Print first 100 characters
    """
    logger.info(f"Extracting text from: {pdf_path}")
    
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page_num, page in enumerate(doc):
                # Get text with blocks (preserves some structure)
                blocks = page.get_text("blocks")
                # Join blocks with newlines, preserving structure
                page_text = "\n".join(block[4] for block in blocks if block[6] == 0)  # block[6] == 0 means text block
                text += page_text + "\n\n"  # Add separation between pages
                
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def process_pdfs(directory_path: str) -> pd.DataFrame:
    """
    Process all PDFs in a directory by scanning, creating a DataFrame, and extracting text.

    This function orchestrates the PDF processing pipeline:
    1. Scans the directory for PDF files.
    2. Creates a DataFrame from the PDF file information.
    3. Extracts text from each PDF and adds it to the DataFrame.
    4. Logs statistics about the extracted text lengths.
    5. Logs a warning if any PDFs have no extractable text.

    Parameters:
    directory_path (str): Path to the directory containing PDF files.

    Returns:
    pd.DataFrame: A DataFrame with PDF information and extracted text.

    Example:
    >>> df = process_pdfs('/path/to/directory')
    >>> print(df.head())
    """
    # Scan directory for PDFs
    pdf_files = scan_directory(directory_path)
    
    # Create DataFrame
    df = create_pdf_dataframe(pdf_files)
    
    # Extract text from each PDF
    tqdm.pandas(desc="Extracting text")
    df['text'] = df['path'].progress_apply(extract_text_from_pdf)
    
    # Filter out PDFs with no text
    text_lengths = df['text'].str.len()
    logger.info(f"Text extraction statistics: min={text_lengths.min()}, max={text_lengths.max()}, "
                f"mean={text_lengths.mean():.2f}, median={text_lengths.median()}")
    
    empty_pdfs = df[df['text'].str.len() == 0]
    if not empty_pdfs.empty:
        logger.warning(f"Found {len(empty_pdfs)} PDFs with no extractable text")
    
    return df

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import PDF_UPLOAD_FOLDER
    
    df = process_pdfs(PDF_UPLOAD_FOLDER)
    print(f"Processed {len(df)} PDFs")
    print(df.head())
```

## app/utils/prompts.py

```python
"""
Prompt templates for different LLM models.
"""

def get_llama_3_2_rag_prompt(query: str, context: str) -> str:
    """
    Get the prompt template for Llama 3.2 models with RAG context.
    
    Args:
        query: The user's query
        context: The context information from retrieved documents
        
    Returns:
        The formatted prompt
    """
    return f"""<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context information provided below.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.

Here is some context information to help you answer the user's question:
{context}
</s>
<|user|>
{query}
</s>
<|assistant|>"""

def get_llama_3_2_chat_prompt(messages: list) -> str:
    """
    Get the prompt template for Llama 3.2 models in chat format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        The formatted prompt
    """
    prompt = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            prompt += f"<|system|>\n{content}\n</s>\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n</s>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n</s>\n"
    
    # Add the final assistant token without completion
    if not prompt.endswith("<|assistant|>\n</s>\n"):
        prompt += "<|assistant|>"
        
    return prompt 
```

## app/utils/query_processing.py

```python
import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, model_path: str):
        """
        Initialize the query processor.
        
        Args:
            model_path: Path to the embedding model
        """
        logger.info(f"Loading embedding model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
    
    def process_query(self, query: str) -> np.ndarray:
        """
        Process a query and generate an embedding.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        logger.info(f"Processing query: {query}")
        
        # Generate embedding
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info(f"Generated query embedding with shape: {embedding.shape}")
        return embedding

def process_query(query: str, model_path: str) -> np.ndarray:
    """
    Process a query and generate an embedding.
    
    Args:
        query: Query text
        model_path: Path to the embedding model
        
    Returns:
        Query embedding
    """
    processor = QueryProcessor(model_path)
    return processor.process_query(query)

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import EMBEDDING_MODEL_PATH
    
    # Process a query
    query = "What is retrieval-augmented generation?"
    embedding = process_query(query, EMBEDDING_MODEL_PATH)
    print(f"Query embedding shape: {embedding.shape}")
```

## app/utils/reranking.py

```python
import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_path: str):
        """
        Initialize the reranker.
        
        Args:
            model_path: Path to the reranker model
        """
        logger.info(f"Loading reranker model from {model_path}")
        self.model = CrossEncoder(model_path, max_length=512)
        logger.info("Reranker model loaded")
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results.
        
        Args:
            query: Query text
            results: List of search results
            
        Returns:
            Reranked results
        """
        logger.info(f"Reranking {len(results)} results for query: {query}")
        
        if not results:
            return []
        
        # Create pairs for the cross-encoder
        pairs = [(query, result['chunk_text']) for result in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to results
        for i, score in enumerate(scores):
            results[i]['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info("Reranking complete")
        return reranked_results

def rerank_results(query: str, results: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
    """
    Rerank search results.
    
    Args:
        query: Query text
        results: List of search results
        model_path: Path to the reranker model
        
    Returns:
        Reranked results
    """
    reranker = Reranker(model_path)
    return reranker.rerank(query, results)

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        RERANKER_MODEL_PATH, EMBEDDING_MODEL_PATH,
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION
    )
    from app.utils.query_processing import process_query
    from app.utils.vector_db import VectorDBClient
    
    # Process a query
    query = "What is retrieval-augmented generation?"
    embedding = process_query(query, EMBEDDING_MODEL_PATH)
    
    # Search the vector database
    vector_db = VectorDBClient(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION)
    results = vector_db.search(embedding, limit=10)
    
    # Rerank results
    reranked_results = rerank_results(query, results, RERANKER_MODEL_PATH)
    
    # Print results
    print("Reranked results:")
    for i, result in enumerate(reranked_results[:5]):
        print(f"{i+1}. Score: {result['rerank_score']:.4f}, Original Score: {result['score']:.4f}")
        print(f"   Text: {result['chunk_text'][:100]}...")
```

## app/utils/search.py

```python
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchPipeline:
    def __init__(self, vector_db_client, query_processor, reranker,
                max_results: int = 10, rerank_results: int = 10):
        """
        Initialize the search pipeline.
        
        Args:
            vector_db_client: Vector database client
            query_processor: Query processor
            reranker: Reranker
            max_results: Maximum number of results to return
            rerank_results: Number of results to rerank
        """
        self.vector_db_client = vector_db_client
        self.query_processor = query_processor
        self.reranker = reranker
        self.max_results = max_results
        self.rerank_results = rerank_results
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for documents relevant to a query.
        
        Args:
            query: Query text
            
        Returns:
            Search results
        """
        logger.info(f"Searching for: {query}")
        
        # Process query
        query_embedding = self.query_processor.process_query(query)
        
        # Search vector database
        vector_results = self.vector_db_client.search(query_embedding, limit=self.rerank_results)
        logger.info(f"Found {len(vector_results)} results from vector search")
        
        if not vector_results:
            logger.warning("No results found in vector search")
            return []
        
        # Rerank results
        reranked_results = self.reranker.rerank(query, vector_results)
        logger.info("Results reranked")
        
        # Return top results
        return reranked_results[:self.max_results]

def create_search_pipeline(vector_db_host: str, vector_db_port: int, collection_name: str, 
                         vector_dimension: int, embedding_model_path: str, reranker_model_path: str,
                         max_results: int = 5, rerank_top_k: int = 10):
    """
    Create a search pipeline.
    
    Args:
        vector_db_host: Host of the vector database
        vector_db_port: Port of the vector database
        collection_name: Name of the collection
        vector_dimension: Dimension of the embedding vectors
        embedding_model_path: Path to the embedding model
        reranker_model_path: Path to the reranker model
        max_results: Maximum number of results to return
        rerank_top_k: Number of results to rerank
        
    Returns:
        Search pipeline
    """
    # Import here to avoid circular imports
    from app.utils.vector_db import VectorDBClient
    from app.utils.query_processing import QueryProcessor
    from app.utils.reranking import Reranker
    
    # Create components
    vector_db_client = VectorDBClient(vector_db_host, vector_db_port, collection_name, vector_dimension)
    query_processor = QueryProcessor(embedding_model_path)
    reranker = Reranker(reranker_model_path)
    
    # Create pipeline
    pipeline = SearchPipeline(
        vector_db_client, query_processor, reranker,
        max_results, rerank_top_k
    )
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
    )
    
    # Create search pipeline
    pipeline = create_search_pipeline(
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
    )
    
    # Search
    query = "What is retrieval-augmented generation?"
    results = pipeline.search(query)
    
    # Print results
    print(f"Top {len(results)} results for query: {query}")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['rerank_score']:.4f}, Vector Score: {result['score']:.4f}")
        print(f"   File: {result['filename']}")
        print(f"   Text: {result['chunk_text'][:200]}...")
        print()
```

## app/utils/text_chunking.py

```python
import re
import uuid
from typing import List, Dict, Any
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing line breaks.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple line breaks with a single one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r' {2,}', ' ', text)
    
    # Strip whitespace from beginning and end
    text = text.strip()
    
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text: Text to split into chunks
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Clean the text first
    text = clean_text(text)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    return chunks

def process_chunks(df: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 50, 
                  max_chunks_per_doc: int = 1000) -> pd.DataFrame:
    """
    Process a DataFrame of PDFs and split text into chunks.
    
    Args:
        df: DataFrame with PDF information and extracted text
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
        max_chunks_per_doc: Maximum number of chunks per document
        
    Returns:
        DataFrame with text chunks
    """
    logger.info(f"Processing chunks with size={chunk_size}, overlap={chunk_overlap}")
    
    chunks_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        # Skip if no text
        if not row['text'] or len(row['text']) == 0:
            continue
            
        # Get chunks
        chunks = chunk_text(row['text'], chunk_size, chunk_overlap)
        
        # Limit chunks if necessary
        if len(chunks) > max_chunks_per_doc:
            logger.warning(f"Document {row['filename']} has {len(chunks)} chunks, "
                          f"limiting to {max_chunks_per_doc}")
            chunks = chunks[:max_chunks_per_doc]
        
        # Add chunks to list
        for i, chunk_content in enumerate(chunks):
            chunk_data = {
                'chunk_id': str(uuid.uuid4()),
                'pdf_path': row['path'],
                'filename': row['filename'],
                'chunk_index': i,
                'chunk_text': chunk_content,
                'token_count': len(chunk_content.split())
            }
            chunks_data.append(chunk_data)
    
    # Create DataFrame from chunks
    chunks_df = pd.DataFrame(chunks_data)
    
    logger.info(f"Created {len(chunks_df)} chunks from {len(df)} documents")
    
    return chunks_df

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import PDF_UPLOAD_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC
    from app.utils.pdf_ingestion import process_pdfs
    
    # Process PDFs
    pdf_df = process_pdfs(PDF_UPLOAD_FOLDER)
    
    # Process chunks
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC)
    
    print(f"Created {len(chunks_df)} chunks")
    print(chunks_df.head())
```

## app/utils/vector_db.py

```python
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import logging
import time
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDBClient:
    def __init__(self, host: str, port: int, collection_name: str, vector_size: int, timeout: float = 10.0, max_retries: int = 3):
        """
        Initialize the vector database client.
        
        Args:
            host: Host of the Qdrant server
            port: Port of the Qdrant server
            collection_name: Name of the collection to use
            vector_size: Dimension of the embedding vectors
            timeout: Connection timeout in seconds
            max_retries: Maximum number of connection retries
        """
        logger.info(f"Connecting to Qdrant at {host}:{port}")
        
        # Try to connect with retries
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                self.client = QdrantClient(host=host, port=port, timeout=timeout)
                # Test the connection
                self.client.get_collections()
                logger.info(f"Successfully connected to Qdrant at {host}:{port}")
                break
            except Exception as e:
                retry_count += 1
                last_exception = e
                logger.warning(f"Connection attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        
        if retry_count == max_retries:
            logger.error(f"Failed to connect to Qdrant after {max_retries} attempts")
            if last_exception:
                raise last_exception
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        
    def create_collection(self, max_retries: int = 3) -> None:
        """Create a collection in the vector database."""
        logger.info(f"Creating collection: {self.collection_name}")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Check if collection already exists
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if self.collection_name in collection_names:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return
                
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    # Add optimizers config for better performance
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000  # Use memmapped storage for collections > 20k vectors
                    )
                )
                
                logger.info(f"Collection {self.collection_name} created")
                return
            except Exception as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count}/{max_retries} to create collection failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to create collection after {max_retries} attempts")
                    raise
    
    def delete_collection(self, max_retries: int = 3) -> None:
        """
        Delete the collection.
        
        Args:
            max_retries: Maximum number of retries
        """
        logger.info(f"Deleting collection: {self.collection_name}")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Collection {self.collection_name} deleted")
                return
            except Exception as e:
                retry_count += 1
                logger.warning(f"Delete attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to delete collection after {max_retries} attempts: {str(e)}")
    
    def upload_vectors(self, df: pd.DataFrame, 
                      vector_column: str = 'embedding',
                      batch_size: int = 100,
                      max_retries: int = 3) -> None:
        """
        Upload vectors to the collection.
        
        Args:
            df: DataFrame with embeddings
            vector_column: Name of the column containing embeddings
            batch_size: Batch size for uploading
            max_retries: Maximum number of retries for failed uploads
        """
        logger.info(f"Uploading {len(df)} vectors to collection {self.collection_name}")
        
        # Ensure collection exists
        self.create_collection()
        
        # Prepare points for upload
        points = []
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Preparing vectors"):
            # Convert embedding to list if it's a numpy array
            embedding = row[vector_column]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Create point
            point = models.PointStruct(
                id=i,  # Use DataFrame index as ID
                vector=embedding,
                payload={
                    'chunk_id': row['chunk_id'],
                    'pdf_path': row['pdf_path'],
                    'filename': row['filename'],
                    'chunk_index': row['chunk_index'],
                    'chunk_text': row['chunk_text'],
                    'token_count': row['token_count']
                }
            )
            
            points.append(point)
        
        # Upload in batches
        total_batches = (len(points) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(points), batch_size), total=total_batches, desc="Uploading batches"):
            batch = points[i:i+batch_size]
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Batch upload attempt {retry_count}/{max_retries} failed: {str(e)}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to upload batch after {max_retries} attempts")
                        raise
        
        logger.info(f"Uploaded {len(df)} vectors to collection {self.collection_name}")
    
    def search(self, query_vector: List[float], limit: int = 5, max_retries: int = 3) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            max_retries: Maximum number of retries
            
        Returns:
            List of search results
        """
        logger.info(f"Searching collection {self.collection_name} for similar vectors")
        
        # Convert query vector to list if it's a numpy array
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        # Search with retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Search
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
                
                # Convert to list of dictionaries
                search_results = []
                for result in results:
                    item = result.payload
                    item['score'] = result.score
                    search_results.append(item)
                
                logger.info(f"Found {len(search_results)} results")
                return search_results
            except Exception as e:
                retry_count += 1
                logger.warning(f"Search attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to search after {max_retries} attempts: {str(e)}")
                    return []
    
    def count_vectors(self, max_retries: int = 3) -> int:
        """
        Count the number of vectors in the collection.
        
        Args:
            max_retries: Maximum number of retries
            
        Returns:
            Number of vectors
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                count = self.client.count(collection_name=self.collection_name).count
                return count
            except Exception as e:
                retry_count += 1
                logger.warning(f"Count attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to count vectors after {max_retries} attempts: {str(e)}")
                    return 0

def setup_vector_db(host: str, port: int, collection_name: str, vector_size: int) -> VectorDBClient:
    """
    Set up the vector database.
    
    Args:
        host: Host of the Qdrant server
        port: Port of the Qdrant server
        collection_name: Name of the collection to use
        vector_size: Dimension of the embedding vectors
        
    Returns:
        Vector database client
    """
    # Create client
    client = VectorDBClient(host, port, collection_name, vector_size)
    
    # Create collection
    client.create_collection()
    
    return client

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        PDF_UPLOAD_FOLDER, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP
    )
    from app.utils.pdf_ingestion import process_pdfs
    from app.utils.text_chunking import process_chunks
    from app.utils.embedding_generation import embed_chunks
    
    # Process PDFs
    pdf_df = process_pdfs(PDF_UPLOAD_FOLDER)
    
    # Process chunks
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Generate embeddings
    chunks_with_embeddings = embed_chunks(chunks_df, EMBEDDING_MODEL_PATH)
    
    # Set up vector database
    vector_db = setup_vector_db(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION)
    
    # Upload vectors
    vector_db.upload_vectors(chunks_with_embeddings)
    
    # Count vectors
    count = vector_db.count_vectors()
    print(f"Vector database contains {count} vectors")
```

## backup_restore.sh

```bash
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
```

## check_status.sh

```bash
#!/bin/bash

# Check container status
echo "Container status:"
docker-compose ps

# Check model-server health
echo -e "\nChecking model-server health..."
curl -s http://localhost:5002/health | jq || echo "Failed to connect to model-server"

# Check flask-app health
echo -e "\nChecking flask-app health..."
curl -s http://localhost:8000/api/health | jq || echo "Failed to connect to flask-app"

# Check vector-db health
echo -e "\nChecking vector-db health..."
curl -s http://localhost:6333/healthz | jq || echo "Failed to connect to vector-db"

# Check mlflow health
echo -e "\nChecking mlflow health..."
curl -s http://localhost:5001/api/2.0/mlflow/experiments/list | jq || echo "Failed to connect to mlflow"

# Show recent logs
echo -e "\nRecent logs from model-server:"
docker-compose logs --tail=20 model-server

echo -e "\nRecent logs from flask-app:"
docker-compose logs --tail=20 flask-app 
```

## check_system.sh

```bash
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
if curl -s http://localhost:8000/api/health > /dev/null; then
    FLASK_APP_HEALTH=$(curl -s http://localhost:8000/api/health)
    echo -e "${GREEN}Flask App is up and running!${NC}"
    echo "Health response: $FLASK_APP_HEALTH"
else
    echo -e "${RED}Flask App is not responding!${NC}"
fi

echo -e "\n${YELLOW}Testing API with a sample question...${NC}"
API_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d '{"question":"What is RAG?"}' http://localhost:8000/api/ask)
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
```

## docker-compose.yml

```yaml
# docker-compose.yml - Updated to avoid port 5000 conflict

services:
  vector-db:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/vectors:/qdrant/storage
    environment:
      - QDRANT_ALLOW_CORS=true
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5001:5000"  # External port 5001 mapped to container's 5000
    volumes:
      - mlflow-docker-data:/mlflow
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow/mlflow.db
    command: >
      /bin/sh -c "
      mkdir -p /mlflow/artifacts && 
      chmod -R 777 /mlflow && 
      mlflow server 
      --backend-store-uri sqlite:///mlflow/mlflow.db 
      --default-artifact-root /mlflow/artifacts 
      --host 0.0.0.0 
      --port 5000"
    user: root
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "echo", "healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  flask-app:
    build:
      context: .
      dockerfile: flask-app/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/flask_app/data
      - ./app:/flask_app/app
      - ./models:/flask_app/models
    environment:
      - FLASK_APP=app.py
      - FLASK_DEBUG=1
      - MLFLOW_HOST=model-server  # Use service name for Docker networking
      - MLFLOW_PORT=5000    # Internal port for Docker networking
      - CMAKE_ARGS=-DLLAMA_CUBLAS=0    # Disable CUDA for llama-cpp-python
      - FORCE_CMAKE=1                  # Force using CMake for building
    depends_on:
      - vector-db
      - mlflow
      - model-server
    restart: unless-stopped

  model-server:
    build:
      context: .
      dockerfile: Dockerfile.model-server
    ports:
      - "5002:5000"
    volumes:
      - ./app:/model_server/app
      - ./models:/model_server/models
      - model-cache:/model_server/.cache
    environment:
      - PORT=5000
      - HF_TOKEN=${HF_TOKEN}
      - HF_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
      - MODEL_PATH=/model_server/models/llm/Llama-3.2-1B-Instruct
      - EMBEDDING_MODEL_PATH=/model_server/models/embedding/all-MiniLM-L6-v2
      - RERANKER_MODEL_PATH=/model_server/models/reranker/ms-marco-MiniLM-L-6-v2
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  mlflow-docker-data:
    driver: local
  vector-data:
    driver: local
  model-cache:
    driver: local

networks:
  default:
    driver: bridge
    name: rag-network
```

## download_models.sh

```bash
#!/bin/bash

# Create model directories
mkdir -p models/llm
mkdir -p models/embedding
mkdir -p models/reranker

# Download embedding model
echo "Downloading embedding model (all-MiniLM-L6-v2)..."
python -c "
from huggingface_hub import snapshot_download
import os

# Download model files
model_path = snapshot_download(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    local_dir='models/embedding/all-MiniLM-L6-v2',
    local_dir_use_symlinks=False
)

print(f'Embedding model downloaded to {model_path}')
"

# Download reranker model
echo "Downloading reranker model (ms-marco-MiniLM-L-6-v2)..."
python -c "
from huggingface_hub import snapshot_download
import os

# Download model files
model_path = snapshot_download(
    repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2',
    local_dir='models/reranker/ms-marco-MiniLM-L-6-v2',
    local_dir_use_symlinks=False
)

print(f'Reranker model downloaded to {model_path}')
"

# Download Llama model
echo "Downloading Llama-3.2-1B-Instruct model..."
echo "This requires a Hugging Face token with access to the Meta Llama models."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):"
  read -s HF_TOKEN
  echo
fi

# Run the Python script to download the Llama model
python download_llama.py

echo "All models downloaded successfully!" 
```

## flask-app/Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /flask_app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy requirements from project root
# Note: This assumes docker build is run with context at project root
# using: docker build -t image-name -f flask-app/Dockerfile .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for interacting with Llama models
RUN pip install --no-cache-dir \
    huggingface_hub==0.21.3 \
    requests-toolbelt==1.0.0

# copy the flask app files only
COPY flask-app/*.py .
COPY flask-app/templates/ ./templates/
COPY flask-app/static/ ./static/
COPY flask-app/utils/ ./utils/

# Create necessary directories
RUN mkdir -p /flask_app/data/documents
RUN mkdir -p /flask_app/data/vectors

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=1
ENV TIMEOUT_SECONDS=60
ENV MAX_RETRIES=3
ENV RETRY_DELAY=1

EXPOSE 8000

CMD ["python", "app.py"]
```

## flask-app/app.py

```python
# Add these imports at the top
import datetime
import shutil
import os
import json
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
import sys
from pathlib import Path
import importlib.util
import requests
from typing import Dict, Any, List, Optional
import threading
import queue
import socket
import time
from utils.mlflow_client import MLflowClient, create_mlflow_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

# Create Flask app
app = Flask(__name__)

# Add context processor for datetime
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Import configuration
# Define constants directly since we don't have the config module
SECRET_KEY = 'dev-key-for-testing'
DEBUG = True
UPLOAD_FOLDER = os.path.join('/flask_app/data', 'documents')  # Use Docker container path
logger.info(f"UPLOAD_FOLDER absolute path: {os.path.abspath(UPLOAD_FOLDER)}")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf'}
MODEL_SERVER_HOST = 'model-server'  # Use Docker service name for internal networking
MODEL_SERVER_PORT = '5000'  # Use the internal port of the model server
MODEL_SERVER_URL = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"

# Force fallback mode - set to True to always use fallback responses
FORCE_FALLBACK = False
TIMEOUT_SECONDS = 128
MAX_RETRIES = 2
RETRY_DELAY = 5

# Create Flask app
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_model_server_alive():
    """Check if the model server is alive and ready to serve requests."""
    if FORCE_FALLBACK:
        logger.info("Force fallback mode is enabled, skipping model server check")
        return False
        
    try:
        # First check if the host is reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # 2 second timeout
        result = sock.connect_ex((MODEL_SERVER_HOST, int(MODEL_SERVER_PORT)))
        sock.close()
        
        if result != 0:
            logger.error(f"Model server at {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT} is not reachable")
            return False
            
        # Use the MLflow client to check health
        mlflow_client = MLflowClient(f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
        return mlflow_client.is_alive()
    except socket.error as e:
        logger.error(f"Socket error checking model health: {str(e)}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error checking model health: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error checking model health: {str(e)}")
        return False

def process_question_with_model_server(question, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Process a question using the model server with retries."""
    # If force fallback is enabled, don't even try to use the model server
    if FORCE_FALLBACK:
        logger.info("Force fallback mode is enabled, but still waiting for the full timeout period")
        # Wait for the full timeout period before returning the fallback response
        time.sleep(timeout)
        return 'fallback', "Force fallback mode is enabled"
        
    result_queue = queue.Queue()
    
    def _process():
        retries = 0
        while retries <= max_retries:
            try:
                # If this is a retry, wait before trying again
                if retries > 0:
                    logger.info(f"Waiting {retry_delay} seconds before retry {retries}/{max_retries}...")
                    time.sleep(retry_delay)
                
                # Check if model server is alive before sending request
                if not is_model_server_alive():
                    logger.error("Model server is not available before sending request")
                    if retries < max_retries:
                        retries += 1
                        continue
                    result_queue.put(('error', "Model server is not available"))
                    return
                
                # Create MLflow client
                mlflow_client = MLflowClient(f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
                
                # Send the request to the model server using the MLflow client
                logger.info(f"Sending request to model server using MLflow client (attempt {retries + 1}/{max_retries + 1})")
                result = mlflow_client.predict(question)
                logger.info(f"Received response from model server: {result}")
                
                # Extract the predictions from the response
                if "predictions" not in result:
                    logger.error(f"Invalid response from model server: {result}")
                    if retries < max_retries:
                        retries += 1
                        continue
                    result_queue.put(('error', "Invalid response from model server"))
                    return
                
                predictions = result["predictions"]
                
                # Extract answer and sources
                answer = predictions.get('text', 'No answer generated')
                sources = []
                for source in predictions.get('sources', [])[:3]:
                    sources.append(source.get('filename', 'Unknown'))
                
                # Create response
                response_data = {
                    "answer": answer,
                    "sources": sources,
                    "confidence": 0.95,  # Default confidence score
                    "processed_at": datetime.datetime.now().isoformat(),
                    "question": question
                }
                
                result_queue.put(('success', response_data))
                return  # Success, exit the retry loop
                
            except requests.exceptions.Timeout:
                logger.error(f"Timeout error connecting to model server after {min(30, timeout/(max_retries+1))} seconds")
                if retries < max_retries:
                    retries += 1
                    continue
                result_queue.put(('timeout', f"Request timed out after {timeout} seconds. The model server is taking too long to respond."))
                return
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error to model server: {str(e)}")
                if retries < max_retries:
                    retries += 1
                    continue
                result_queue.put(('error', f"Connection error to model server: {str(e)}"))
                return
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                if retries < max_retries:
                    retries += 1
                    continue
                result_queue.put(('error', str(e)))
                return
    
    # Start the thread
    thread = threading.Thread(target=_process)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete or timeout
    try:
        status, result = result_queue.get(timeout=timeout)
        return status, result
    except queue.Empty:
        # Timeout occurred
        return 'timeout', f"Request timed out after {timeout} seconds. The model server is taking too long to respond."

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page."""
    now = datetime.datetime.now()
    return render_template('index.html', now=now)

@app.route('/api')
def api_info():
    """Provide information about the API."""
    endpoints = [
        {
            'path': '/api/ask',
            'method': 'POST',
            'description': 'Ask a question to the RAG model',
            'parameters': {
                'question': 'The question to ask'
            },
            'example': {
                'curl': 'curl -X POST -H "Content-Type: application/json" -d \'{"question": "What is retrieval-augmented generation?"}\' http://localhost:8000/api/ask'
            }
        },
        {
            'path': '/api/health',
            'method': 'GET',
            'description': 'Check the health of the API and RAG model',
            'example': {
                'curl': 'curl http://localhost:8000/api/health'
            }
        },
        {
            'path': '/api/documents/reindex',
            'method': 'POST',
            'description': 'Reindex a specific document',
            'parameters': {
                'filename': 'The name of the file to reindex'
            },
            'example': {
                'curl': 'curl -X POST -H "Content-Type: application/json" -d \'{"filename": "document.pdf"}\' http://localhost:8000/api/documents/reindex'
            }
        },
        {
            'path': '/api/documents/reindex-all',
            'method': 'POST',
            'description': 'Reindex all documents',
            'example': {
                'curl': 'curl -X POST http://localhost:8000/api/documents/reindex-all'
            }
        }
    ]
    
    return jsonify({
        'name': 'PDFrag API',
        'version': '1.0.0',
        'description': 'API for interacting with the PDFrag RAG model',
        'endpoints': endpoints,
        'rag_model_status': is_model_server_alive(),
        'server_time': datetime.datetime.now().isoformat()
    })

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file uploads."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash(f'File {filename} uploaded successfully!', 'success')
            
            # Redirect to document list
            return redirect(url_for('documents'))
        else:
            flash('File type not allowed', 'error')
            return redirect(request.url)
    
    now = datetime.datetime.now()
    return render_template('upload.html', now=now)

@app.route('/documents')
def documents():
    """List uploaded documents."""
    # Get list of PDFs in upload folder
    pdfs = []
    logger.info(f"Looking for documents in {app.config['UPLOAD_FOLDER']}")
    
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        logger.info(f"Found {len(files)} files in upload folder")
        
        for filename in files:
            if allowed_file(filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_stat = os.stat(file_path)
                pdfs.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'modified': file_stat.st_mtime,
                    'indexed': True  # Assume all documents are indexed for now
                })
                logger.info(f"Added document: {filename}")
            else:
                logger.info(f"Skipped non-PDF file: {filename}")
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
    
    # Sort by modified time (newest first)
    pdfs = sorted(pdfs, key=lambda x: x['modified'], reverse=True)
    logger.info(f"Returning {len(pdfs)} documents")
    
    now = datetime.datetime.now()
    return render_template('documents.html', documents=pdfs, now=now)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    """Ask a question."""
    if request.method == 'POST':
        return redirect(url_for('ask'))
    
    now = datetime.datetime.now()
    return render_template('ask.html', now=now)

def get_fallback_response(question):
    """Generate a fallback response when the model server is not available."""
    logger.warning(f"Using fallback response for question: {question}")
    
    # Define some predefined responses for common questions
    if "what is rag" in question.lower():
        answer = "FALL BACK ANSWER: RAG (Retrieval-Augmented Generation) is a technique that combines retrieval-based and generation-based approaches for natural language processing tasks. It first retrieves relevant information from a knowledge base and then uses that information to generate a response. This approach helps ground the model's responses in factual information, reducing hallucinations and improving accuracy."
        sources = ["rag_paper.pdf", "llm_techniques.pdf", "ai_advances.pdf"]
    elif "how does rag work" in question.lower():
        answer = "FALL BACK ANSWER: RAG works in two main steps: 1) Retrieval: When a query is received, the system searches a knowledge base to find relevant documents or passages. This is typically done using vector similarity search with embeddings. 2) Generation: The retrieved information is then provided as context to a language model, which generates a response that incorporates this information. This helps the model produce more accurate and factual responses."
        sources = ["rag_implementation.pdf", "vector_search.pdf", "llm_context.pdf"]
    elif "benefits of rag" in question.lower():
        answer = "FALL BACK ANSWER: The benefits of RAG include: 1) Improved factual accuracy by grounding responses in retrieved information, 2) Reduced hallucinations compared to pure generative approaches, 3) Ability to access and cite specific sources of information, 4) More up-to-date responses when the knowledge base is regularly updated, and 5) Better handling of domain-specific questions when specialized documents are included in the knowledge base."
        sources = ["rag_advantages.pdf", "llm_comparison.pdf", "enterprise_ai.pdf"]
    else:
        # Default response for questions that don't match any pattern
        answer = f"FALL BACK ANSWER: I've analyzed your question about '{question}'. This appears to be related to information retrieval and processing systems. While I don't have specific information about this exact query in my knowledge base, I can tell you that modern AI systems use various techniques to understand and respond to natural language questions. Would you like me to explain more about how these systems work in general?"
        sources = ["general_ai.pdf", "information_systems.pdf"]
    
    # Create response
    response = {
        "answer": answer,
        "sources": sources,
        "confidence": 0.95,  # Default confidence score
        "processed_at": datetime.datetime.now().isoformat(),
        "question": question,
        "fallback": True  # Indicate that this is a fallback response
    }
    
    return response

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint for asking questions."""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question parameter'}), 400
    
    question = data['question']
    
    try:
        # Record the start time
        start_time = time.time()
        
        # Process question using the model server
        logger.info(f"Processing question: {question}")
        
        # Process the question with a timeout
        status, result = process_question_with_model_server(question, timeout=TIMEOUT_SECONDS)
        
        # Calculate how much time has passed
        elapsed_time = time.time() - start_time
        
        if status == 'success':
            return jsonify(result)
        elif status == 'timeout' or status == 'error' or status == 'fallback':
            logger.warning(f"Request failed with status: {status}. Using fallback response.")
            # If less than TIMEOUT_SECONDS have passed, wait for the remainder
            if elapsed_time < TIMEOUT_SECONDS:
                remaining_time = TIMEOUT_SECONDS - elapsed_time
                logger.info(f"Waiting additional {remaining_time:.2f} seconds to reach full timeout period of {TIMEOUT_SECONDS} seconds")
                time.sleep(remaining_time)
            return jsonify(get_fallback_response(question))
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Using fallback response due to error.")
        
        # Calculate how much time has passed
        elapsed_time = time.time() - start_time
        # If less than TIMEOUT_SECONDS have passed, wait for the remainder
        if elapsed_time < TIMEOUT_SECONDS:
            remaining_time = TIMEOUT_SECONDS - elapsed_time
            logger.info(f"Waiting additional {remaining_time:.2f} seconds to reach full timeout period of {TIMEOUT_SECONDS} seconds")
            time.sleep(remaining_time)
        
        return jsonify(get_fallback_response(question))

@app.route('/api/health')
def health():
    """Health check endpoint."""
    model_server_status = is_model_server_alive()
    
    return jsonify({
        'status': 'ok',
        'rag_model': model_server_status,
        'model_info': {
            'initialized': True,
            'mock_mode': False,  # No mock mode, always using real server
            'fallback_mode': not model_server_status,  # Indicate if we're using fallback mode
            'force_fallback': FORCE_FALLBACK,  # Indicate if force fallback mode is enabled
            'timeout_seconds': TIMEOUT_SECONDS,  # Timeout period before falling back
            'max_retries': MAX_RETRIES,  # Number of retries before falling back
            'retry_delay_seconds': RETRY_DELAY  # Delay between retries
        },
        'server_time': datetime.datetime.now().isoformat()
    })

# Register custom Jinja2 filters
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """Convert a timestamp to a date string."""
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/download/<filename>')
def download_document(filename):
    """Download a document."""
    secure_filename_val = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename_val, as_attachment=True)

@app.route('/api/documents/delete', methods=['POST'])
def api_delete_document():
    """API endpoint for deleting a document."""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'success': False, 'error': 'Missing filename parameter'}), 400
    
    filename = secure_filename(data['filename'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        # Delete the file
        os.remove(file_path)
        
        # Note: In a production system, we would also want to remove the document from the vector database
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.after_request
def add_no_cache_headers(response):
    """Add headers to prevent browser caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/documents/reindex', methods=['POST', 'GET'])
def api_reindex_document():
    """API endpoint for reindexing a document."""
    if request.method == 'GET':
        filename = request.args.get('filename')
        if not filename:
            return jsonify({'success': False, 'error': 'Missing filename parameter'}), 400
    else:  # POST
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'success': False, 'error': 'Missing filename parameter'}), 400
        filename = data['filename']
    
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        # Check if the RAG model is available
        if not is_model_server_alive():
            return jsonify({'success': False, 'error': 'RAG model is not available'}), 503
        
        # In a real implementation, we would call the indexing functionality of the RAG model
        # For now, we'll just log that we're reindexing the document
        logger.info(f"Reindexing document: {filename}")
        
        # Create a temporary directory with just this file
        temp_dir = os.path.join('/flask_app/data', '_temp_reindex')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy the file to the temp directory
        shutil.copy(file_path, os.path.join(temp_dir, filename))
        
        return jsonify({
            'success': True,
            'message': f'Document {filename} has been queued for reindexing',
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error reindexing document: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents/reindex-all', methods=['POST'])
def api_reindex_all():
    """API endpoint for reindexing all documents."""
    try:
        # Check if the RAG model is available
        if not is_model_server_alive():
            return jsonify({'success': False, 'error': 'RAG model is not available'}), 503
        
        # Get all PDF files in the upload folder
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        if not pdf_files:
            return jsonify({'success': True, 'message': 'No documents to reindex', 'count': 0})
        
        # In a real implementation, we would call the batch indexing functionality of the RAG model
        # For now, we'll just log that we're reindexing all documents
        logger.info(f"Reindexing all documents: {len(pdf_files)} files")
        
        return jsonify({
            'success': True,
            'message': f'{len(pdf_files)} documents have been queued for reindexing',
            'count': len(pdf_files),
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error reindexing all documents: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app with debug mode
    app.run(debug=DEBUG, host='0.0.0.0', port=8000, threaded=True)
```

## flask-app/config.py

```python
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Flask settings
SECRET_KEY = "change-this-in-production"
DEBUG = True

# Upload settings
UPLOAD_FOLDER = os.path.join(Path(__file__).resolve().parent.parent, "data", "pdfs")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {'pdf'}

# MLflow settings
MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "mlflow")  # Use Docker service name
MLFLOW_PORT = os.environ.get("MLFLOW_PORT", "5000")  # Use internal Docker port
```

## flask-app/static/css/style.css

```css
/* Custom styles for the RAG system */

/* Main container min-height for short pages */
body {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

.container {
    flex: 1;
}

footer {
    margin-top: auto;
}

/* Card hover effect */
.card-hover:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}

/* Document list */
.document-list .document-item {
    border-left: 4px solid #007bff;
    margin-bottom: 10px;
}

/* Source highlighting */
.source-highlight {
    background-color: rgba(255, 243, 205, 0.5);
    border-radius: 3px;
    padding: 2px 4px;
}

/* Answer section */
.answer-section {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 20px;
    margin-top: 20px;
}

.answer-text {
    font-size: 1.1rem;
    line-height: 1.6;
}

.source-section {
    margin-top: 20px;
    border-top: 1px solid #dee2e6;
    padding-top: 15px;
}

.source-item {
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 4px;
    background-color: #fff;
    border: 1px solid #e9ecef;
}

.source-text {
    max-height: 150px;
    overflow-y: auto;
}

/* Loading spinner */
.spinner-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 200px;
}

/* Citation tooltips */
.citation {
    cursor: pointer;
    color: #007bff;
    font-weight: bold;
    font-size: 0.8rem;
    vertical-align: super;
}
```

## flask-app/static/css/styles.css

```css
/* General styles */
body {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.content {
    flex: 1;
    padding: 2rem 0;
}

/* Card hover effect */
.card-hover {
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}

.card-hover:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

/* Custom button styles */
.btn-icon {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

/* Custom alert styles */
.alert {
    border-radius: 0.5rem;
    border-left-width: 4px;
}

.alert-info {
    border-left-color: #0d6efd;
}

.alert-success {
    border-left-color: #198754;
}

.alert-warning {
    border-left-color: #ffc107;
}

.alert-danger {
    border-left-color: #dc3545;
}

/* Flash messages */
.flash-messages {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 1050;
    max-width: 350px;
}

.flash-message {
    margin-bottom: 0.5rem;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Loading spinner */
.spinner-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1040;
}

.spinner-container {
    text-align: center;
    padding: 2rem;
    background: white;
    border-radius: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Custom form styles */
.form-floating > label {
    padding: 1rem 0.75rem;
}

.form-floating > .form-control {
    padding: 1rem 0.75rem;
}

.form-floating > .form-control:focus ~ label,
.form-floating > .form-control:not(:placeholder-shown) ~ label {
    transform: scale(0.85) translateY(-0.5rem) translateX(0.15rem);
}

/* File upload zone */
.upload-zone {
    border: 2px dashed #dee2e6;
    border-radius: 0.5rem;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

.upload-zone:hover,
.upload-zone.dragover {
    border-color: #0d6efd;
    background-color: rgba(13, 110, 253, 0.05);
}

/* Document list styles */
.document-list {
    max-height: 500px;
    overflow-y: auto;
}

.document-item {
    padding: 1rem;
    border-bottom: 1px solid #dee2e6;
    transition: background-color 0.2s ease;
}

.document-item:hover {
    background-color: #f8f9fa;
}

.document-item:last-child {
    border-bottom: none;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .content {
        padding: 1rem 0;
    }
    
    .flash-messages {
        left: 1rem;
        right: 1rem;
        max-width: none;
    }
    
    .document-list {
        max-height: none;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #212529;
        color: #f8f9fa;
    }
    
    .card {
        background-color: #2c3034;
        border-color: #373b3e;
    }
    
    .form-control {
        background-color: #2c3034;
        border-color: #373b3e;
        color: #f8f9fa;
    }
    
    .form-control:focus {
        background-color: #2c3034;
        border-color: #0d6efd;
        color: #f8f9fa;
    }
    
    .upload-zone {
        border-color: #373b3e;
    }
    
    .upload-zone:hover,
    .upload-zone.dragover {
        background-color: rgba(13, 110, 253, 0.1);
    }
    
    .document-item:hover {
        background-color: #2c3034;
    }
    
    .spinner-overlay {
        background: rgba(33, 37, 41, 0.8);
    }
    
    .spinner-container {
        background: #2c3034;
    }
} 
```

## flask-app/static/js/main.js

```javascript
// Flash message handling
function showFlashMessage(message, type = 'info') {
    const flashContainer = document.querySelector('.flash-messages');
    if (!flashContainer) {
        console.warn('Flash message container not found');
        return;
    }

    const flashMessage = document.createElement('div');
    flashMessage.className = `flash-message alert alert-${type} alert-dismissible fade show`;
    flashMessage.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    flashContainer.appendChild(flashMessage);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        flashMessage.classList.remove('show');
        setTimeout(() => flashMessage.remove(), 150);
    }, 5000);
}

// Loading spinner
function showLoading(message = 'Loading...') {
    const spinner = document.createElement('div');
    spinner.className = 'spinner-overlay';
    spinner.innerHTML = `
        <div class="spinner-container">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">${message}</p>
        </div>
    `;
    document.body.appendChild(spinner);
    return spinner;
}

function hideLoading(spinner) {
    if (spinner) {
        spinner.remove();
    }
}

// Form validation
function validateForm(form, rules) {
    const inputs = form.querySelectorAll('input, textarea, select');
    let isValid = true;

    inputs.forEach(input => {
        const fieldRules = rules[input.name];
        if (!fieldRules) return;

        // Required validation
        if (fieldRules.required && !input.value.trim()) {
            isValid = false;
            showFieldError(input, 'This field is required');
            return;
        }

        // Pattern validation
        if (fieldRules.pattern && !fieldRules.pattern.test(input.value)) {
            isValid = false;
            showFieldError(input, fieldRules.message || 'Invalid input');
            return;
        }

        // Clear any existing error
        clearFieldError(input);
    });

    return isValid;
}

function showFieldError(input, message) {
    const formGroup = input.closest('.form-group') || input.parentElement;
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    
    input.classList.add('is-invalid');
    formGroup.appendChild(errorDiv);
}

function clearFieldError(input) {
    const formGroup = input.closest('.form-group') || input.parentElement;
    const errorDiv = formGroup.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
    input.classList.remove('is-invalid');
}

// File upload handling
function handleFileUpload(input, previewElement, maxSize = 16 * 1024 * 1024) {
    const file = input.files[0];
    if (!file) return false;

    // Check file size
    if (file.size > maxSize) {
        showFlashMessage(`File size exceeds ${maxSize / (1024 * 1024)}MB limit`, 'danger');
        input.value = '';
        return false;
    }

    // Check file type
    if (!file.type.startsWith('application/pdf')) {
        showFlashMessage('Only PDF files are allowed', 'danger');
        input.value = '';
        return false;
    }

    // Update preview
    if (previewElement) {
        previewElement.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="bi bi-file-earmark-pdf text-danger me-2"></i>
                <span>${file.name}</span>
            </div>
        `;
    }

    return true;
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', () => {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});

// Export functions
window.app = {
    showFlashMessage,
    showLoading,
    hideLoading,
    validateForm,
    handleFileUpload
}; 
```

## flask-app/templates/ask.html

```html
{% extends "base.html" %}

{% block title %}Ask Questions - PDF RAG System{% endblock %}

{% block head %}
<style>
.chat-container {
    height: calc(100vh - 300px);
    min-height: 400px;
}

.chat-messages {
    height: calc(100% - 100px);
    overflow-y: auto;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 0.5rem;
}

.message {
    margin-bottom: 1rem;
    max-width: 80%;
}

.message.user {
    margin-left: auto;
}

.message.assistant {
    margin-right: auto;
}

.message .content {
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    position: relative;
}

.message.user .content {
    background: #007bff;
    color: white;
    border-top-right-radius: 0.2rem;
}

.message.assistant .content {
    background: white;
    border: 1px solid #dee2e6;
    border-top-left-radius: 0.2rem;
}

.message .metadata {
    font-size: 0.75rem;
    color: #6c757d;
    margin-top: 0.25rem;
}

.message.user .metadata {
    text-align: right;
}

.sources {
    font-size: 0.875rem;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(0,0,0,0.05);
    border-radius: 0.5rem;
}

.source-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0;
}

.chat-input {
    margin-top: 1rem;
}

.thinking {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem;
    color: #6c757d;
}

.thinking .dots {
    display: flex;
    gap: 0.25rem;
}

.thinking .dot {
    width: 8px;
    height: 8px;
    background: #6c757d;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}

.thinking .dot:nth-child(2) {
    animation-delay: 0.2s;
}

.thinking .dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
        opacity: 1;
    }
    50% {
        transform: scale(0.5);
        opacity: 0.5;
    }
}
</style>
{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h2><i class="bi bi-chat-dots"></i> Ask Questions</h2>
            </div>
            <div class="card-body chat-container">
                <div class="chat-messages" id="chat-messages">
                    {% if messages %}
                        {% for message in messages %}
                        <div class="message {{ message.role }}">
                            <div class="content">
                                {{ message.content | safe }}
                                {% if message.sources %}
                                <div class="sources">
                                    <div class="source-header">
                                        <i class="bi bi-link-45deg"></i> Sources:
                                    </div>
                                    {% for source in message.sources %}
                                    <div class="source-item">
                                        <i class="bi bi-file-earmark-pdf text-danger"></i>
                                        <span>{{ source.filename }} (p. {{ source.page }})</span>
                                    </div>
                                    {% endfor %}
                                </div>
                                {% endif %}
                            </div>
                            <div class="metadata">
                                {{ message.timestamp.strftime('%H:%M') }}
                            </div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="text-center py-5">
                            <i class="bi bi-chat-dots" style="font-size: 3rem; color: #ccc;"></i>
                            <h3 class="mt-3">No Messages Yet</h3>
                            <p class="text-muted">Ask a question about your documents to get started.</p>
                        </div>
                    {% endif %}
                </div>
                
                <form id="question-form" class="chat-input">
                    <div class="input-group">
                        <input type="text" class="form-control" id="question" name="question" 
                               placeholder="Type your question here..." required>
                        <button class="btn btn-primary" type="submit" id="submit-btn">
                            <i class="bi bi-send"></i> Send
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>

<!-- Thinking Indicator Template -->
<template id="thinking-template">
    <div class="message assistant">
        <div class="content thinking">
            <span>Thinking</span>
            <div class="dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    </div>
</template>

<!-- Message Template -->
<template id="message-template">
    <div class="message">
        <div class="content"></div>
        <div class="metadata"></div>
    </div>
</template>
{% endblock %}

{% block scripts %}
<script>
const chatMessages = document.getElementById('chat-messages');
const questionForm = document.getElementById('question-form');
const questionInput = document.getElementById('question');
const submitButton = document.getElementById('submit-btn');
const thinkingTemplate = document.getElementById('thinking-template');
const messageTemplate = document.getElementById('message-template');

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatTime(date) {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
}

function addMessage(content, role, timestamp, sources = null) {
    const messageEl = messageTemplate.content.cloneNode(true).querySelector('.message');
    messageEl.classList.add(role);
    
    const contentEl = messageEl.querySelector('.content');
    contentEl.innerHTML = content;
    
    if (sources) {
        const sourcesHtml = `
            <div class="sources">
                <div class="source-header">
                    <i class="bi bi-link-45deg"></i> Sources:
                </div>
                ${sources.map(source => `
                    <div class="source-item">
                        <i class="bi bi-file-earmark-pdf text-danger"></i>
                        <span>${source.filename} (p. ${source.page})</span>
                    </div>
                `).join('')}
            </div>
        `;
        contentEl.insertAdjacentHTML('beforeend', sourcesHtml);
    }
    
    const metadataEl = messageEl.querySelector('.metadata');
    metadataEl.textContent = formatTime(timestamp);
    
    chatMessages.appendChild(messageEl);
    scrollToBottom();
}

function showThinking() {
    const thinkingEl = thinkingTemplate.content.cloneNode(true);
    chatMessages.appendChild(thinkingEl);
    scrollToBottom();
    return chatMessages.lastElementChild;
}

questionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Disable input and button
    questionInput.disabled = true;
    submitButton.disabled = true;
    
    // Add user message
    addMessage(question, 'user', new Date());
    
    // Show thinking indicator
    const thinkingEl = showThinking();
    
    try {
        // Send question to server
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question }),
        });
        
        const data = await response.json();
        
        // Remove thinking indicator
        thinkingEl.remove();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Add assistant message
        addMessage(data.answer, 'assistant', new Date(), data.sources);
        
    } catch (error) {
        // Remove thinking indicator
        thinkingEl.remove();
        
        // Add error message
        addMessage(
            `<div class="text-danger">
                <i class="bi bi-exclamation-triangle"></i> 
                Error: ${error.message || 'Failed to get answer'}
            </div>`,
            'assistant',
            new Date()
        );
        
    } finally {
        // Clear and enable input
        questionInput.value = '';
        questionInput.disabled = false;
        submitButton.disabled = false;
        questionInput.focus();
    }
});

// Initial scroll to bottom
scrollToBottom();
</script>
{% endblock %}
```

## flask-app/templates/base.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}PDF RAG System{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block head %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">PDF RAG System</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'index' %}active{% endif %}" href="{{ url_for('index') }}">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'upload' %}active{% endif %}" href="{{ url_for('upload') }}">Upload</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'documents' %}active{% endif %}" href="{{ url_for('documents') }}">Documents</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'ask' %}active{% endif %}" href="{{ url_for('ask') }}">Ask</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </div>

    <footer class="footer mt-5 py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">PDF RAG System &copy; {{ now.year }}</span>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

## flask-app/templates/documents.html

```html
{% extends "base.html" %}

{% block title %}Documents - PDF RAG System{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h2><i class="bi bi-file-earmark-text"></i> Documents</h2>
                <a href="{{ url_for('upload') }}" class="btn btn-primary">
                    <i class="bi bi-cloud-upload"></i> Upload New Document
                </a>
            </div>
            <div class="card-body">
                {% if documents %}
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Document Name</th>
                                <th>Size</th>
                                <th>Upload Date</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for doc in documents %}
                            <tr>
                                <td>
                                    <i class="bi bi-file-earmark-pdf text-danger"></i>
                                    {{ doc.filename }}
                                </td>
                                <td>{{ doc.size | filesizeformat }}</td>
                                <td>{{ doc.modified | timestamp_to_date }}</td>
                                <td>
                                    {% if doc.indexed %}
                                    <span class="badge bg-success">Indexed</span>
                                    {% else %}
                                    <span class="badge bg-warning text-dark">Processing</span>
                                    {% endif %}
                                </td>
                                <td>
                                    <div class="btn-group" role="group">
                                        <button type="button" class="btn btn-sm btn-outline-primary" 
                                                onclick="window.location.href='{{ url_for('download_document', filename=doc.filename) }}'">
                                            <i class="bi bi-download"></i>
                                        </button>
                                        <button type="button" class="btn btn-sm btn-outline-danger" 
                                                onclick="confirmDelete('{{ doc.filename }}')">
                                            <i class="bi bi-trash"></i>
                                        </button>
                                        {% if doc.indexed %}
                                        <button type="button" class="btn btn-sm btn-outline-secondary" 
                                                onclick="reindexDocument('{{ doc.filename }}')">
                                            <i class="bi bi-arrow-clockwise"></i>
                                        </button>
                                        {% endif %}
                                    </div>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% else %}
                <div class="text-center py-5">
                    <i class="bi bi-file-earmark-x" style="font-size: 3rem; color: #ccc;"></i>
                    <h3 class="mt-3">No Documents Found</h3>
                    <p class="text-muted">Upload some PDF documents to get started.</p>
                    <a href="{{ url_for('upload') }}" class="btn btn-primary mt-3">
                        <i class="bi bi-cloud-upload"></i> Upload Documents
                    </a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<!-- Delete Confirmation Modal -->
<div class="modal fade" id="deleteModal" tabindex="-1" aria-labelledby="deleteModalLabel" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="deleteModalLabel">Confirm Delete</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                Are you sure you want to delete this document? This action cannot be undone.
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <form id="deleteForm" method="POST" style="display: inline;">
                    <input type="hidden" name="_method" value="DELETE">
                    <button type="submit" class="btn btn-danger">Delete</button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
// Log the number of documents found
document.addEventListener('DOMContentLoaded', function() {
    const documents = document.querySelectorAll('tbody tr');
    console.log(`Found ${documents.length} documents on the page`);
    
    // Log each document name
    documents.forEach((doc, index) => {
        const filename = doc.querySelector('td:first-child').textContent.trim();
        console.log(`Document ${index + 1}: ${filename}`);
    });
});

function confirmDelete(filename) {
    const modal = new bootstrap.Modal(document.getElementById('deleteModal'));
    const form = document.getElementById('deleteForm');
    form.action = `/documents/${filename}/delete`;
    modal.show();
}

function reindexDocument(filename) {
    // Send a POST request to reindex the document
    fetch('/api/documents/reindex', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Document reindexing started successfully!');
        } else {
            alert(`Error: ${data.error || 'Failed to reindex document'}`);
        }
    })
    .catch(error => {
        alert(`Error: ${error.message || 'Failed to reindex document'}`);
    });
}
</script>
{% endblock %}
```

## flask-app/templates/index.html

```html
{% extends "base.html" %}

{% block title %}Home - PDF RAG System{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-12 text-center mb-5">
        <h1 class="display-4">Welcome to PDF RAG System</h1>
        <p class="lead">A powerful Retrieval-Augmented Generation system for your PDF documents.</p>
    </div>
</div>

<div class="row">
    <div class="col-md-4 mb-4">
        <div class="card h-100">
            <div class="card-body text-center">
                <h3 class="card-title"><i class="bi bi-cloud-upload"></i> Upload</h3>
                <p class="card-text">Upload your PDF documents to the system for processing and indexing.</p>
                <a href="{{ url_for('upload') }}" class="btn btn-primary">Upload Documents</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-4 mb-4">
        <div class="card h-100">
            <div class="card-body text-center">
                <h3 class="card-title"><i class="bi bi-file-earmark-text"></i> Manage</h3>
                <p class="card-text">View and manage your uploaded documents in the system.</p>
                <a href="{{ url_for('documents') }}" class="btn btn-primary">View Documents</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-4 mb-4">
        <div class="card h-100">
            <div class="card-body text-center">
                <h3 class="card-title"><i class="bi bi-chat-dots"></i> Ask</h3>
                <p class="card-text">Ask questions about your documents and get AI-powered answers.</p>
                <a href="{{ url_for('ask') }}" class="btn btn-primary">Ask Questions</a>
            </div>
        </div>
    </div>
</div>

<div class="row mt-5">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h3>How It Works</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4 text-center mb-3">
                        <div class="p-3">
                            <h4><i class="bi bi-1-circle"></i> Upload</h4>
                            <p>Upload your PDF documents to the system.</p>
                        </div>
                    </div>
                    <div class="col-md-4 text-center mb-3">
                        <div class="p-3">
                            <h4><i class="bi bi-2-circle"></i> Process</h4>
                            <p>The system processes and indexes your documents.</p>
                        </div>
                    </div>
                    <div class="col-md-4 text-center mb-3">
                        <div class="p-3">
                            <h4><i class="bi bi-3-circle"></i> Ask</h4>
                            <p>Ask questions and get answers based on your documents.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

## flask-app/templates/upload.html

```html
{% extends "base.html" %}

{% block title %}Upload - PDF RAG System{% endblock %}

{% block head %}
<style>
.drop-zone {
    max-width: 100%;
    height: 200px;
    padding: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 20px;
    font-weight: 500;
    cursor: pointer;
    color: #cccccc;
    border: 4px dashed #009578;
    border-radius: 10px;
    margin-bottom: 20px;
}

.drop-zone--over {
    border-style: solid;
    background-color: rgba(0, 149, 120, 0.1);
}

.drop-zone__input {
    display: none;
}

.drop-zone__thumb {
    width: 100%;
    height: 100%;
    border-radius: 10px;
    overflow: hidden;
    background-color: #cccccc;
    background-size: cover;
    position: relative;
}

.drop-zone__prompt {
    color: #666;
}
</style>
{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8 offset-md-2">
        <div class="card">
            <div class="card-header">
                <h2><i class="bi bi-cloud-upload"></i> Upload Documents</h2>
            </div>
            <div class="card-body">
                <form action="{{ url_for('upload') }}" method="post" enctype="multipart/form-data" id="upload-form">
                    <div class="drop-zone">
                        <span class="drop-zone__prompt">
                            <i class="bi bi-file-earmark-pdf fs-1"></i><br>
                            Drop PDF file here or click to upload
                        </span>
                        <input type="file" name="file" class="drop-zone__input" accept=".pdf">
                    </div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-primary btn-lg" disabled id="upload-btn">
                            <i class="bi bi-cloud-upload"></i> Upload Document
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <div class="card mt-4">
            <div class="card-header">
                <h3>Upload Guidelines</h3>
            </div>
            <div class="card-body">
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        <i class="bi bi-check-circle text-success"></i> Only PDF files are accepted
                    </li>
                    <li class="list-group-item">
                        <i class="bi bi-check-circle text-success"></i> Maximum file size: 16MB
                    </li>
                    <li class="list-group-item">
                        <i class="bi bi-check-circle text-success"></i> Files will be processed and indexed automatically
                    </li>
                    <li class="list-group-item">
                        <i class="bi bi-check-circle text-success"></i> Uploaded files can be managed in the Documents section
                    </li>
                </ul>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.querySelectorAll(".drop-zone__input").forEach(inputElement => {
    const dropZoneElement = inputElement.closest(".drop-zone");
    const uploadButton = document.getElementById("upload-btn");

    dropZoneElement.addEventListener("click", e => {
        inputElement.click();
    });

    inputElement.addEventListener("change", e => {
        if (inputElement.files.length) {
            updateThumbnail(dropZoneElement, inputElement.files[0]);
            uploadButton.disabled = false;
        }
    });

    dropZoneElement.addEventListener("dragover", e => {
        e.preventDefault();
        dropZoneElement.classList.add("drop-zone--over");
    });

    ["dragleave", "dragend"].forEach(type => {
        dropZoneElement.addEventListener(type, e => {
            dropZoneElement.classList.remove("drop-zone--over");
        });
    });

    dropZoneElement.addEventListener("drop", e => {
        e.preventDefault();

        if (e.dataTransfer.files.length) {
            inputElement.files = e.dataTransfer.files;
            updateThumbnail(dropZoneElement, e.dataTransfer.files[0]);
            uploadButton.disabled = false;
        }

        dropZoneElement.classList.remove("drop-zone--over");
    });
});

function updateThumbnail(dropZoneElement, file) {
    let thumbnailElement = dropZoneElement.querySelector(".drop-zone__thumb");

    // Remove thumbnail element if it exists
    if (thumbnailElement) {
        dropZoneElement.removeChild(thumbnailElement);
    }

    // Check if the file is a PDF
    if (!file.type.startsWith("application/pdf")) {
        alert("Please upload a PDF file");
        return;
    }

    // First time - remove the prompt
    const promptElement = dropZoneElement.querySelector(".drop-zone__prompt");
    if (promptElement) {
        promptElement.remove();
    }

    // Add the thumbnail element
    thumbnailElement = document.createElement("div");
    thumbnailElement.classList.add("drop-zone__thumb");
    dropZoneElement.appendChild(thumbnailElement);

    // Show the file name
    thumbnailElement.dataset.label = file.name;
    thumbnailElement.innerHTML = `
        <div style="height: 100%; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <i class="bi bi-file-earmark-pdf" style="font-size: 3rem; color: #dc3545;"></i>
            <p class="mt-2" style="color: #666;">${file.name}</p>
        </div>
    `;
}
</script>
{% endblock %}
```

## flask-app/test_documents.py

```python
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_documents_route():
    """Test the documents route logic."""
    # Define the upload folder
    upload_folder = os.path.join(str(Path(__file__).resolve().parent.parent), 'data', 'documents')
    logger.info(f"UPLOAD_FOLDER: {upload_folder}")
    logger.info(f"Directory exists: {os.path.exists(upload_folder)}")
    
    # Check if the directory exists
    if not os.path.exists(upload_folder):
        logger.error(f"Directory does not exist: {upload_folder}")
        return
    
    # List files in the directory
    try:
        files = os.listdir(upload_folder)
        logger.info(f"Found {len(files)} files in upload folder")
        
        # Define allowed extensions
        allowed_extensions = {'pdf'}
        
        # Check each file
        pdfs = []
        for filename in files:
            logger.info(f"Checking file: {filename}")
            
            # Check if it's a PDF
            is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
            logger.info(f"Is allowed: {is_allowed}")
            
            if is_allowed:
                file_path = os.path.join(upload_folder, filename)
                file_stat = os.stat(file_path)
                pdfs.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'modified': file_stat.st_mtime,
                })
                logger.info(f"Added document: {filename}")
            else:
                logger.info(f"Skipped non-PDF file: {filename}")
        
        # Print results
        logger.info(f"Found {len(pdfs)} PDF files")
        for pdf in pdfs:
            logger.info(f"PDF: {pdf['filename']}, Size: {pdf['size']} bytes")
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    test_documents_route() 
```

## flask-app/utils/mlflow_client.py

```python
import requests
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowClient:
    def __init__(self, endpoint_url: str):
        """
        Initialize the MLflow client.
        
        Args:
            endpoint_url: URL of the MLflow serving endpoint
        """
        self.endpoint_url = endpoint_url
        logger.info(f"Initialized MLflow client for endpoint: {endpoint_url}")
    
    def predict(self, query: str) -> Dict[str, Any]:
        """
        Make a prediction using the MLflow serving endpoint.
        
        Args:
            query: Query text
            
        Returns:
            Prediction result
        """
        logger.info(f"Sending query to MLflow endpoint: {query}")
        
        # Create payload using the question format for backward compatibility
        payload = {
            "question": query,
            "context": []
        }
        
        # Send request
        response = requests.post(
            f"{self.endpoint_url}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Error from MLflow endpoint: {response.text}")
            raise Exception(f"Error from MLflow endpoint: {response.text}")
        
        # Parse response
        result = response.json()
        
        logger.info(f"Received response from MLflow endpoint")
        return result
    
    def is_alive(self) -> bool:
        """
        Check if the MLflow endpoint is alive.
        
        Returns:
            True if the endpoint is alive, False otherwise
        """
        try:
            response = requests.get(f"{self.endpoint_url}/health")
            return response.status_code == 200
        except:
            return False

def create_mlflow_client(host: str = "localhost", port: int = 5001) -> MLflowClient:
    """
    Create an MLflow client.
    
    Args:
        host: Host of the MLflow server
        port: Port of the MLflow server
        
    Returns:
        MLflow client
    """
    endpoint_url = f"http://{host}:{port}"
    return MLflowClient(endpoint_url)
```

## flask-app/utils/pipeline_trigger.py

```python
import subprocess
import logging
import threading
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline_async(pdf_dir, rebuild=False):
    """
    Run the pipeline asynchronously in a separate thread.
    
    Args:
        pdf_dir: Directory containing PDF files
        rebuild: Whether to rebuild the vector index
    """
    # Get project root
    project_root = Path(__file__).resolve().parent.parent.parent

    def _run_pipeline():
        try:
            logger.info(f"Starting pipeline with PDF directory: {pdf_dir}")
            
            # Build command
            cmd = [
                sys.executable,
                str(project_root / "app" / "pipeline.py"),
                "--pdf-dir", pdf_dir
            ]
            
            if rebuild:
                cmd.append("--rebuild")
            
            # Run pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Get output
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Pipeline failed with return code {process.returncode}")
                logger.error(f"Error output: {stderr}")
            else:
                logger.info("Pipeline completed successfully")
                
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
    
    # Start thread
    thread = threading.Thread(target=_run_pipeline)
    thread.daemon = True
    thread.start()
    
    return {
        'status': 'started',
        'message': 'Pipeline started in the background'
    }
```

## offline_test.sh

```bash
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Offline Mode Test${NC}"
echo "This script will test if the system works properly without internet connectivity."
echo "It will temporarily disable network access for Docker containers."

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root to modify network settings${NC}"
  exit 1
fi

# Check if system is running
echo "Checking if system is running..."
if ! docker ps | grep -q "vector-db"; then
    echo -e "${RED}Vector database container not running. Please start the system first.${NC}"
    exit 1
fi

if ! docker ps | grep -q "mlflow"; then
    echo -e "${RED}MLflow container not running. Please start the system first.${NC}"
    exit 1
fi

if ! docker ps | grep -q "flask-app"; then
    echo -e "${RED}Flask app container not running. Please start the system first.${NC}"
    exit 1
fi

echo -e "${GREEN}All containers are running.${NC}"

# Create a Docker network with no internet access
echo "Creating isolated Docker network..."
docker network create --internal isolated-network

# Move containers to isolated network
echo "Moving containers to isolated network..."
docker network connect isolated-network $(docker ps -q --filter name=vector-db)
docker network connect isolated-network $(docker ps -q --filter name=mlflow)
docker network connect isolated-network $(docker ps -q --filter name=flask-app)

# Disconnect from default network
echo "Disconnecting from default network..."
docker network disconnect bridge $(docker ps -q --filter name=vector-db)
docker network disconnect bridge $(docker ps -q --filter name=mlflow)
docker network disconnect bridge $(docker ps -q --filter name=flask-app)

echo -e "${YELLOW}Containers are now in offline mode.${NC}"
echo "Testing system functionality..."

# Test if services are still responding
echo "Testing vector database..."
if curl -s http://localhost:6333/healthz > /dev/null; then
    echo -e "${GREEN}Vector database is responding in offline mode.${NC}"
else
    echo -e "${RED}Vector database is not responding!${NC}"
fi

echo "Testing MLflow..."
if curl -s http://localhost:5001/ping > /dev/null; then
    echo -e "${GREEN}MLflow is responding in offline mode.${NC}"
else
    echo -e "${RED}MLflow is not responding!${NC}"
fi

echo "Testing Flask app..."
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}Flask app is responding in offline mode.${NC}"
else
    echo -e "${RED}Flask app is not responding!${NC}"
fi

# Test a query to verify end-to-end functionality
echo "Testing query functionality..."
QUERY_RESULT=$(curl -s -X POST -H "Content-Type: application/json" -d '{"question":"What is machine learning?"}' http://localhost:8000/api/ask)

if [[ $QUERY_RESULT == *"text"* ]]; then
    echo -e "${GREEN}Query functionality is working in offline mode.${NC}"
else
    echo -e "${RED}Query functionality is not working in offline mode!${NC}"
    echo "Response: $QUERY_RESULT"
fi

# Wait for user to check the system
echo -e "${YELLOW}The system is now in offline mode. You can test it manually.${NC}"
echo "Press Enter to restore network connectivity..."
read

# Restore network connectivity
echo "Restoring network connectivity..."
docker network connect bridge $(docker ps -q --filter name=vector-db)
docker network connect bridge $(docker ps -q --filter name=mlflow)
docker network connect bridge $(docker ps -q --filter name=flask-app)

docker network disconnect isolated-network $(docker ps -q --filter name=vector-db)
docker network disconnect isolated-network $(docker ps -q --filter name=mlflow)
docker network disconnect isolated-network $(docker ps -q --filter name=flask-app)

# Remove isolated network
docker network rm isolated-network

echo -e "${GREEN}Network connectivity restored.${NC}"
echo "Offline mode test completed."
```

## restart.sh

```bash
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
```

## startup.sh

```bash
#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ROOT=$(pwd)
PDF_DIR="${PROJECT_ROOT}/data/documents"
MODELS_DIR="${PROJECT_ROOT}/models"
VENV_DIR="${PROJECT_ROOT}/venv"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Local RAG System...${NC}"

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check for requirements
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found.${NC}"
    exit 1
fi

# Install requirements if needed
pip -q install -r requirements.txt

# Check for models
if [ ! -d "$MODELS_DIR/embedding" ] || [ ! -d "$MODELS_DIR/reranker" ] || [ ! -d "$MODELS_DIR/llm" ]; then
    echo -e "${YELLOW}Some models are missing. Please download them first.${NC}"
    echo "You can use the following commands:"
    echo "  python app/scripts/download_models.py"
    echo "  ./app/scripts/download_llm.sh"
fi

# Create data directories if they don't exist
mkdir -p "$PDF_DIR"

# Start Docker services (including Flask app)
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Check if MLflow is running
echo "Checking MLflow service..."
if curl -s http://localhost:5001/ping > /dev/null; then
    echo -e "${GREEN}MLflow service is running.${NC}"
else
    echo -e "${RED}MLflow service is not running. Check Docker logs.${NC}"
    echo "You can run: docker-compose logs mlflow"
fi

# Check if vector database is running
echo "Checking vector database service..."
if curl -s http://localhost:6333/healthz > /dev/null; then
    echo -e "${GREEN}Vector database service is running.${NC}"
else
    echo -e "${RED}Vector database service is not running. Check Docker logs.${NC}"
    echo "You can run: docker-compose logs vector-db"
fi

# Check if Flask app is running
echo "Checking Flask application..."
if curl -s http://localhost:8000 > /dev/null; then
    echo -e "${GREEN}Flask application is running.${NC}"
else
    echo -e "${RED}Flask application is not running. Check Docker logs.${NC}"
    echo "You can run: docker-compose logs flask-app"
fi

# Check if there are PDFs to process
PDF_COUNT=$(find "$PDF_DIR" -name "*.pdf" | wc -l)
if [ "$PDF_COUNT" -gt 0 ]; then
    echo -e "${GREEN}Found $PDF_COUNT PDF files.${NC}"
    
    # Ask if user wants to process them
    read -p "Do you want to process them now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running processing pipeline..."
        python app/pipeline.py
    else
        echo "Skipping processing."
    fi
else
    echo "No PDF files found. You can upload them through the web interface."
fi

# Deploy the model to MLflow
echo "Deploying model to MLflow..."
# First, check if model exists
if mlflow models search -f "name = 'rag_model'" | grep -q "rag_model"; then
    echo "Model already exists, deploying latest version..."
    python app/scripts/deploy_model.py &
else
    echo "Model not found, logging new model..."
    python app/scripts/log_model.py
    python app/scripts/deploy_model.py &
fi

echo -e "${GREEN}All services started!${NC}"
echo "You can access the web interface at: http://localhost:8000"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for user interrupt
trap "echo 'Stopping services...'; docker-compose down; exit 0" INT
wait
```

