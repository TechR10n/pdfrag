import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Vector database settings
VECTOR_DB_HOST = "localhost"
VECTOR_DB_PORT = 6333
VECTOR_DIMENSION = 384  # For all-MiniLM-L6-v2
COLLECTION_NAME = "pdf_chunks"

# Document processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_DOC = 1000

# Model paths
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "embedding", "all-MiniLM-L6-v2")
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "reranker", "ms-marco-MiniLM-L-6-v2")
# Update to use the actual GGUF file instead of the directory
LLM_MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "llama-3.2-3b-instruct-q4.gguf")

# For the larger model (better quality, but slower)
# LLM_MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "Llama-3.2-3B-Instruct")

# MLflow settings
MLFLOW_TRACKING_URI = f"file://{os.path.join(BASE_DIR, 'mlruns')}"  # File-based tracking URI
MLFLOW_MODEL_NAME = "rag_model"
MLFLOW_ARTIFACT_ROOT = os.path.join(BASE_DIR, "mlruns", "artifacts")  # Absolute path for artifacts

# Flask settings
FLASK_SECRET_KEY = "change-this-in-production"
PDF_UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")
ALLOWED_EXTENSIONS = {'pdf'}
