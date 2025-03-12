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
