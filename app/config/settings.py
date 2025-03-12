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
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
HF_EMBEDDING_MODEL_ID = os.getenv("HF_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
HF_RERANKER_MODEL_ID = os.getenv("HF_RERANKER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Extract the model names from the model IDs (part after the last slash)
MODEL_NAME = HF_MODEL_ID.split('/')[-1]
EMBEDDING_MODEL_NAME = HF_EMBEDDING_MODEL_ID.split('/')[-1]
RERANKER_MODEL_NAME = HF_RERANKER_MODEL_ID.split('/')[-1]

# Model paths - all derived from environment variables
MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", MODEL_NAME)
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "embedding", EMBEDDING_MODEL_NAME)
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "reranker", RERANKER_MODEL_NAME)

# Alternative model locations
HOME_DIR = os.path.expanduser("~")
ALT_MODEL_PATHS = [
    os.path.join(HOME_DIR, ".models", MODEL_NAME),
    os.path.join(HOME_DIR, ".cache", "huggingface", "hub", f"models--{HF_MODEL_ID.replace('/', '--')}"),
    os.path.join(BASE_DIR, "models", "llm", f"{MODEL_NAME.lower()}-q4.gguf"),
    os.path.join(HOME_DIR, ".llama", f"{MODEL_NAME.lower()}-q4.gguf"),
]

# Alternative embedding model locations
ALT_EMBEDDING_MODEL_PATHS = [
    os.path.join(HOME_DIR, ".cache", "huggingface", "hub", f"models--{HF_EMBEDDING_MODEL_ID.replace('/', '--')}"),
    os.path.join(HOME_DIR, ".models", EMBEDDING_MODEL_NAME),
]

# Alternative reranker model locations
ALT_RERANKER_MODEL_PATHS = [
    os.path.join(HOME_DIR, ".cache", "huggingface", "hub", f"models--{HF_RERANKER_MODEL_ID.replace('/', '--')}"),
    os.path.join(HOME_DIR, ".models", RERANKER_MODEL_NAME),
]

# MLflow settings
MLFLOW_TRACKING_URI = f"file://{os.path.join(BASE_DIR, 'mlruns')}"  # File-based tracking URI
MLFLOW_MODEL_NAME = "rag_model"
MLFLOW_ARTIFACT_ROOT = os.path.join(BASE_DIR, "mlruns", "artifacts")  # Absolute path for artifacts

# Flask settings
FLASK_SECRET_KEY = "change-this-in-production"
PDF_UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "documents")
ALLOWED_EXTENSIONS = {'pdf'}
