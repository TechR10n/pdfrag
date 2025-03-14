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