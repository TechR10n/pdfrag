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