FROM python:3.10-slim

WORKDIR /flask_app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy requirements from project root
# Note: This assumes docker build is run with context at project root
# using: docker build -t image-name -f flask-app/Dockerfile .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the flask app files only
COPY flask-app/*.py .
COPY flask-app/templates/ ./templates/
COPY flask-app/static/ ./static/

EXPOSE 8000

CMD ["python", "app.py"]