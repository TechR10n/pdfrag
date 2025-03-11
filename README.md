# PDFrag - PDF Retrieval Augmented Generation

PDFrag is a system for retrieving information from PDF documents using a Retrieval Augmented Generation (RAG) approach with the Llama-3.2-1B-Instruct model.

## Prerequisites

- Docker and Docker Compose
- Hugging Face account with access to Meta Llama models
- Hugging Face token with read access

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag
   ```

2. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file and add your Hugging Face token:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

4. Download the required models:
   ```bash
   ./download_models.sh
   ```
   This will download:
   - Llama-3.2-1B-Instruct model
   - all-MiniLM-L6-v2 embedding model
   - ms-marco-MiniLM-L-6-v2 reranker model

5. Build and start the Docker containers:
   ```bash
   docker-compose up -d --build
   ```

## Usage

1. Access the web interface at http://localhost:8000

2. Upload PDF documents through the web interface

3. Ask questions about the uploaded documents

## API

The system provides a REST API for programmatic access:

- `POST /api/ask` - Ask a question about the uploaded documents
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"question":"What is retrieval-augmented generation?"}' http://localhost:8000/api/ask
  ```

- `GET /api/health` - Check the health of the system
  ```bash
  curl http://localhost:8000/api/health
  ```

## Troubleshooting

If you encounter issues with the model server:

1. Check the logs:
   ```bash
   docker-compose logs model-server
   ```

2. Ensure your Hugging Face token has access to the Meta Llama models

3. Increase the timeout settings in the `.env` file if needed:
   ```
   TIMEOUT_SECONDS=120
   MAX_RETRIES=5
   ```

4. Restart the containers:
   ```bash
   docker-compose restart
   ```

## Architecture

The system consists of the following components:

- **flask-app**: Web interface and API
- **model-server**: Serves the Llama-3.2-1B-Instruct model
- **vector-db**: Qdrant vector database for storing document embeddings
- **mlflow**: MLflow for model tracking and serving

## License

This project is licensed under the MIT License - see the LICENSE file for details. 