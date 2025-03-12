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

2. Run the setup script (for Ubuntu):
   ```bash
   ./setup_ubuntu.sh
   ```
   
   This script will:
   - Install all required dependencies (Docker, Docker Compose, etc.)
   - Create necessary directories
   - Set up the environment file
   - Download required models
   - Initialize the vector database
   - Build and start all containers
   - Optionally index sample documents

   For other operating systems, follow these manual steps:
   
   a. Create a `.env` file from the example:
      ```bash
      cp .env.example .env
      ```

   b. Edit the `.env` file and add your Hugging Face token:
      ```
      HF_TOKEN=your_huggingface_token_here
      ```

   c. Download the required models:
      ```bash
      ./download_models.sh
      ```

   d. Build and start the Docker containers:
      ```bash
      docker-compose up -d --build
      ```

## Starting the Application

Use the startup script to start the application:

```bash
./startup.sh
```

This script will:
- Start all Docker containers
- Ensure the vector database directory exists
- Wait for services to be healthy
- Provide access to the web interface

### Vector Database Management

The startup script includes options for vector database management:

1. **Reset the vector database** (clears all indexed documents):
   ```bash
   ./startup.sh --reset-vector-db
   ```

2. **Reset and rebuild the index**:
   ```bash
   ./startup.sh --reset-vector-db --rebuild-index
   ```

3. **Just rebuild the index** (without resetting):
   ```bash
   ./startup.sh --rebuild-index
   ```

## Usage

1. Access the web interface at http://localhost:8001

2. Upload PDF documents through the web interface

3. Ask questions about the uploaded documents

## API

The system provides a REST API for programmatic access:

- `POST /api/ask` - Ask a question about the uploaded documents
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"question":"What is retrieval-augmented generation?"}' http://localhost:8001/api/ask
  ```

- `GET /api/health` - Check the health of the system
  ```bash
  curl http://localhost:8001/api/health
  ```

## Testing

The project includes a comprehensive test suite using pytest. To run the tests:

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run all tests using the test runner script:
   ```bash
   ./run_tests.py
   ```

3. Run tests with coverage report:
   ```bash
   ./run_tests.py --coverage
   ```

4. Run specific test files:
   ```bash
   ./run_tests.py --test-file app/tests/test_pdf_ingestion.py
   ```

For more information about testing, see the [Testing Guide](app/tests/README.md).

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

### Common Vector Database Issues

1. **"Temporary failure in name resolution" error**:
   - If running outside Docker, ensure `VECTOR_DB_HOST=localhost` in your `.env` file
   - If running inside Docker, ensure `VECTOR_DB_HOST=vector-db` in your `.env` file

2. **"Connection refused" error**:
   - Ensure the vector-db container is running: `docker-compose ps`
   - Check the vector-db logs: `docker-compose logs vector-db`
   - Try resetting the vector database: `./startup.sh --reset-vector-db`

3. **Corrupted vector database**:
   - If you see errors about missing shards or corrupted data in the logs, use `./startup.sh --reset-vector-db`

## Architecture

The system consists of the following components:

- **flask-app**: Web interface and API
- **model-server**: Serves the Llama-3.2-1B-Instruct model
- **vector-db**: Qdrant vector database for storing document embeddings
- **mlflow**: MLflow for model tracking and serving

## License

This project is licensed under the MIT License - see the LICENSE file for details. 