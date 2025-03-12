# PDFrag - PDF Retrieval Augmented Generation

PDFrag is a system for retrieving information from PDF documents using a Retrieval Augmented Generation (RAG) approach with the Llama-3.2-1B-Instruct model.

## Prerequisites

- Docker and Docker Compose
- Hugging Face account with access to Meta Llama models
- Hugging Face token with read access

## Setup

### For Ubuntu Users

Run the setup script which will install all dependencies, set up the environment, and initialize the vector database:

```bash
./setup_ubuntu.sh
```

You can view all available options with:

```bash
./setup_ubuntu.sh --help
```

### For Other Operating Systems

1. Install Docker and Docker Compose
2. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```
3. Edit the `.env` file and add your Hugging Face token:
   ```bash
   HUGGINGFACE_TOKEN=your_token_here
   ```
4. Create directories for documents and models:
   ```bash
   mkdir -p data/documents data/vectors models
   chmod 777 data/documents data/vectors models
   ```
5. Download the required models:
   ```bash
   python -m app.download_models
   ```
6. Build and start the Docker containers:
   ```bash
   docker-compose up -d
   ```

## Starting the Application

Use the startup script to start the application:

```bash
./startup.sh
```

You can view all available options with:

```bash
./startup.sh --help
```

### Vector Database Management

The startup script provides options for managing the vector database:

- To reset the vector database (removes all indexed documents):
  ```bash
  ./startup.sh --reset-vector-db
  ```

- To rebuild the vector index:
  ```bash
  ./startup.sh --rebuild-index
  ```

- To perform a complete reset and rebuild:
  ```bash
  ./startup.sh --reset-vector-db --rebuild-index
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

### Advanced Diagnostics

For more detailed diagnostics of vector database connection issues, you can use the included diagnostic script:

```bash
python check_vector_db.py
```

This script performs comprehensive checks including:
- DNS resolution testing
- Port accessibility verification
- Connection testing
- Docker container status checks

It provides detailed output and suggestions for fixing any detected issues.

## Architecture

The system consists of the following components:

- **flask-app**: Web interface and API
- **model-server**: Serves the Llama-3.2-1B-Instruct model
- **vector-db**: Qdrant vector database for storing document embeddings
- **mlflow**: MLflow for model tracking and serving

## License

This project is licensed under the MIT License - see the LICENSE file for details. 