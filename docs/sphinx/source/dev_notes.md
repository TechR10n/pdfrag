# Developer Notes for PDF RAG System

## Overview

This document contains development notes, technical details, and implementation guidelines for the PDF RAG (Retrieval Augmented Generation) System. It serves as an appendix to the main documentation and is intended for developers working on the project.

## Architecture Details

### System Components

The PDF RAG System consists of the following main components:

1. **Web Interface (Flask)**
   - Handles user interactions
   - Manages file uploads
   - Displays query results
   - Provides administrative functions

2. **Document Processing Pipeline**
   - PDF parsing and text extraction
   - Chunking and preprocessing
   - Vector embedding generation
   - Storage in vector database

3. **Query Processing Engine**
   - Query understanding and preprocessing
   - Vector similarity search
   - Context assembly
   - Response generation with LLM

4. **Experiment Tracking (MLflow)**
   - Model versioning
   - Experiment logging
   - Performance metrics
   - Model registry

### Data Flow

1. **Document Ingestion Flow**
   ```
   Upload PDF → Extract Text → Chunk Text → Generate Embeddings → Store in Vector DB
   ```

2. **Query Processing Flow**
   ```
   User Query → Preprocess Query → Generate Query Embedding → Similarity Search → 
   Retrieve Relevant Chunks → Assemble Context → Generate Response → Display to User
   ```

## Implementation Notes

### Vector Database Selection

We evaluated several vector databases for this project:

| Database | Pros | Cons | Decision |
|----------|------|------|----------|
| Chroma | Easy integration, Python-native | Limited scaling | Selected for development |
| Pinecone | Excellent scaling, managed service | Cost, external dependency | Consider for production |
| Weaviate | Rich features, good performance | More complex setup | Alternative option |

### Embedding Models

The system supports multiple embedding models:

- OpenAI's text-embedding-ada-002
- Sentence Transformers (all-MiniLM-L6-v2)
- BERT-based models

Current default: `all-MiniLM-L6-v2` for local deployment

### LLM Integration

The system is designed to work with various LLMs:

- OpenAI models (GPT-3.5, GPT-4)
- Local models via llama.cpp
- Hugging Face models

## Development Guidelines

### Code Organization

```
app/
├── api/                  # API endpoints
│   ├── __init__.py
│   ├── documents.py      # Document management endpoints
│   └── queries.py        # Query processing endpoints
├── models/               # Core functionality
│   ├── __init__.py
│   ├── document.py       # Document processing
│   ├── embeddings.py     # Embedding generation
│   ├── retrieval.py      # Vector retrieval
│   └── llm.py            # LLM integration
├── scripts/              # Utility scripts
│   ├── convert_md_to_pdf.py
│   └── setup_sphinx.py
├── static/               # Static assets
│   ├── css/
│   ├── js/
│   └── images/
└── templates/            # HTML templates
    ├── base.html
    ├── index.html
    └── results.html
```

### Coding Standards

1. **Python Style**
   - Follow PEP 8 guidelines
   - Use type hints for function parameters and return values
   - Write docstrings for all functions, classes, and modules

2. **Testing**
   - Write unit tests for all core functionality
   - Aim for at least 80% code coverage
   - Include integration tests for critical paths

3. **Documentation**
   - Document all public APIs
   - Keep this dev_notes.md file updated with architectural decisions
   - Use inline comments for complex logic

## Deployment

### Local Development Environment

1. **Setup**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag

   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Run the development server
   flask run --debug
   ```

2. **Environment Variables**
   Create a `.env` file with the following variables:
   ```
   FLASK_APP=app
   FLASK_ENV=development
   VECTOR_DB_PATH=./data/vectordb
   MODEL_CACHE_DIR=./data/models
   ```

### Production Deployment

1. **Docker Deployment**
   ```bash
   # Build the Docker image
   docker build -t pdfrag:latest .

   # Run the container
   docker run -p 5000:5000 -v ./data:/app/data pdfrag:latest
   ```

2. **Cloud Deployment Options**
   - AWS Elastic Beanstalk
   - Google Cloud Run
   - Azure App Service

## Performance Optimization

### Memory Usage

- Use streaming responses for large documents
- Implement pagination for vector search results
- Consider chunking strategy trade-offs (size vs. context preservation)

### Processing Speed

- Batch processing for document ingestion
- Caching for frequent queries
- Asynchronous processing for non-blocking operations

## Future Development

### Planned Features

- [ ] Multi-user support with authentication
- [ ] Document metadata extraction and filtering
- [ ] Custom training of embedding models
- [ ] API rate limiting and usage tracking
- [ ] Advanced query capabilities (filters, time ranges)

### Technical Debt

- Refactor document processing pipeline for better modularity
- Improve error handling and logging
- Enhance test coverage for edge cases

## Troubleshooting

### Common Issues

1. **PDF Extraction Failures**
   - Check PDF format compatibility
   - Verify OCR settings for scanned documents

2. **Vector Database Connection Issues**
   - Ensure proper initialization of vector store
   - Check persistence directory permissions

3. **LLM Integration Problems**
   - Verify API keys and environment variables
   - Check model availability and compatibility

## References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Chroma Vector DB](https://docs.trychroma.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

*This document is a living resource and should be updated as the project evolves.* 