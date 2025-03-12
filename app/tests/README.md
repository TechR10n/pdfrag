# Testing Guide for PDF RAG System

This directory contains tests for the PDF RAG System. The tests are organized by module and functionality.

## Running Tests

You can run tests using the provided `run_tests.py` script in the project root:

```bash
# Run all tests with pytest (default)
./run_tests.py

# Run with coverage report
./run_tests.py --coverage

# Run specific test file
./run_tests.py --test-file app/tests/test_pdf_ingestion.py

# Run with unittest framework
./run_tests.py --framework unittest
```

### Test Categories

The tests are categorized using markers. You can run specific categories of tests:

```bash
# Run only unit tests
./run_tests.py --unit

# Run only integration tests
./run_tests.py --integration

# Run only API tests
./run_tests.py --api

# Run only model tests
./run_tests.py --model

# Run only PDF-related tests
./run_tests.py --pdf

# Include slow tests (skipped by default)
./run_tests.py --runslow
```

Alternatively, you can use pytest or unittest directly:

```bash
# Using pytest
pytest app/tests
pytest app/tests --cov=app
pytest app/tests -m unit  # Run only unit tests

# Using unittest
python -m unittest discover app/tests
```

## Test Structure

- `conftest.py`: Contains pytest fixtures used across multiple test files
- `test_pdf_ingestion.py`: Tests for PDF scanning and text extraction
- `test_pdf_extraction.py`: Mock tests for PDF extraction functionality
- `test_integration.py`: Integration tests for the full system
- `test_text_chunking.py`: Unit tests for text chunking functionality
- `test_text_chunking_integration.py`: Integration tests for text chunking with other components

## Text Chunking Tests

The text chunking tests are organized into several files:

- `test_text_chunking.py`: Contains unit tests for the text chunking functionality
- `test_text_chunking_integration.py`: Contains integration tests for text chunking with other components
- `test_data_generator.py`: Utility for generating test data for chunking tests
- `run_chunking_tests.py`: Script to run all text chunking tests

### Running Text Chunking Tests

You can run the text chunking tests specifically using:

```bash
# Run all text chunking tests
python app/tests/run_chunking_tests.py

# Generate test data for chunking tests
python app/tests/run_chunking_tests.py --generate-data

# Generate data and run tests
python app/tests/run_chunking_tests.py --generate-data --run-tests
```

### Text Chunking Test Structure

The text chunking tests cover:

1. **Unit Tests**:
   - `TestCleanText`: Tests for the text cleaning functionality
   - `TestChunkText`: Tests for the text chunking functionality
   - `TestProcessChunks`: Tests for processing chunks from a DataFrame

2. **Integration Tests**:
   - `TestTextChunkingWithPDFIngestion`: Tests integration with PDF ingestion
   - `TestTextChunkingWithVectorDB`: Tests integration with vector database
   - `TestEndToEndProcessing`: End-to-end tests for the document processing pipeline

3. **Test Data Generation**:
   - `generate_random_text()`: Generates random text with paragraphs
   - `generate_test_pdf()`: Generates a test PDF with specified text
   - `generate_test_pdfs()`: Generates multiple test PDFs
   - `generate_test_dataframe()`: Generates a test DataFrame with PDF metadata
   - `generate_test_chunks_dataframe()`: Generates a test DataFrame with text chunks
   - `generate_test_embeddings()`: Generates test embeddings for text chunks

## Writing New Tests

When writing new tests:

1. Create a new file named `test_<module_name>.py`
2. Use pytest fixtures from `conftest.py` when possible
3. Use mocks for external dependencies
4. Ensure tests are isolated and don't depend on external resources
5. Add appropriate markers to categorize your tests:
   ```python
   @pytest.mark.unit
   def test_something():
       # Unit test implementation
       
   @pytest.mark.integration
   def test_integration():
       # Integration test implementation
       
   @pytest.mark.slow
   def test_slow_operation():
       # Slow test implementation
   ```

## Test Data

The `conftest.py` file provides fixtures for creating test data, including:

- `sample_pdf_dir`: A temporary directory with sample PDFs
- `empty_dir`: An empty directory for testing
- `sample_pdf_data`: Sample PDF metadata
- `sample_text_data`: Sample text data for testing text chunking
- `sample_pdf_dataframe`: Sample DataFrame with PDF data including text content

## Code Coverage

To generate a code coverage report:

```bash
./run_tests.py --coverage
```

This will create an HTML report in the `htmlcov` directory that you can open in a browser. 