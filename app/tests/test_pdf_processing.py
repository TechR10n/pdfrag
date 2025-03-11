import os
import sys
import pytest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.pdf_ingestion import scan_directory, extract_text_from_pdf, process_pdfs
from app.utils.text_chunking import chunk_text, process_chunks
from app.config.settings import PDF_UPLOAD_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP

def test_scan_directory():
    """Test scanning directory for PDFs."""
    # Create a test PDF if none exists
    if not list(Path(PDF_UPLOAD_FOLDER).rglob('*.pdf')):
        pytest.skip("No PDF files found for testing")
    
    pdfs = scan_directory(PDF_UPLOAD_FOLDER)
    assert len(pdfs) > 0, "No PDFs found in directory"
    assert 'path' in pdfs[0], "PDF info missing 'path'"
    assert 'filename' in pdfs[0], "PDF info missing 'filename'"

def test_extract_text():
    """Test extracting text from a PDF."""
    # Create a test PDF if none exists
    if not list(Path(PDF_UPLOAD_FOLDER).rglob('*.pdf')):
        pytest.skip("No PDF files found for testing")
    
    pdf_path = next(Path(PDF_UPLOAD_FOLDER).rglob('*.pdf'))
    text = extract_text_from_pdf(str(pdf_path))
    assert text, "No text extracted from PDF"
    
def test_chunk_text():
    """Test chunking text."""
    sample_text = """
    This is a sample document that will be split into chunks.
    It has multiple sentences and paragraphs.
    
    This is the second paragraph with some more text.
    We want to make sure the chunking works correctly.
    
    Let's add a third paragraph to ensure we have enough text to create multiple chunks.
    This should be enough for the test.
    """
    
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1, "Text not split into multiple chunks"
    assert len(chunks[0]) <= 100 + 20, "Chunk size exceeds expected maximum"

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
