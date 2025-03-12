import os
import tempfile
import shutil
import pytest
import fitz
import pandas as pd

@pytest.fixture
def sample_pdf_dir():
    """Create a temporary directory with sample PDFs for testing."""
    test_dir = tempfile.mkdtemp()
    
    # Create sample PDFs
    create_sample_pdf_with_text(os.path.join(test_dir, "text.pdf"), "Hello, world!")
    create_blank_pdf(os.path.join(test_dir, "blank.pdf"))
    create_corrupted_pdf(os.path.join(test_dir, "corrupted.pdf"))
    
    # Create a subdirectory with a PDF
    sub_dir = os.path.join(test_dir, "subdir")
    os.mkdir(sub_dir)
    create_sample_pdf_with_text(os.path.join(sub_dir, "sub_text.pdf"), "Subdirectory PDF")
    
    # Create a non-PDF file
    with open(os.path.join(test_dir, "not_a_pdf.txt"), 'w') as f:
        f.write("This is not a PDF.")
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(test_dir)

@pytest.fixture
def empty_dir():
    """Create an empty directory for testing."""
    empty_dir = tempfile.mkdtemp()
    yield empty_dir
    shutil.rmtree(empty_dir)

@pytest.fixture
def sample_pdf_data():
    """Return sample PDF metadata for testing."""
    return [
        {'path': '/path/to/file1.pdf', 'filename': 'file1.pdf', 'size_bytes': 1000},
        {'path': '/path/to/file2.pdf', 'filename': 'file2.pdf', 'size_bytes': 2000}
    ]

@pytest.fixture
def sample_text_data():
    """Return sample text data for testing text chunking."""
    return {
        'short_text': "This is a short text that won't be chunked.",
        'medium_text': "This is a medium-length text. " * 10,
        'long_text': "This is a longer text with multiple sentences. " * 30,
        'paragraphs': (
            "Paragraph 1 with some content.\n\n"
            "Paragraph 2 with different content.\n\n"
            "Paragraph 3 with even more content.\n\n"
            "Paragraph 4 to ensure we have enough text."
        ),
        'whitespace_text': "  Text with   excessive    whitespace   \n\n\n   and line breaks.  ",
        'empty_text': ""
    }

@pytest.fixture
def sample_pdf_dataframe():
    """Return a sample DataFrame with PDF data including text content."""
    return pd.DataFrame([
        {
            'path': '/path/to/doc1.pdf',
            'filename': 'doc1.pdf',
            'text': "This is the text content of document 1. " * 20,
            'size_bytes': 1000
        },
        {
            'path': '/path/to/doc2.pdf',
            'filename': 'doc2.pdf',
            'text': "This is the text content of document 2. " * 20,
            'size_bytes': 2000
        },
        {
            'path': '/path/to/empty.pdf',
            'filename': 'empty.pdf',
            'text': "",
            'size_bytes': 500
        }
    ])

# Helper functions
def create_sample_pdf_with_text(filename, text):
    """Create a PDF with specified text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), text)
    doc.save(filename)

def create_blank_pdf(filename):
    """Create a blank PDF with no text."""
    doc = fitz.open()
    doc.new_page()
    doc.save(filename)

def create_corrupted_pdf(filename):
    """Create a corrupted PDF by writing plain text."""
    with open(filename, 'w') as f:
        f.write("This is not a PDF.") 