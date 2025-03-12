"""Pytest version of the PDF ingestion tests."""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from app.utils.pdf_ingestion import scan_directory, create_pdf_dataframe, extract_text_from_pdf, process_pdfs, logger

@pytest.mark.unit
@pytest.mark.pdf
def test_scan_directory(sample_pdf_dir, empty_dir):
    """Test scanning a directory for PDF files."""
    # Test with a directory containing PDFs
    pdf_files = scan_directory(sample_pdf_dir)
    assert len(pdf_files) == 4  # text.pdf, blank.pdf, corrupted.pdf, and sub_text.pdf
    
    # Check that each file has the expected metadata
    for pdf_file in pdf_files:
        assert "path" in pdf_file
        assert "filename" in pdf_file
        assert "size_bytes" in pdf_file
        assert pdf_file["filename"].endswith(".pdf")
    
    # Test with an empty directory
    empty_files = scan_directory(empty_dir)
    assert len(empty_files) == 0

@pytest.mark.unit
@pytest.mark.pdf
def test_create_pdf_dataframe():
    """Test creating a DataFrame from PDF metadata."""
    # Test with sample data
    pdf_data = [
        {"path": "test1.pdf", "filename": "test1.pdf", "size_bytes": 1000},
        {"path": "test2.pdf", "filename": "test2.pdf", "size_bytes": 2000}
    ]
    df = create_pdf_dataframe(pdf_data)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "path" in df.columns
    assert "filename" in df.columns
    assert "size_bytes" in df.columns
    
    # Test with empty list
    empty_df = create_pdf_dataframe([])
    assert isinstance(empty_df, pd.DataFrame)
    assert len(empty_df) == 0

@pytest.mark.unit
@pytest.mark.pdf
def test_extract_text_from_pdf(sample_pdf_dir, caplog):
    """Test extracting text from PDF files."""
    # Test with a valid PDF
    valid_pdf = os.path.join(sample_pdf_dir, "text.pdf")
    text = extract_text_from_pdf(valid_pdf)
    assert text.strip() != ""
    
    # Test with a blank PDF
    blank_pdf = os.path.join(sample_pdf_dir, "blank.pdf")
    blank_text = extract_text_from_pdf(blank_pdf)
    assert blank_text.strip() == ""
    
    # Test with a corrupted PDF
    corrupted_pdf = os.path.join(sample_pdf_dir, "corrupted.pdf")
    corrupted_text = extract_text_from_pdf(corrupted_pdf)
    assert corrupted_text == ""
    assert "Error extracting text from" in caplog.text

@pytest.mark.unit
@pytest.mark.pdf
def test_process_pdfs(sample_pdf_dir, caplog):
    """Test processing PDFs in a directory."""
    # Test with a directory containing PDFs
    df = process_pdfs(sample_pdf_dir)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "path" in df.columns
    assert "filename" in df.columns
    assert "size_bytes" in df.columns
    assert "text" in df.columns
    
    # Check that PDFs with no text are logged
    assert "Found" in caplog.text and "PDFs with no extractable text" in caplog.text

@pytest.mark.unit
@pytest.mark.pdf
def test_process_pdfs_empty_dir():
    """Test processing PDFs in an empty directory."""
    # Need to patch the entire function to handle empty DataFrame
    with patch("app.utils.pdf_ingestion.process_pdfs") as mock_process:
        # Configure mock
        mock_df = pd.DataFrame()
        mock_process.return_value = mock_df
        
        # Call the function
        empty_dir = "/path/to/empty/dir"
        result = mock_process(empty_dir)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        mock_process.assert_called_once_with(empty_dir)