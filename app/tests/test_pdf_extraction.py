"""Tests for PDF extraction functionality using mocks."""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.utils.pdf_ingestion import extract_text_from_pdf, process_pdfs


@pytest.fixture
def mock_pdf_path():
    return os.path.join("test_dir", "sample.pdf")


@pytest.fixture
def mock_pdf_dir():
    return "test_dir"


@pytest.mark.unit
@pytest.mark.pdf
def test_extract_text_with_mock(mock_pdf_path):
    """Test extract_text_from_pdf with a mocked PDF file."""
    expected_text = "This is sample text from a PDF file."
    
    # Mock fitz.open
    with patch("app.utils.pdf_ingestion.fitz.open") as mock_open:
        # Configure the mock
        mock_doc = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_doc
        
        # Set up the pages
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (0, 0, 0, 0, expected_text, 0, 0, 0)
        ]
        mock_doc.__iter__.return_value = [mock_page]
        
        # Call the function
        result = extract_text_from_pdf(mock_pdf_path)
        
        # Assertions
        assert expected_text in result
        mock_open.assert_called_once_with(mock_pdf_path)


@pytest.mark.unit
@pytest.mark.pdf
def test_extract_text_with_exception(mock_pdf_path):
    """Test extract_text_from_pdf with an exception."""
    # Mock fitz.open to raise an exception
    with patch("app.utils.pdf_ingestion.fitz.open") as mock_open:
        mock_open.side_effect = Exception("PDF Error")
        
        # Mock the logger
        with patch("app.utils.pdf_ingestion.logger") as mock_logger:
            # Call the function
            result = extract_text_from_pdf(mock_pdf_path)
            
            # Assertions
            assert result == ""
            mock_logger.error.assert_called_once()
            assert "Error extracting text from" in mock_logger.error.call_args[0][0]


@pytest.mark.unit
@pytest.mark.pdf
@patch("app.utils.pdf_ingestion.scan_directory")
@patch("app.utils.pdf_ingestion.extract_text_from_pdf")
@patch("app.utils.pdf_ingestion.create_pdf_dataframe")
@patch("app.utils.pdf_ingestion.tqdm.pandas")
def test_process_pdfs_with_mocks(mock_tqdm_pandas, mock_create_df, mock_extract, mock_scan, mock_pdf_dir):
    """Test process_pdfs with mocked dependencies."""
    # Configure mocks
    mock_scan.return_value = [
        {"path": "test_dir/doc1.pdf", "filename": "doc1.pdf", "size_bytes": 1000},
        {"path": "test_dir/doc2.pdf", "filename": "doc2.pdf", "size_bytes": 2000},
    ]
    
    # Create a mock DataFrame
    mock_df = MagicMock()
    mock_df.__getitem__.return_value.progress_apply.return_value = ["Text from doc1", "Text from doc2"]
    
    # Mock the text_lengths Series
    mock_text_lengths = MagicMock()
    mock_text_lengths.min.return_value = 10
    mock_text_lengths.max.return_value = 20
    mock_text_lengths.mean.return_value = 15
    mock_text_lengths.median.return_value = 15
    
    # Set up the str.len() to return the mock Series
    mock_df.__getitem__.return_value.str.len.return_value = mock_text_lengths
    
    # Set up the empty text count
    mock_df.__getitem__.return_value.apply.return_value.sum.return_value = 0
    
    mock_create_df.return_value = mock_df
    
    # Call the function
    result = process_pdfs(mock_pdf_dir)
    
    # Assertions
    assert result is mock_df
    mock_scan.assert_called_once_with(mock_pdf_dir)
    mock_create_df.assert_called_once_with(mock_scan.return_value) 