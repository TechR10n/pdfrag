"""
Pytest version of the PDF ingestion tests.
"""

import os
import pytest
import pandas as pd
from app.utils.pdf_ingestion import scan_directory, create_pdf_dataframe, extract_text_from_pdf, process_pdfs, logger

def test_scan_directory(sample_pdf_dir, empty_dir):
    """Test the scan_directory function."""
    # Test with a directory containing PDFs and other files
    pdf_files = scan_directory(sample_pdf_dir)
    assert len(pdf_files) == 4  # Expect 4 PDFs: text.pdf, blank.pdf, corrupted.pdf, sub_text.pdf
    
    filenames = [f['filename'] for f in pdf_files]
    assert 'text.pdf' in filenames
    assert 'blank.pdf' in filenames
    assert 'corrupted.pdf' in filenames
    assert 'sub_text.pdf' in filenames

    # Verify metadata for each PDF
    for f in pdf_files:
        if f['filename'] == 'text.pdf':
            assert f['page_count'] == 1
            assert f['parent_dir'] == sample_pdf_dir
        elif f['filename'] == 'blank.pdf':
            assert f['page_count'] == 1
        elif f['filename'] == 'corrupted.pdf':
            assert f['page_count'] == 0  # Corrupted PDF should have page_count 0
        elif f['filename'] == 'sub_text.pdf':
            assert f['page_count'] == 1
            assert f['parent_dir'] == os.path.join(sample_pdf_dir, 'subdir')

    # Test with an empty directory
    pdf_files = scan_directory(empty_dir)
    assert len(pdf_files) == 0

    # Verify logging for corrupted PDF
    with pytest.warns(UserWarning, match="Could not read PDF metadata"):
        scan_directory(sample_pdf_dir)

def test_create_pdf_dataframe(sample_pdf_data):
    """Test the create_pdf_dataframe function."""
    # Test with a sample list of dictionaries
    df = create_pdf_dataframe(sample_pdf_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['path', 'filename', 'size_bytes']

    # Test with an empty list
    df = create_pdf_dataframe([])
    assert len(df) == 0

def test_extract_text_from_pdf(sample_pdf_dir):
    """Test the extract_text_from_pdf function."""
    # Test with a PDF containing text
    text = extract_text_from_pdf(os.path.join(sample_pdf_dir, 'text.pdf'))
    assert 'Hello, world!' in text

    # Test with a blank PDF
    text = extract_text_from_pdf(os.path.join(sample_pdf_dir, 'blank.pdf'))
    assert text == ''

    # Test with a corrupted PDF
    with pytest.raises(Exception):
        extract_text_from_pdf(os.path.join(sample_pdf_dir, 'corrupted.pdf'))

def test_process_pdfs(sample_pdf_dir, empty_dir):
    """Test the process_pdfs function."""
    # Test with a directory containing PDFs
    df = process_pdfs(sample_pdf_dir)
    assert len(df) == 4  # Expect 4 rows in the DataFrame
    
    text_pdf_text = df.loc[df['filename'] == 'text.pdf', 'text'].iloc[0]
    assert 'Hello, world!' in text_pdf_text
    
    blank_pdf_text = df.loc[df['filename'] == 'blank.pdf', 'text'].iloc[0]
    assert blank_pdf_text == ''
    
    corrupted_pdf_text = df.loc[df['filename'] == 'corrupted.pdf', 'text'].iloc[0]
    assert corrupted_pdf_text == ''
    
    sub_text_pdf_text = df.loc[df['filename'] == 'sub_text.pdf', 'text'].iloc[0]
    assert 'Subdirectory PDF' in sub_text_pdf_text

    # Verify logging for PDFs with no extractable text
    with pytest.warns(UserWarning, match="Found 2 PDFs with no extractable text"):
        process_pdfs(sample_pdf_dir)

    # Test with an empty directory
    df = process_pdfs(empty_dir)
    assert len(df) == 0 