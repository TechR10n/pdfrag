# PDF Ingestion

This section documents the PDF ingestion functionality of the system.

## Overview

The PDF ingestion module provides functionality for scanning directories for PDF files, extracting text from PDFs, and processing them for use in the RAG system.

## Main Functions

### scan_directory

```python
def scan_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Scan a directory for PDF files and collect their metadata.
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        List of dictionaries with PDF metadata
    """
```

### extract_text_from_pdf

```python
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
```

### process_pdfs

```python
def process_pdfs(directory_path: str) -> pd.DataFrame:
    """
    Process all PDFs in a directory by scanning, creating a DataFrame, and extracting text.
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        DataFrame with PDF information and extracted text
    """
```

## Usage Example

```python
from app.utils.pdf_ingestion import process_pdfs

# Process PDFs in a directory
pdf_df = process_pdfs('/path/to/pdfs')

# Access the extracted text
for _, row in pdf_df.iterrows():
    print(f"File: {row['filename']}")
    print(f"Text length: {len(row['text'])}")
``` 