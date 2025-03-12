# Text Chunking

This section documents the text chunking functionality of the system.

## Overview

The text chunking module provides functionality for splitting text into manageable chunks for processing by the RAG system. It includes functions for cleaning text, chunking it into smaller pieces, and processing chunks from a DataFrame of PDF data.

## Main Functions

### clean_text

```python
def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing line breaks.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
```

### chunk_text

```python
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text: Text to split into chunks
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
```

### process_chunks

```python
def process_chunks(df: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 50, 
                  max_chunks_per_doc: int = 1000) -> pd.DataFrame:
    """
    Process a DataFrame of PDFs and split text into chunks.
    
    Args:
        df: DataFrame with PDF information and extracted text
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
        max_chunks_per_doc: Maximum number of chunks per document
        
    Returns:
        DataFrame with text chunks
    """
```

## Usage Example

```python
from app.utils.text_chunking import chunk_text, process_chunks
from app.utils.pdf_ingestion import process_pdfs

# Process PDFs
pdf_df = process_pdfs('/path/to/pdfs')

# Process chunks
chunks_df = process_chunks(pdf_df, chunk_size=200, chunk_overlap=50)

# Access the chunks
for _, row in chunks_df.iterrows():
    print(f"Chunk {row['chunk_index']} from {row['filename']}")
    print(f"Text: {row['chunk_text'][:100]}...")
    print(f"Token count: {row['token_count']}")
``` 