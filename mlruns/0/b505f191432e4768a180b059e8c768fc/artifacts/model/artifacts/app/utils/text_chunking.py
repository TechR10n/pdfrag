import re
import uuid
from typing import List, Dict, Any
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing line breaks.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple line breaks with a single one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r' {2,}', ' ', text)
    
    # Strip whitespace from beginning and end
    text = text.strip()
    
    return text

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
    # Clean the text first
    text = clean_text(text)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    return chunks

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
    logger.info(f"Processing chunks with size={chunk_size}, overlap={chunk_overlap}")
    
    chunks_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        # Skip if no text
        if not row['text'] or len(row['text']) == 0:
            continue
            
        # Get chunks
        chunks = chunk_text(row['text'], chunk_size, chunk_overlap)
        
        # Limit chunks if necessary
        if len(chunks) > max_chunks_per_doc:
            logger.warning(f"Document {row['filename']} has {len(chunks)} chunks, "
                          f"limiting to {max_chunks_per_doc}")
            chunks = chunks[:max_chunks_per_doc]
        
        # Add chunks to list
        for i, chunk_content in enumerate(chunks):
            chunk_data = {
                'chunk_id': str(uuid.uuid4()),
                'pdf_path': row['path'],
                'filename': row['filename'],
                'chunk_index': i,
                'chunk_text': chunk_content,
                'token_count': len(chunk_content.split())
            }
            chunks_data.append(chunk_data)
    
    # Create DataFrame from chunks
    chunks_df = pd.DataFrame(chunks_data)
    
    logger.info(f"Created {len(chunks_df)} chunks from {len(df)} documents")
    
    return chunks_df

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import PDF_UPLOAD_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC
    from app.utils.pdf_ingestion import process_pdfs
    
    # Process PDFs
    pdf_df = process_pdfs(PDF_UPLOAD_FOLDER)
    
    # Process chunks
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC)
    
    print(f"Created {len(chunks_df)} chunks")
    print(chunks_df.head())
