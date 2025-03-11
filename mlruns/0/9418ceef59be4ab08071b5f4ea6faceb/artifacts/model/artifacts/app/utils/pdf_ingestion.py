import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scan_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Scan a directory for PDF files.
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        List of dictionaries with PDF file information
    """
    logger.info(f"Scanning directory: {directory_path}")
    pdf_files = []
    
    for path in tqdm(list(Path(directory_path).rglob('*.pdf'))):
        try:
            # Get basic file information
            file_info = {
                'path': str(path),
                'filename': path.name,
                'parent_dir': str(path.parent),
                'size_bytes': path.stat().st_size,
                'last_modified': path.stat().st_mtime
            }
            
            # Get PDF-specific metadata if possible
            try:
                with fitz.open(str(path)) as doc:
                    file_info['page_count'] = len(doc)
                    file_info['metadata'] = doc.metadata
            except Exception as e:
                logger.warning(f"Could not read PDF metadata for {path}: {str(e)}")
                file_info['page_count'] = 0
                file_info['metadata'] = {}
                
            pdf_files.append(file_info)
        except Exception as e:
            logger.error(f"Error processing {path}: {str(e)}")
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files

def create_pdf_dataframe(pdf_files: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame from PDF file information.
    
    Args:
        pdf_files: List of dictionaries with PDF file information
        
    Returns:
        DataFrame with PDF file information
    """
    return pd.DataFrame(pdf_files)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text
    """
    logger.info(f"Extracting text from: {pdf_path}")
    
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page_num, page in enumerate(doc):
                # Get text with blocks (preserves some structure)
                blocks = page.get_text("blocks")
                # Join blocks with newlines, preserving structure
                page_text = "\n".join(block[4] for block in blocks if block[6] == 0)  # block[6] == 0 means text block
                text += page_text + "\n\n"  # Add separation between pages
                
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def process_pdfs(directory_path: str) -> pd.DataFrame:
    """
    Process all PDFs in a directory.
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        DataFrame with PDF information and extracted text
    """
    # Scan directory for PDFs
    pdf_files = scan_directory(directory_path)
    
    # Create DataFrame
    df = create_pdf_dataframe(pdf_files)
    
    # Extract text from each PDF
    tqdm.pandas(desc="Extracting text")
    df['text'] = df['path'].progress_apply(extract_text_from_pdf)
    
    # Filter out PDFs with no text
    text_lengths = df['text'].str.len()
    logger.info(f"Text extraction statistics: min={text_lengths.min()}, max={text_lengths.max()}, "
                f"mean={text_lengths.mean():.2f}, median={text_lengths.median()}")
    
    empty_pdfs = df[df['text'].str.len() == 0]
    if not empty_pdfs.empty:
        logger.warning(f"Found {len(empty_pdfs)} PDFs with no extractable text")
    
    return df

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import PDF_UPLOAD_FOLDER
    
    df = process_pdfs(PDF_UPLOAD_FOLDER)
    print(f"Processed {len(df)} PDFs")
    print(df.head())
