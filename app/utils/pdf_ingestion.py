"""
A script for processing PDF files in a directory, extracting metadata and text, and organizing the results into a pandas DataFrame.

This script scans a specified directory for PDF files, collects metadata such as file path, size, page count, and last modified time,
extracts the text from each PDF, and stores all the information in a pandas DataFrame for further analysis. It uses PyMuPDF for PDF operations,
pandas for data handling, and tqdm for progress tracking.
"""

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
    Scan a directory for PDF files and collect their metadata.

    This function recursively searches the given directory for PDF files and gathers
    basic file information along with PDF-specific metadata using PyMuPDF.

    Parameters:
    directory_path (str): Path to the directory containing PDF files.

    Returns:
    List[Dict[str, Any]]: A list of dictionaries, each containing information about a PDF file.
        Each dictionary has the following keys:
        - 'path': Full path to the PDF file.
        - 'filename': Name of the PDF file.
        - 'parent_dir': Path to the parent directory of the PDF file.
        - 'size_bytes': Size of the file in bytes.
        - 'last_modified': Last modification time of the file (Unix timestamp).
        - 'page_count': Number of pages in the PDF (0 if unable to read).
        - 'metadata': Dictionary of PDF metadata (empty if unable to read).

    Raises:
    None: The function handles exceptions internally and logs errors.

    Example:
    >>> pdf_files = scan_directory('/path/to/directory')
    >>> print(pdf_files[0]['filename'])
    some_file.pdf
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
    Create a pandas DataFrame from a list of PDF file information.

    This function takes a list of dictionaries containing PDF file information and
    converts it into a pandas DataFrame for easy data manipulation and analysis.

    Parameters:
    pdf_files (List[Dict[str, Any]]): List of dictionaries with PDF file information.

    Returns:
    pd.DataFrame: A DataFrame where each row represents a PDF file and columns correspond to the dictionary keys.

    Example:
    >>> pdf_files = [{'path': '/path/to/file.pdf', 'filename': 'file.pdf', ...}]
    >>> df = create_pdf_dataframe(pdf_files)
    >>> print(df.head())
    """
    return pd.DataFrame(pdf_files)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    This function opens a PDF file using PyMuPDF and extracts its text, preserving some structure
    by using text blocks (e.g., paragraphs). It joins the blocks with newlines and adds double newlines
    between pages for readability.

    Parameters:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: The extracted text from the PDF. If extraction fails, an empty string is returned.

    Raises:
    None: The function handles exceptions internally and logs errors.

    Example:
    >>> text = extract_text_from_pdf('/path/to/file.pdf')
    >>> print(text[:100])  # Print first 100 characters
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
    Process all PDFs in a directory by scanning, creating a DataFrame, and extracting text.

    This function orchestrates the PDF processing pipeline:
    1. Scans the directory for PDF files.
    2. Creates a DataFrame from the PDF file information.
    3. Extracts text from each PDF and adds it to the DataFrame.
    4. Logs statistics about the extracted text lengths.
    5. Logs a warning if any PDFs have no extractable text.

    Parameters:
    directory_path (str): Path to the directory containing PDF files.

    Returns:
    pd.DataFrame: A DataFrame with PDF information and extracted text.

    Example:
    >>> df = process_pdfs('/path/to/directory')
    >>> print(df.head())
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
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import PDF_UPLOAD_FOLDER
    
    df = process_pdfs(PDF_UPLOAD_FOLDER)
    print(f"Processed {len(df)} PDFs")
    print(df.head())
