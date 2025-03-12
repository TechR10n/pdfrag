import os
import random
import string
import pandas as pd
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF

def generate_random_text(min_length=100, max_length=1000, paragraphs=3):
    """
    Generate random text with paragraphs for testing.
    
    Args:
        min_length: Minimum length of the text
        max_length: Maximum length of the text
        paragraphs: Number of paragraphs to generate
        
    Returns:
        Random text with paragraphs
    """
    # Generate random text length
    length = random.randint(min_length, max_length)
    
    # Generate random words
    words = []
    while len(' '.join(words)) < length:
        word_length = random.randint(3, 12)
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    
    # Split into paragraphs
    words_per_paragraph = len(words) // paragraphs
    paragraphs_text = []
    
    for i in range(paragraphs):
        start = i * words_per_paragraph
        end = (i + 1) * words_per_paragraph if i < paragraphs - 1 else len(words)
        paragraph = ' '.join(words[start:end])
        
        # Add some sentences
        paragraph = paragraph.replace(' ', '. ', random.randint(3, 8))
        paragraph = paragraph.capitalize() + '.'
        paragraphs_text.append(paragraph)
    
    # Join paragraphs with double newlines
    return '\n\n'.join(paragraphs_text)

def generate_test_pdf(output_path, text=None, pages=1):
    """
    Generate a test PDF with specified text.
    
    Args:
        output_path: Path to save the PDF
        text: Text to include in the PDF (if None, random text is generated)
        pages: Number of pages to generate
        
    Returns:
        Path to the generated PDF
    """
    # Create a new PDF document
    doc = fitz.open()
    
    for _ in range(pages):
        # Add a page
        page = doc.new_page()
        
        # Generate random text if not provided
        if text is None:
            text = generate_random_text()
        
        # Insert text
        page.insert_text((50, 50), text)
    
    # Save the PDF
    doc.save(output_path)
    
    return output_path

def generate_test_pdfs(output_dir, count=5, min_pages=1, max_pages=5):
    """
    Generate multiple test PDFs with random text.
    
    Args:
        output_dir: Directory to save the PDFs
        count: Number of PDFs to generate
        min_pages: Minimum number of pages per PDF
        max_pages: Maximum number of pages per PDF
        
    Returns:
        List of paths to the generated PDFs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_paths = []
    
    for i in range(count):
        # Generate random number of pages
        pages = random.randint(min_pages, max_pages)
        
        # Generate PDF path
        pdf_path = os.path.join(output_dir, f"test_pdf_{i+1}.pdf")
        
        # Generate PDF
        generate_test_pdf(pdf_path, pages=pages)
        
        pdf_paths.append(pdf_path)
    
    return pdf_paths

def generate_test_dataframe(pdf_paths=None, count=5):
    """
    Generate a test DataFrame with PDF metadata and text.
    
    Args:
        pdf_paths: List of PDF paths (if None, random paths are generated)
        count: Number of PDFs to include in the DataFrame
        
    Returns:
        DataFrame with PDF metadata and text
    """
    if pdf_paths is None:
        # Generate random PDF paths
        pdf_paths = [f"/path/to/pdf_{i+1}.pdf" for i in range(count)]
    
    data = []
    
    for path in pdf_paths:
        # Get filename from path
        filename = os.path.basename(path)
        
        # Generate random text
        text = generate_random_text(min_length=500, max_length=5000, paragraphs=5)
        
        # Generate random size
        size_bytes = random.randint(1000, 10000000)
        
        # Add to data
        data.append({
            'path': path,
            'filename': filename,
            'text': text,
            'size_bytes': size_bytes
        })
    
    return pd.DataFrame(data)

def generate_test_chunks_dataframe(pdf_df=None, chunks_per_doc=5):
    """
    Generate a test DataFrame with text chunks.
    
    Args:
        pdf_df: DataFrame with PDF metadata and text
        chunks_per_doc: Number of chunks per document
        
    Returns:
        DataFrame with text chunks
    """
    if pdf_df is None:
        # Generate random PDF DataFrame
        pdf_df = generate_test_dataframe()
    
    chunks_data = []
    
    for _, row in pdf_df.iterrows():
        # Skip if no text
        if not row['text'] or len(row['text']) == 0:
            continue
        
        # Split text into chunks (simple splitting for testing)
        text_length = len(row['text'])
        chunk_size = text_length // chunks_per_doc
        
        for i in range(chunks_per_doc):
            # Calculate chunk start and end
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < chunks_per_doc - 1 else text_length
            
            # Extract chunk text
            chunk_text = row['text'][start:end]
            
            # Add chunk to list
            chunk_data = {
                'chunk_id': f"chunk_{row['filename']}_{i+1}",
                'pdf_path': row['path'],
                'filename': row['filename'],
                'chunk_index': i,
                'chunk_text': chunk_text,
                'token_count': len(chunk_text.split())
            }
            chunks_data.append(chunk_data)
    
    return pd.DataFrame(chunks_data)

def generate_test_embeddings(chunks_df=None, embedding_dim=384):
    """
    Generate test embeddings for text chunks.
    
    Args:
        chunks_df: DataFrame with text chunks
        embedding_dim: Dimension of the embeddings
        
    Returns:
        DataFrame with text chunks and embeddings
    """
    if chunks_df is None:
        # Generate random chunks DataFrame
        chunks_df = generate_test_chunks_dataframe()
    
    # Copy the DataFrame
    embeddings_df = chunks_df.copy()
    
    # Add random embeddings
    embeddings = []
    for _ in range(len(chunks_df)):
        # Generate random embedding
        embedding = np.random.randn(embedding_dim)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
    
    # Add embeddings to DataFrame
    embeddings_df['embedding'] = embeddings
    
    return embeddings_df

if __name__ == "__main__":
    # Example usage
    output_dir = "app/tests/data/test_pdfs"
    
    # Generate test PDFs
    pdf_paths = generate_test_pdfs(output_dir, count=3)
    print(f"Generated {len(pdf_paths)} test PDFs in {output_dir}")
    
    # Generate test DataFrame
    pdf_df = generate_test_dataframe(pdf_paths)
    print(f"Generated DataFrame with {len(pdf_df)} PDFs")
    
    # Generate test chunks
    chunks_df = generate_test_chunks_dataframe(pdf_df)
    print(f"Generated {len(chunks_df)} chunks")
    
    # Generate test embeddings
    embeddings_df = generate_test_embeddings(chunks_df)
    print(f"Generated embeddings with dimension {len(embeddings_df['embedding'][0])}") 