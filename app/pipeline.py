import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import (
    PDF_UPLOAD_FOLDER, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC,
    VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION
)
from app.utils.pdf_ingestion import process_pdfs
from app.utils.text_chunking import process_chunks
from app.utils.embedding_generation import embed_chunks
from app.utils.vector_db import setup_vector_db

def run_pipeline(pdf_dir: str, rebuild_index: bool = False):
    """
    Run the full pipeline from PDF ingestion to vector database upload.
    
    Args:
        pdf_dir: Directory containing PDF files
        rebuild_index: Whether to rebuild the vector index (delete and recreate)
    """
    logger.info(f"Starting pipeline with PDF directory: {pdf_dir}")
    
    # Step 1: Process PDFs
    logger.info("Step 1: Processing PDFs")
    pdf_df = process_pdfs(pdf_dir)
    logger.info(f"Processed {len(pdf_df)} PDFs")
    
    # Step 2: Process chunks
    logger.info("Step 2: Processing chunks")
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC)
    logger.info(f"Created {len(chunks_df)} chunks")
    
    # Step 3: Generate embeddings
    logger.info("Step 3: Generating embeddings")
    chunks_with_embeddings = embed_chunks(chunks_df, EMBEDDING_MODEL_PATH)
    logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Step 4: Set up vector database
    logger.info(f"Step 4: Setting up vector database at {VECTOR_DB_HOST}:{VECTOR_DB_PORT}")
    vector_db = setup_vector_db(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION)
    
    # Delete collection if rebuilding index
    if rebuild_index:
        logger.info("Rebuilding vector index: deleting existing collection")
        vector_db.delete_collection()
        vector_db.create_collection()
    
    # Step 5: Upload vectors
    logger.info("Step 5: Uploading vectors")
    vector_db.upload_vectors(chunks_with_embeddings)
    
    # Verify upload
    count = vector_db.count_vectors()
    logger.info(f"Pipeline complete. Vector database contains {count} vectors")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the PDF processing pipeline')
    parser.add_argument('--pdf-dir', type=str, default=PDF_UPLOAD_FOLDER,
                        help='Directory containing PDF files')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild the vector index (delete and recreate)')
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(args.pdf_dir, args.rebuild)
