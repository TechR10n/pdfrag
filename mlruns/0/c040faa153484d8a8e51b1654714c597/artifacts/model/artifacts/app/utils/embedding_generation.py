import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_path: str, batch_size: int = 32):
        """
        Initialize the embedding generator.
        
        Args:
            model_path: Path to the embedding model
            batch_size: Batch size for embedding generation
        """
        logger.info(f"Loading embedding model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {self.batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'chunk_text',
                         embedding_column: str = 'embedding') -> pd.DataFrame:
        """
        Process a DataFrame and add embeddings.
        
        Args:
            df: DataFrame with text chunks
            text_column: Name of the column containing text
            embedding_column: Name of the column to store embeddings
            
        Returns:
            DataFrame with embeddings
        """
        logger.info(f"Processing DataFrame with {len(df)} rows")
        
        # Get texts
        texts = df[text_column].tolist()
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to DataFrame
        df[embedding_column] = list(embeddings)
        
        return df

def embed_chunks(chunks_df: pd.DataFrame, model_path: str, batch_size: int = 32) -> pd.DataFrame:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks_df: DataFrame with text chunks
        model_path: Path to the embedding model
        batch_size: Batch size for embedding generation
        
    Returns:
        DataFrame with embeddings
    """
    # Create embedder
    embedder = EmbeddingGenerator(model_path, batch_size)
    
    # Process DataFrame
    chunks_df = embedder.process_dataframe(chunks_df)
    
    return chunks_df

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import PDF_UPLOAD_FOLDER, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP
    from app.utils.pdf_ingestion import process_pdfs
    from app.utils.text_chunking import process_chunks
    
    # Process PDFs
    pdf_df = process_pdfs(PDF_UPLOAD_FOLDER)
    
    # Process chunks
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Generate embeddings
    chunks_with_embeddings = embed_chunks(chunks_df, EMBEDDING_MODEL_PATH)
    
    print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    print(f"Embedding dimension: {len(chunks_with_embeddings['embedding'].iloc[0])}")
