import os
import numpy as np
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""This module provides functionality to generate embeddings for text chunks using a Sentence Transformer model.

It includes a class for generating embeddings and a utility function to process pandas DataFrames containing 
text chunks, enabling efficient embedding generation for text data.
"""

class EmbeddingGenerator:
    """The EmbeddingGenerator class handles loading a Sentence Transformer model and generating embeddings for text data.

    It offers methods to create embeddings from lists of text strings and to process pandas DataFrames by adding 
    embeddings to specified columns, facilitating downstream tasks like similarity computation.
    """

    def __init__(self, model_path: str, batch_size: int = 32):
        """Initializes the EmbeddingGenerator with a model path and batch size for processing.

        Args:
            model_path (str): The file path to the pre-trained Sentence Transformer model.
            batch_size (int, optional): The number of text items to process in each batch. Defaults to 32.

        The constructor loads the specified model and retrieves its embedding dimension for later use.
        """
        logger.info(f"Loading embedding model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Creates embeddings for a list of text strings using the Sentence Transformer model.

        Args:
            texts (List[str]): A list of text strings to generate embeddings for.

        Returns:
            np.ndarray: A 2D NumPy array where each row represents the embedding vector of the corresponding text.

        Texts are processed in batches for efficiency, and embeddings are normalized to support similarity calculations.
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
        """Processes a DataFrame by adding embeddings for text in a specified column to a new column.

        Args:
            df (pd.DataFrame): The input DataFrame with text data to process.
            text_column (str, optional): The column name containing text to embed. Defaults to 'chunk_text'.
            embedding_column (str, optional): The column name to store embeddings. Defaults to 'embedding'.

        Returns:
            pd.DataFrame: The input DataFrame augmented with a new column of embeddings.
        """
        logger.info(f"Processing DataFrame with {len(df)} rows")
        
        # Extract texts from the specified column
        texts = df[text_column].tolist()
        
        # Generate embeddings for the texts
        embeddings = self.generate_embeddings(texts)
        
        # Assign embeddings to the new column
        df[embedding_column] = list(embeddings)
        
        return df

def embed_chunks(chunks_df: pd.DataFrame, model_path: str, batch_size: int = 32) -> pd.DataFrame:
    """Generates embeddings for text chunks in a DataFrame using a Sentence Transformer model.

    Args:
        chunks_df (pd.DataFrame): A DataFrame with text chunks to embed.
        model_path (str): The file path to the pre-trained Sentence Transformer model.
        batch_size (int, optional): The number of texts to process per batch. Defaults to 32.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'embedding' column containing the generated embeddings.

    This function acts as a convenient wrapper, initializing an EmbeddingGenerator and processing the DataFrame in one step.
    """
    # Initialize the embedder with the specified model and batch size
    embedder = EmbeddingGenerator(model_path, batch_size)
    
    # Process the DataFrame to add embeddings
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
    
    # Process PDFs into a DataFrame
    pdf_df = process_pdfs(PDF_UPLOAD_FOLDER)
    
    # Chunk the text data
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Generate embeddings for the chunks
    chunks_with_embeddings = embed_chunks(chunks_df, EMBEDDING_MODEL_PATH)
    
    print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    print(f"Embedding dimension: {len(chunks_with_embeddings['embedding'].iloc[0])}")
