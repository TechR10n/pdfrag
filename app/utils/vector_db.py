import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import logging
import time
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDBClient:
    def __init__(self, host: str, port: int, collection_name: str, vector_size: int, timeout: float = 10.0, max_retries: int = 3):
        """
        Initialize the vector database client.
        
        Args:
            host: Host of the Qdrant server
            port: Port of the Qdrant server
            collection_name: Name of the collection to use
            vector_size: Dimension of the embedding vectors
            timeout: Connection timeout in seconds
            max_retries: Maximum number of connection retries
        """
        logger.info(f"Connecting to Qdrant at {host}:{port}")
        
        # Try to connect with retries
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                self.client = QdrantClient(host=host, port=port, timeout=timeout)
                # Test the connection
                self.client.get_collections()
                logger.info(f"Successfully connected to Qdrant at {host}:{port}")
                break
            except Exception as e:
                retry_count += 1
                last_exception = e
                logger.warning(f"Connection attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        
        if retry_count == max_retries:
            logger.error(f"Failed to connect to Qdrant after {max_retries} attempts")
            if last_exception:
                raise last_exception
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        
    def create_collection(self, max_retries: int = 3) -> None:
        """Create a collection in the vector database."""
        logger.info(f"Creating collection: {self.collection_name}")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Check if collection already exists
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if self.collection_name in collection_names:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return
                
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    # Add optimizers config for better performance
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000  # Use memmapped storage for collections > 20k vectors
                    )
                )
                
                logger.info(f"Collection {self.collection_name} created")
                return
            except Exception as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count}/{max_retries} to create collection failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to create collection after {max_retries} attempts")
                    raise
    
    def delete_collection(self, max_retries: int = 3) -> None:
        """
        Delete the collection.
        
        Args:
            max_retries: Maximum number of retries
        """
        logger.info(f"Deleting collection: {self.collection_name}")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Collection {self.collection_name} deleted")
                return
            except Exception as e:
                retry_count += 1
                logger.warning(f"Delete attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to delete collection after {max_retries} attempts: {str(e)}")
    
    def upload_vectors(self, df: pd.DataFrame, 
                      vector_column: str = 'embedding',
                      batch_size: int = 100,
                      max_retries: int = 3) -> None:
        """
        Upload vectors to the collection.
        
        Args:
            df: DataFrame with embeddings
            vector_column: Name of the column containing embeddings
            batch_size: Batch size for uploading
            max_retries: Maximum number of retries for failed uploads
        """
        logger.info(f"Uploading {len(df)} vectors to collection {self.collection_name}")
        
        # Ensure collection exists
        self.create_collection()
        
        # Prepare points for upload
        points = []
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Preparing vectors"):
            # Convert embedding to list if it's a numpy array
            embedding = row[vector_column]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Create point
            point = models.PointStruct(
                id=i,  # Use DataFrame index as ID
                vector=embedding,
                payload={
                    'chunk_id': row['chunk_id'],
                    'pdf_path': row['pdf_path'],
                    'filename': row['filename'],
                    'chunk_index': row['chunk_index'],
                    'chunk_text': row['chunk_text'],
                    'token_count': row['token_count']
                }
            )
            
            points.append(point)
        
        # Upload in batches
        total_batches = (len(points) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(points), batch_size), total=total_batches, desc="Uploading batches"):
            batch = points[i:i+batch_size]
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Batch upload attempt {retry_count}/{max_retries} failed: {str(e)}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to upload batch after {max_retries} attempts")
                        raise
        
        logger.info(f"Uploaded {len(df)} vectors to collection {self.collection_name}")
    
    def search(self, query_vector: List[float], limit: int = 5, max_retries: int = 3) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            max_retries: Maximum number of retries
            
        Returns:
            List of search results
        """
        logger.info(f"Searching collection {self.collection_name} for similar vectors")
        
        # Convert query vector to list if it's a numpy array
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        # Search with retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Search
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
                
                # Convert to list of dictionaries
                search_results = []
                for result in results:
                    item = result.payload
                    item['score'] = result.score
                    search_results.append(item)
                
                logger.info(f"Found {len(search_results)} results")
                return search_results
            except Exception as e:
                retry_count += 1
                logger.warning(f"Search attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to search after {max_retries} attempts: {str(e)}")
                    return []
    
    def count_vectors(self, max_retries: int = 3) -> int:
        """
        Count the number of vectors in the collection.
        
        Args:
            max_retries: Maximum number of retries
            
        Returns:
            Number of vectors
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                count = self.client.count(collection_name=self.collection_name).count
                return count
            except Exception as e:
                retry_count += 1
                logger.warning(f"Count attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to count vectors after {max_retries} attempts: {str(e)}")
                    return 0

def setup_vector_db(host: str, port: int, collection_name: str, vector_size: int) -> VectorDBClient:
    """
    Set up the vector database.
    
    Args:
        host: Host of the Qdrant server
        port: Port of the Qdrant server
        collection_name: Name of the collection to use
        vector_size: Dimension of the embedding vectors
        
    Returns:
        Vector database client
    """
    # Create client
    client = VectorDBClient(host, port, collection_name, vector_size)
    
    # Create collection
    client.create_collection()
    
    return client

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from app.config.settings import (
        VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
        PDF_UPLOAD_FOLDER, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP
    )
    from app.utils.pdf_ingestion import process_pdfs
    from app.utils.text_chunking import process_chunks
    from app.utils.embedding_generation import embed_chunks
    
    # Process PDFs
    pdf_df = process_pdfs(PDF_UPLOAD_FOLDER)
    
    # Process chunks
    chunks_df = process_chunks(pdf_df, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Generate embeddings
    chunks_with_embeddings = embed_chunks(chunks_df, EMBEDDING_MODEL_PATH)
    
    # Set up vector database
    vector_db = setup_vector_db(VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION)
    
    # Upload vectors
    vector_db.upload_vectors(chunks_with_embeddings)
    
    # Count vectors
    count = vector_db.count_vectors()
    print(f"Vector database contains {count} vectors")
