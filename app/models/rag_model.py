import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any
import mlflow.pyfunc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Remove the USE_MOCK flag
# Set this to True to use a mock implementation for testing
# USE_MOCK = False

class RAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        """Initialize the RAG model wrapper."""
        self.rag_processor = None
    
    def load_context(self, context):
        """
        Load model artifacts.
        
        Args:
            context: MLflow model context
        """
        logger.info("Loading RAG model context")
        
        # Add paths
        sys.path.append(os.path.dirname(context.artifacts['app_dir']))
        
        # Import here to avoid circular imports
        from app.config.settings import (
            VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
            EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, MODEL_PATH
        )
        from app.utils.search import create_search_pipeline
        from app.utils.llm import create_rag_processor
        
        # Create search pipeline
        search_pipeline = create_search_pipeline(
            VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
            EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
        )
        
        # Create RAG processor with real model (no mock)
        self.rag_processor = create_rag_processor(search_pipeline, MODEL_PATH)
        
        logger.info("RAG model context loaded")
    
    def predict(self, context, model_input):
        """
        Generate predictions.
        
        Args:
            context: MLflow model context
            model_input: Input data
            
        Returns:
            Model predictions
        """
        # Check if input is a pandas DataFrame
        if isinstance(model_input, pd.DataFrame):
            # Extract query
            if 'query' in model_input.columns:
                query = model_input['query'].iloc[0]
            else:
                raise ValueError("Input DataFrame must have a 'query' column")
        elif isinstance(model_input, dict):
            # Extract query from dictionary
            if 'query' in model_input:
                query = model_input['query']
            else:
                raise ValueError("Input dictionary must have a 'query' key")
        else:
            # Assume input is a string query
            query = str(model_input)
        
        logger.info(f"Processing query: {query}")
        
        # Process query
        response = self.rag_processor.process_query(query)
        
        return response

def get_pip_requirements():
    """Get pip requirements for the model."""
    return [
        "pandas",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "qdrant-client",
        "llama-cpp-python",
        "mlflow"
    ]

if __name__ == "__main__":
    # Test the model
    model = RAGModel()
    
    # Create a dummy context
    class DummyContext:
        def __init__(self):
            self.artifacts = {'app_dir': os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    
    # Load context
    model.load_context(DummyContext())
    
    # Test prediction
    result = model.predict(None, "What is retrieval-augmented generation?")
    print(result['text'])
