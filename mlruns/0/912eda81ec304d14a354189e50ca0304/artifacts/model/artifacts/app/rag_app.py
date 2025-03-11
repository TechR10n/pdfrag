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
    VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, LLM_MODEL_PATH
)
from app.utils.search import create_search_pipeline
from app.utils.llm import create_rag_processor

class RAGApplication:
    def __init__(self):
        """Initialize the RAG application."""
        logger.info("Initializing RAG application")
        
        # Create search pipeline
        self.search_pipeline = create_search_pipeline(
            VECTOR_DB_HOST, VECTOR_DB_PORT, COLLECTION_NAME, VECTOR_DIMENSION,
            EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
        )
        
        # Create RAG processor
        self.rag_processor = create_rag_processor(self.search_pipeline, LLM_MODEL_PATH)
        
        logger.info("RAG application initialized")
    
    def process_query(self, query: str):
        """
        Process a query.
        
        Args:
            query: User query
            
        Returns:
            RAG response
        """
        return self.rag_processor.process_query(query)

def interactive_mode(app):
    """Run the application in interactive mode."""
    print("RAG Application - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("----------------------------------")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        try:
            response = app.process_query(query)
            
            print("\nAnswer:")
            print(response['text'])
            
            print("\nSources:")
            for i, source in enumerate(response['sources']):
                print(f"{i+1}. {source['filename']} (Score: {source['rerank_score']:.4f})")
                print(f"   Excerpt: {source['chunk_text'][:100]}...")
            
            print("\nMetadata:")
            print(f"Total tokens: {response['metadata']['llm']['tokens_used']}")
            print(f"Search results: {response['metadata']['search_results']}")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the RAG application')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--query', type=str,
                        help='Query to process')
    args = parser.parse_args()
    
    # Initialize application
    app = RAGApplication()
    
    if args.interactive:
        interactive_mode(app)
    elif args.query:
        response = app.process_query(args.query)
        print(response['text'])
    else:
        parser.print_help()
