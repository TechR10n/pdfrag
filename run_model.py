import os
import sys
import json
from app.models.rag_model import RAGModel

def main():
    """Run the RAG model directly"""
    # Create a dummy context
    class DummyContext:
        def __init__(self):
            self.artifacts = {'app_dir': os.path.join(os.getcwd(), 'app')}
    
    # Initialize model
    print("Initializing model...")
    model = RAGModel()
    
    # Load context
    print("Loading model context...")
    model.load_context(DummyContext())
    
    # Process queries
    while True:
        try:
            # Get query from user
            query = input("\nEnter your query (or 'exit' to quit): ")
            
            # Exit if requested
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            # Process query
            print(f"Processing query: {query}")
            result = model.predict(None, query)
            
            # Print result
            print("\nResult:")
            print(f"Answer: {result['text']}")
            print("\nSources:")
            for i, source in enumerate(result['sources'][:3]):  # Show top 3 sources
                print(f"{i+1}. {source['filename']}")
                print(f"   Score: {source['rerank_score']:.2f}")
                print(f"   Text: {source['chunk_text'][:100]}...")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Exiting...")

if __name__ == "__main__":
    main() 