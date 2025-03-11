import os
import sys
from app.models.rag_model import RAGModel

# Create a dummy context
class DummyContext:
    def __init__(self):
        self.artifacts = {'app_dir': os.path.join(os.getcwd(), 'app')}

def main():
    print("Testing RAG model...")
    
    # Create model instance
    model = RAGModel()
    
    # Load context
    print("Loading model context...")
    model.load_context(DummyContext())
    
    # Test prediction
    query = "What is retrieval-augmented generation?"
    print(f"Processing query: {query}")
    
    try:
        result = model.predict(None, query)
        print("\nResult:")
        print(result)
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 