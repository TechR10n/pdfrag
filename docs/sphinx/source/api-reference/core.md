# Core API

This section documents the core API of the PDF RAG System.

## Main Classes

### RAGApplication

The main application class that coordinates all components.

```python
class RAGApplication:
    def __init__(self):
        """Initialize the RAG application."""
        
    def process_query(self, query: str):
        """
        Process a query.
        
        Args:
            query: User query
            
        Returns:
            RAG response
        """
```

## Configuration

The core configuration options for the system. 