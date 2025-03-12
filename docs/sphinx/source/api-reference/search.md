# Search API

This section documents the search API of the PDF RAG System.

## Vector Search

### VectorDBClient

The client for interacting with the vector database.

```python
class VectorDBClient:
    def __init__(self, host, port, collection_name, vector_dimension):
        """Initialize the vector database client."""
        
    def search(self, query_vector, limit=5):
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query vector
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
```

## Reranking

The reranking API for improving search results. 