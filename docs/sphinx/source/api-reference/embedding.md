# Embedding API

This section documents the embedding API of the PDF RAG System.

## Embedding Generation

### EmbeddingGenerator

The class for generating embeddings from text.

```python
class EmbeddingGenerator:
    def __init__(self, model_path):
        """Initialize the embedding generator."""
        
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
```

## Text Processing

The text processing utilities for preparing text for embedding. 