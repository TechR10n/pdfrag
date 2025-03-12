# LLM API

This section documents the Language Model API of the PDF RAG System.

## LLM Processor

### LLMProcessor

The class for processing queries with the language model.

```python
class LLMProcessor:
    def __init__(self, model_path):
        """Initialize the LLM processor."""
        
    def generate_response(self, prompt, max_tokens=512):
        """
        Generate a response from the language model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
```

## RAG Processing

The RAG processing utilities for combining search results with LLM generation. 