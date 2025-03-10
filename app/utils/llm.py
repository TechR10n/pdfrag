def create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for Llama 3.
    """
    # Format context
    context_text = ""
    for i, doc in enumerate(context):
        context_text += f"Document {i+1}:\n{doc['chunk_text']}\n\n"
    
    # Llama 3.x prompt format
    prompt = f"""<|system|>
You are a helpful AI assistant that provides accurate and concise answers based on the provided context documents. 
If the answer is not contained in the documents, say "I don't have enough information to answer this question."
Do not make up or hallucinate any information that is not supported by the documents.
</|system|>

<|user|>
I need information about the following topic:

{query}

Here are relevant documents to help answer this question:

{context_text}
</|user|>

<|assistant|>
"""
    return prompt