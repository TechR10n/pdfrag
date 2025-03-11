"""
Prompt templates for different LLM models.
"""

def get_llama_3_2_rag_prompt(query: str, context: str) -> str:
    """
    Get the prompt template for Llama 3.2 models with RAG context.
    
    Args:
        query: The user's query
        context: The context information from retrieved documents
        
    Returns:
        The formatted prompt
    """
    return f"""<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context information provided below.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.

Here is some context information to help you answer the user's question:
{context}
</s>
<|user|>
{query}
</s>
<|assistant|>"""

def get_llama_3_2_chat_prompt(messages: list) -> str:
    """
    Get the prompt template for Llama 3.2 models in chat format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        The formatted prompt
    """
    prompt = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            prompt += f"<|system|>\n{content}\n</s>\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n</s>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n</s>\n"
    
    # Add the final assistant token without completion
    if not prompt.endswith("<|assistant|>\n</s>\n"):
        prompt += "<|assistant|>"
        
    return prompt 