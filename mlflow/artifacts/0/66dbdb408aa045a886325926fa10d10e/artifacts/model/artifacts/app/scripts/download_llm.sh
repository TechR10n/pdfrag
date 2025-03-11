#!/bin/bash

# Directory for LLM model
LLM_DIR="models/llm"
mkdir -p $LLM_DIR

# Prompt for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
  echo "Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):"
  read -s HF_TOKEN
  echo
fi

# Download using huggingface-cli
echo "Installing huggingface_hub if needed..."
pip install -q huggingface_hub

echo "Downloading Llama-3.2-3B-Instruct model..."
echo "This will take some time depending on your connection."

python -c "
from huggingface_hub import snapshot_download
import os

# Set token
os.environ['HF_TOKEN'] = '$HF_TOKEN'

# Download model files
model_path = snapshot_download(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    local_dir='$LLM_DIR/Llama-3.2-3B-Instruct',
    local_dir_use_symlinks=False
)

print(f'Model downloaded to {model_path}')
"

# Update settings.py to use this model
SETTINGS_PATH="app/config/settings.py"
if [ -f "$SETTINGS_PATH" ]; then
    if grep -q "LLM_MODEL_PATH" "$SETTINGS_PATH"; then
        sed -i '' 's|LLM_MODEL_PATH = .*|LLM_MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "Llama-3.2-3B-Instruct")|' "$SETTINGS_PATH"
        echo "Updated settings.py to use the Llama-3.2-3B-Instruct model."
    fi
fi

echo "Now installing transformers to use with Meta's model format..."
pip install -q transformers accelerate

# Create a model loader adapter
mkdir -p app/utils/adapters
cat > app/utils/adapters/meta_llama_adapter.py << 'EOF'
import os
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class MetaLlamaAdapter:
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        """Initialize the Meta Llama adapter."""
        logger.info(f"Loading Meta Llama model from {model_path}")
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        logger.info("Meta Llama model loaded successfully")
    
    def __call__(self, prompt: str, **kwargs):
        """Generate text using the Meta Llama model."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.2) > 0,
        }
        
        # Generate text
        outputs = self.pipe(
            prompt,
            **generation_kwargs
        )
        
        # Format to match llama-cpp-python output format
        generated_text = outputs[0]["generated_text"][len(prompt):]
        
        # Count tokens
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(generated_text))
        
        return {
            "choices": [
                {
                    "text": generated_text,
                    "finish_reason": "length" if output_tokens >= generation_kwargs["max_new_tokens"] else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
EOF

# Update llm.py to use the Meta Llama adapter
sed -i '' 's/from llama_cpp import Llama/from llama_cpp import Llama\nfrom app.utils.adapters.meta_llama_adapter import MetaLlamaAdapter/' app/utils/llm.py

# Update the LLMProcessor.__init__ method
sed -i '' 's/self.model = Llama(/# Check if using Meta Llama model\n        if "Llama-3" in model_path and os.path.isdir(model_path):\n            self.model = MetaLlamaAdapter(\n                model_path=model_path,\n                max_new_tokens=max_tokens\n            )\n        else:\n            # Use llama-cpp for GGUF models\n            self.model = Llama(/' app/utils/llm.py

echo "Setup complete for using meta-llama/Llama-3.2-3B-Instruct"