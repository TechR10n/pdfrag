import os
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_huggingface_model(
    model_id: str, 
    hf_token: str, 
    output_path: str,
    quantize: bool = True
) -> Optional[str]:
    """
    Download a model from Hugging Face and optionally quantize it to GGUF format.
    
    Args:
        model_id: Hugging Face model ID (e.g., meta-llama/Llama-3.2-3B-Instruct)
        hf_token: Hugging Face API token
        output_path: Path to save the model
        quantize: Whether to quantize the model to GGUF format
        
    Returns:
        Path to the downloaded model or None if download failed
    """
    if not hf_token:
        logger.error("No Hugging Face token provided")
        return None
        
    # Create download directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If the output path exists and it's either empty or invalid, remove it
    if os.path.exists(output_path):
        if os.path.getsize(output_path) < 1000000:  # Less than 1MB is probably not valid
            logger.info(f"Removing invalid or empty model file at {output_path}")
            os.remove(output_path)
        else:
            logger.info(f"Valid model file already exists at {output_path}")
            return output_path
    
    # Create a temporary directory for the model
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Downloading model {model_id} to temporary directory")
        
        # Set git credentials
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"  # Skip LFS files initially for faster clone
        os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Disable git prompts
        
        # Clone the repo
        clone_cmd = [
            "git", "clone", 
            f"https://USER:{hf_token}@huggingface.co/{model_id}", 
            temp_dir
        ]
        
        logger.info("Running git clone command")
        
        try:
            subprocess.run(
                clone_cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # Pull LFS files
            lfs_cmd = ["git", "lfs", "pull"]
            subprocess.run(
                lfs_cmd, 
                check=True, 
                cwd=temp_dir, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            logger.info("Model download complete")
            
            # Quantize if requested
            if quantize:
                logger.info("Quantizing model to GGUF format")
                
                # Check for llama.cpp or llamafile-quantize
                quantize_tool = None
                
                # Check for llamafile-quantize
                try:
                    subprocess.run(
                        ["llamafile-quantize", "--help"], 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    quantize_tool = "llamafile-quantize"
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Check for llama.cpp
                    try:
                        llamacpp_path = os.environ.get("LLAMA_CPP_PATH")
                        if llamacpp_path and os.path.exists(os.path.join(llamacpp_path, "convert.py")):
                            quantize_tool = "llama.cpp"
                    except Exception:
                        pass
                
                if quantize_tool == "llamafile-quantize":
                    # Use llamafile-quantize
                    quantize_cmd = [
                        "llamafile-quantize", 
                        f"{temp_dir}", 
                        "--outfile", output_path,
                        "--q4_0"  # Use Q4_0 quantization for balance of quality and size
                    ]
                    subprocess.run(
                        quantize_cmd, 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    return output_path
                elif quantize_tool == "llama.cpp":
                    # Use llama.cpp
                    convert_script = os.path.join(os.environ["LLAMA_CPP_PATH"], "convert.py")
                    convert_cmd = [
                        "python", convert_script,
                        f"{temp_dir}", 
                        "--outfile", output_path,
                        "--outtype", "q4_0"
                    ]
                    subprocess.run(
                        convert_cmd, 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    return output_path
                else:
                    logger.warning("No quantization tools found, preparing model as-is")
                    
                    # Create placeholder file with the right path
                    try:
                        # Find the main model file (usually model.safetensors or pytorch_model.bin)
                        model_files = []
                        for file in os.listdir(temp_dir):
                            if file.endswith(".safetensors") or file.endswith(".bin"):
                                model_files.append(os.path.join(temp_dir, file))
                        
                        if not model_files:
                            logger.error("No model files found in downloaded repository")
                            return None
                            
                        # Create model directory if it doesn't exist
                        model_dir = os.path.dirname(output_path)
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Create a placeholder file to identify this as a model
                        with open(output_path, 'w') as f:
                            # Add a header to identify this as a non-GGUF file
                            f.write("# This is a placeholder for a GGUF model. The original model is available at:\n")
                            f.write(f"# https://huggingface.co/{model_id}\n")
                            f.write("# Please convert to GGUF format using llama.cpp or llamafile-quantize\n")
                        
                        # Copy the tokenizer and other relevant files to the model directory
                        for file in os.listdir(temp_dir):
                            if file.endswith(".json") or file.endswith(".py") or file.endswith(".md"):
                                src = os.path.join(temp_dir, file)
                                dst = os.path.join(model_dir, file)
                                shutil.copy2(src, dst)
                                
                        logger.info(f"Created placeholder model file at {output_path}")
                        return output_path
                    except Exception as e:
                        logger.error(f"Error preparing model: {e}")
                        return None
            else:
                # No quantization, just copy the model files
                try:
                    # Copy all files to output directory
                    output_dir = os.path.dirname(output_path)
                    for item in os.listdir(temp_dir):
                        src = os.path.join(temp_dir, item)
                        dst = os.path.join(output_dir, item)
                        
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                            
                    # Create a symlink or copy of the main model file with the expected name
                    main_model_file = None
                    for file in os.listdir(output_dir):
                        if file.endswith(".safetensors") or file.endswith(".bin"):
                            main_model_file = os.path.join(output_dir, file)
                            break
                            
                    if main_model_file:
                        try:
                            os.symlink(main_model_file, output_path)
                        except (OSError, AttributeError):
                            shutil.copy2(main_model_file, output_path)
                            
                    return output_path
                except Exception as e:
                    logger.error(f"Error copying model files: {e}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None
    
    logger.error("Failed to download the model")
    return None

def find_or_download_model(
    primary_path: str, 
    alt_paths: List[str], 
    model_id: str, 
    hf_token: str
) -> str:
    """
    Find a model in specified paths or download it if not found
    
    Args:
        primary_path: Primary path to look for the model
        alt_paths: Alternative paths to look for the model
        model_id: Hugging Face model ID
        hf_token: Hugging Face API token
        
    Returns:
        Path to the model (either found or downloaded)
    """
    # Check primary path
    if os.path.exists(primary_path) and os.path.getsize(primary_path) > 1000000:  # >1MB is probably valid
        logger.info(f"Found model at primary path: {primary_path}")
        return primary_path
        
    # Check alternative paths
    for path in alt_paths:
        if os.path.exists(path) and os.path.getsize(path) > 1000000:  # >1MB is probably valid
            logger.info(f"Found model at alternative path: {path}")
            return path
            
    # Download if not found and we have necessary info
    if model_id and hf_token:
        logger.info("Model not found in any of the specified paths, downloading from Hugging Face")
        downloaded_path = download_huggingface_model(model_id, hf_token, primary_path)
        if downloaded_path:
            logger.info(f"Successfully downloaded model to {downloaded_path}")
            return downloaded_path
            
    # Return primary path as fallback (even if it doesn't exist or isn't valid)
    return primary_path 