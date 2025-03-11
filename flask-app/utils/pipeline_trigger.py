import subprocess
import logging
import threading
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline_async(pdf_dir, rebuild=False):
    """
    Run the pipeline asynchronously in a separate thread.
    
    Args:
        pdf_dir: Directory containing PDF files
        rebuild: Whether to rebuild the vector index
    """
    # Get project root
    project_root = Path(__file__).resolve().parent.parent.parent

    def _run_pipeline():
        try:
            logger.info(f"Starting pipeline with PDF directory: {pdf_dir}")
            
            # Build command
            cmd = [
                sys.executable,
                str(project_root / "app" / "pipeline.py"),
                "--pdf-dir", pdf_dir
            ]
            
            if rebuild:
                cmd.append("--rebuild")
            
            # Run pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Get output
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Pipeline failed with return code {process.returncode}")
                logger.error(f"Error output: {stderr}")
            else:
                logger.info("Pipeline completed successfully")
                
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
    
    # Start thread
    thread = threading.Thread(target=_run_pipeline)
    thread.daemon = True
    thread.start()
    
    return {
        'status': 'started',
        'message': 'Pipeline started in the background'
    }