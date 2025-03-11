import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_documents_route():
    """Test the documents route logic."""
    # Define the upload folder
    upload_folder = os.path.join(str(Path(__file__).resolve().parent.parent), 'data', 'documents')
    logger.info(f"UPLOAD_FOLDER: {upload_folder}")
    logger.info(f"Directory exists: {os.path.exists(upload_folder)}")
    
    # Check if the directory exists
    if not os.path.exists(upload_folder):
        logger.error(f"Directory does not exist: {upload_folder}")
        return
    
    # List files in the directory
    try:
        files = os.listdir(upload_folder)
        logger.info(f"Found {len(files)} files in upload folder")
        
        # Define allowed extensions
        allowed_extensions = {'pdf'}
        
        # Check each file
        pdfs = []
        for filename in files:
            logger.info(f"Checking file: {filename}")
            
            # Check if it's a PDF
            is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
            logger.info(f"Is allowed: {is_allowed}")
            
            if is_allowed:
                file_path = os.path.join(upload_folder, filename)
                file_stat = os.stat(file_path)
                pdfs.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'modified': file_stat.st_mtime,
                })
                logger.info(f"Added document: {filename}")
            else:
                logger.info(f"Skipped non-PDF file: {filename}")
        
        # Print results
        logger.info(f"Found {len(pdfs)} PDF files")
        for pdf in pdfs:
            logger.info(f"PDF: {pdf['filename']}, Size: {pdf['size']} bytes")
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    test_documents_route() 