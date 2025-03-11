# Add these imports at the top
import datetime
import shutil
import os
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import configuration
from flask_app.config import SECRET_KEY, DEBUG, UPLOAD_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS
from flask_app.utils.mlflow_client import create_mlflow_client

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create MLflow client
try:
    from flask_app.config import MLFLOW_HOST, MLFLOW_PORT
    mlflow_client = create_mlflow_client(MLFLOW_HOST, MLFLOW_PORT)
    logger.info(f"MLflow client created for endpoint: http://{MLFLOW_HOST}:{MLFLOW_PORT}")
except Exception as e:
    logger.error(f"Failed to create MLflow client: {str(e)}")
    mlflow_client = None

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file uploads."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash(f'File {filename} uploaded successfully!', 'success')
            
            # Redirect to document list
            return redirect(url_for('documents'))
        else:
            flash('File type not allowed', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/documents')
def documents():
    """List uploaded documents."""
    # Get list of PDFs in upload folder
    pdfs = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_stat = os.stat(file_path)
            pdfs.append({
                'filename': filename,
                'size': file_stat.st_size,
                'modified': file_stat.st_mtime
            })
    
    # Sort by modified time (newest first)
    pdfs = sorted(pdfs, key=lambda x: x['modified'], reverse=True)
    
    return render_template('documents.html', documents=pdfs)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    """Ask a question."""
    if request.method == 'POST':
        return redirect(url_for('ask'))
    
    return render_template('ask.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint for asking questions."""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question parameter'}), 400
    
    question = data['question']
    
    if not mlflow_client or not mlflow_client.is_alive():
        error_msg = "MLflow endpoint is not available. Please make sure the model is deployed."
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 503
    
    try:
        # Process question
        logger.info(f"Processing question: {question}")
        response = mlflow_client.predict(question)
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    mlflow_status = mlflow_client.is_alive() if mlflow_client else False
    
    return jsonify({
        'status': 'ok',
        'mlflow': mlflow_status
    })

# Register custom Jinja2 filters
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """Convert a timestamp to a formatted date string."""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M')

@app.route('/api/documents/delete', methods=['POST'])
def api_delete_document():
    """API endpoint for deleting a document."""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'success': False, 'error': 'Missing filename parameter'}), 400
    
    filename = secure_filename(data['filename'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        # Delete the file
        os.remove(file_path)
        
        # Note: In a production system, we would also want to remove the document from the vector database
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents/reindex', methods=['POST'])
def api_reindex_document():
    """API endpoint for reindexing a document."""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'success': False, 'error': 'Missing filename parameter'}), 400
    
    filename = secure_filename(data['filename'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        # Import pipeline trigger module
        from flask_app.utils.pipeline_trigger import run_pipeline_async
        
        # Create a temporary directory with just this file
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], '_temp_reindex')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy the file to the temp directory
        shutil.copy(file_path, os.path.join(temp_dir, filename))
        
        # Run the pipeline on the temp directory
        result = run_pipeline_async(temp_dir, rebuild=True)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error reindexing document: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents/reindex-all', methods=['POST'])
def api_reindex_all():
    """API endpoint for reindexing all documents."""
    try:
        # Import pipeline trigger module
        from flask_app.utils.pipeline_trigger import run_pipeline_async
        
        # Run the pipeline on the upload folder
        result = run_pipeline_async(app.config['UPLOAD_FOLDER'], rebuild=True)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error reindexing all documents: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=8000)