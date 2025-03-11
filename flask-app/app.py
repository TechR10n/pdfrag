# Add these imports at the top
import datetime
import shutil
import os
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

# Create Flask app
app = Flask(__name__)

# Add context processor for datetime
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Define a standalone RAG model implementation
class StandaloneRAGModel:
    def __init__(self):
        logger.info("Initializing standalone RAG model")
        
    def process_query(self, query):
        """Process a query and return a response."""
        logger.info(f"Processing query: {query}")
        
        # Mock response
        return {
            'text': f"This is a response to: {query}\n\nRetrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches for natural language processing tasks. It first retrieves relevant information from a knowledge base and then uses that information to generate a response.",
            'sources': [
                {
                    'filename': 'example_doc1.pdf',
                    'chunk_text': 'Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based approaches...',
                    'rerank_score': -0.95
                },
                {
                    'filename': 'example_doc2.pdf',
                    'chunk_text': 'RAG models first retrieve documents from a corpus and then use them as additional context when generating responses...',
                    'rerank_score': -0.85
                }
            ],
            'metadata': {
                'llm': {'tokens_used': 150, 'prompt_tokens': 50, 'completion_tokens': 100},
                'search_results': 2
            }
        }

# Import configuration
# Define constants directly since we don't have the config module
SECRET_KEY = 'dev-key-for-testing'
DEBUG = True
UPLOAD_FOLDER = os.path.join(str(Path(__file__).resolve().parent.parent), 'data', 'pdfs')
logger.info(f"UPLOAD_FOLDER absolute path: {os.path.abspath(UPLOAD_FOLDER)}")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf'}
MLFLOW_HOST = os.environ.get('MLFLOW_HOST', 'mlflow')
MLFLOW_PORT = os.environ.get('MLFLOW_PORT', '5000')

# Create Flask app
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create a simple MLflow client
class SimpleMlflowClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.model = StandaloneRAGModel()
        self.initialized = True
        logger.info("SimpleMlflowClient initialized with standalone RAG model")
        
    def is_alive(self):
        """Check if the model is alive and ready to serve requests."""
        return self.initialized and self.model is not None
            
    def predict(self, question):
        """Process a question using the RAG model."""
        try:
            # Process the question
            logger.info(f"Processing question: {question}")
            result = self.model.process_query(question)
            
            # Format the response
            response = {
                "answer": result.get('text', 'No answer generated'),
                "sources": [source.get('filename', 'Unknown') for source in result.get('sources', [])[:3]],
                "confidence": 0.95,  # Mock confidence score
                "processed_at": datetime.datetime.now().isoformat(),
                "question": question
            }
            
            return response
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Create MLflow client
try:
    mlflow_client = SimpleMlflowClient(MLFLOW_HOST, MLFLOW_PORT)
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
    now = datetime.datetime.now()
    return render_template('index.html', now=now)

@app.route('/api')
def api_info():
    """Provide information about the API."""
    endpoints = [
        {
            'path': '/api/ask',
            'method': 'POST',
            'description': 'Ask a question to the RAG model',
            'parameters': {
                'question': 'The question to ask'
            },
            'example': {
                'curl': 'curl -X POST -H "Content-Type: application/json" -d \'{"question": "What is retrieval-augmented generation?"}\' http://localhost:8000/api/ask'
            }
        },
        {
            'path': '/api/health',
            'method': 'GET',
            'description': 'Check the health of the API and RAG model',
            'example': {
                'curl': 'curl http://localhost:8000/api/health'
            }
        },
        {
            'path': '/api/documents/reindex',
            'method': 'POST',
            'description': 'Reindex a specific document',
            'parameters': {
                'filename': 'The name of the file to reindex'
            },
            'example': {
                'curl': 'curl -X POST -H "Content-Type: application/json" -d \'{"filename": "document.pdf"}\' http://localhost:8000/api/documents/reindex'
            }
        },
        {
            'path': '/api/documents/reindex-all',
            'method': 'POST',
            'description': 'Reindex all documents',
            'example': {
                'curl': 'curl -X POST http://localhost:8000/api/documents/reindex-all'
            }
        }
    ]
    
    return jsonify({
        'name': 'PDFrag API',
        'version': '1.0.0',
        'description': 'API for interacting with the PDFrag RAG model',
        'endpoints': endpoints,
        'rag_model_status': mlflow_client.is_alive() if mlflow_client else False,
        'server_time': datetime.datetime.now().isoformat()
    })

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
    
    now = datetime.datetime.now()
    return render_template('upload.html', now=now)

@app.route('/documents')
def documents():
    """List uploaded documents."""
    # Get list of PDFs in upload folder
    pdfs = []
    logger.info(f"Looking for documents in {app.config['UPLOAD_FOLDER']}")
    
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        logger.info(f"Found {len(files)} files in upload folder")
        
        for filename in files:
            if allowed_file(filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_stat = os.stat(file_path)
                pdfs.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'modified': file_stat.st_mtime,
                    'indexed': True  # Assume all documents are indexed for now
                })
                logger.info(f"Added document: {filename}")
            else:
                logger.info(f"Skipped non-PDF file: {filename}")
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
    
    # Sort by modified time (newest first)
    pdfs = sorted(pdfs, key=lambda x: x['modified'], reverse=True)
    logger.info(f"Returning {len(pdfs)} documents")
    
    now = datetime.datetime.now()
    return render_template('documents.html', documents=pdfs, now=now)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    """Ask a question."""
    if request.method == 'POST':
        return redirect(url_for('ask'))
    
    now = datetime.datetime.now()
    return render_template('ask.html', now=now)

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint for asking questions."""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question parameter'}), 400
    
    question = data['question']
    
    if not mlflow_client or not mlflow_client.is_alive():
        # Try to initialize the model if it's not already initialized
        if mlflow_client and not mlflow_client.initialized:
            try:
                mlflow_client.initialize_model()
            except Exception as e:
                error_msg = f"Failed to initialize RAG model: {str(e)}"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 503
        else:
            error_msg = "RAG model is not available. Please make sure the model is properly configured."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 503
    
    try:
        # Process question
        logger.info(f"Processing question: {question}")
        response = mlflow_client.predict(question)
        
        # Add additional information to the response
        response['processed_at'] = datetime.datetime.now().isoformat()
        response['question'] = question
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({
            'error': str(e),
            'question': question,
            'processed_at': datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    mlflow_status = False
    model_info = {}
    
    if mlflow_client:
        mlflow_status = mlflow_client.is_alive()
        model_info = {
            'initialized': mlflow_client.initialized,
            'mock_mode': True  # We're using the mock mode by default
        }
    
    return jsonify({
        'status': 'ok',
        'rag_model': mlflow_status,
        'model_info': model_info,
        'server_time': datetime.datetime.now().isoformat()
    })

# Register custom Jinja2 filters
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """Convert a timestamp to a date string."""
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/download/<filename>')
def download_document(filename):
    """Download a document."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

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

@app.route('/api/documents/reindex', methods=['POST', 'GET'])
def api_reindex_document():
    """API endpoint for reindexing a document."""
    if request.method == 'GET':
        filename = request.args.get('filename')
        if not filename:
            return jsonify({'success': False, 'error': 'Missing filename parameter'}), 400
    else:  # POST
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'success': False, 'error': 'Missing filename parameter'}), 400
        filename = data['filename']
    
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        # Check if the RAG model is available
        if not mlflow_client or not mlflow_client.is_alive():
            return jsonify({'success': False, 'error': 'RAG model is not available'}), 503
        
        # In a real implementation, we would call the indexing functionality of the RAG model
        # For now, we'll just log that we're reindexing the document
        logger.info(f"Reindexing document: {filename}")
        
        # Create a temporary directory with just this file
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], '_temp_reindex')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy the file to the temp directory
        shutil.copy(file_path, os.path.join(temp_dir, filename))
        
        return jsonify({
            'success': True,
            'message': f'Document {filename} has been queued for reindexing',
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error reindexing document: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents/reindex-all', methods=['POST'])
def api_reindex_all():
    """API endpoint for reindexing all documents."""
    try:
        # Check if the RAG model is available
        if not mlflow_client or not mlflow_client.is_alive():
            return jsonify({'success': False, 'error': 'RAG model is not available'}), 503
        
        # Get all PDF files in the upload folder
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        if not pdf_files:
            return jsonify({'success': True, 'message': 'No documents to reindex', 'count': 0})
        
        # In a real implementation, we would call the batch indexing functionality of the RAG model
        # For now, we'll just log that we're reindexing all documents
        logger.info(f"Reindexing all documents: {len(pdf_files)} files")
        
        return jsonify({
            'success': True,
            'message': f'{len(pdf_files)} documents have been queued for reindexing',
            'count': len(pdf_files),
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error reindexing all documents: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=8000)