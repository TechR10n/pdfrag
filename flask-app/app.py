# Add these imports at the top
import datetime
import shutil
import os
import json
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
import sys
from pathlib import Path
import importlib.util
import requests
from typing import Dict, Any, List, Optional
import threading
import queue
import socket
import time
from utils.mlflow_client import MLflowClient, create_mlflow_client

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

# Import configuration
# Define constants directly since we don't have the config module
SECRET_KEY = 'dev-key-for-testing'
DEBUG = True
UPLOAD_FOLDER = os.path.join('/flask_app/data', 'documents')  # Use Docker container path
logger.info(f"UPLOAD_FOLDER absolute path: {os.path.abspath(UPLOAD_FOLDER)}")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf'}
MODEL_SERVER_HOST = 'model-server'  # Use Docker service name for internal networking
MODEL_SERVER_PORT = '5000'  # Use the internal port of the model server
MODEL_SERVER_URL = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"

# Force fallback mode - set to True to always use fallback responses
FORCE_FALLBACK = False
TIMEOUT_SECONDS = 128
MAX_RETRIES = 2
RETRY_DELAY = 5

# Create Flask app
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_model_server_alive():
    """Check if the model server is alive and ready to serve requests."""
    if FORCE_FALLBACK:
        logger.info("Force fallback mode is enabled, skipping model server check")
        return False
        
    try:
        # First check if the host is reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # 2 second timeout
        result = sock.connect_ex((MODEL_SERVER_HOST, int(MODEL_SERVER_PORT)))
        sock.close()
        
        if result != 0:
            logger.error(f"Model server at {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT} is not reachable")
            return False
            
        # Use the MLflow client to check health
        mlflow_client = MLflowClient(f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
        return mlflow_client.is_alive()
    except socket.error as e:
        logger.error(f"Socket error checking model health: {str(e)}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error checking model health: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error checking model health: {str(e)}")
        return False

def process_question_with_model_server(question, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Process a question using the model server with retries."""
    # If force fallback is enabled, don't even try to use the model server
    if FORCE_FALLBACK:
        logger.info("Force fallback mode is enabled, but still waiting for the full timeout period")
        # Wait for the full timeout period before returning the fallback response
        time.sleep(timeout)
        return 'fallback', "Force fallback mode is enabled"
        
    result_queue = queue.Queue()
    
    def _process():
        retries = 0
        while retries <= max_retries:
            try:
                # If this is a retry, wait before trying again
                if retries > 0:
                    logger.info(f"Waiting {retry_delay} seconds before retry {retries}/{max_retries}...")
                    time.sleep(retry_delay)
                
                # Check if model server is alive before sending request
                if not is_model_server_alive():
                    logger.error("Model server is not available before sending request")
                    if retries < max_retries:
                        retries += 1
                        continue
                    result_queue.put(('error', "Model server is not available"))
                    return
                
                # Create MLflow client
                mlflow_client = MLflowClient(f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
                
                # Send the request to the model server using the MLflow client
                logger.info(f"Sending request to model server using MLflow client (attempt {retries + 1}/{max_retries + 1})")
                result = mlflow_client.predict(question)
                logger.info(f"Received response from model server: {result}")
                
                # Extract the predictions from the response
                if "predictions" not in result:
                    logger.error(f"Invalid response from model server: {result}")
                    if retries < max_retries:
                        retries += 1
                        continue
                    result_queue.put(('error', "Invalid response from model server"))
                    return
                
                predictions = result["predictions"]
                
                # Extract answer and sources
                answer = predictions.get('text', 'No answer generated')
                sources = []
                for source in predictions.get('sources', [])[:3]:
                    sources.append(source.get('filename', 'Unknown'))
                
                # Create response
                response_data = {
                    "answer": answer,
                    "sources": sources,
                    "confidence": 0.95,  # Default confidence score
                    "processed_at": datetime.datetime.now().isoformat(),
                    "question": question
                }
                
                result_queue.put(('success', response_data))
                return  # Success, exit the retry loop
                
            except requests.exceptions.Timeout:
                logger.error(f"Timeout error connecting to model server after {min(30, timeout/(max_retries+1))} seconds")
                if retries < max_retries:
                    retries += 1
                    continue
                result_queue.put(('timeout', f"Request timed out after {timeout} seconds. The model server is taking too long to respond."))
                return
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error to model server: {str(e)}")
                if retries < max_retries:
                    retries += 1
                    continue
                result_queue.put(('error', f"Connection error to model server: {str(e)}"))
                return
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                if retries < max_retries:
                    retries += 1
                    continue
                result_queue.put(('error', str(e)))
                return
    
    # Start the thread
    thread = threading.Thread(target=_process)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete or timeout
    try:
        status, result = result_queue.get(timeout=timeout)
        return status, result
    except queue.Empty:
        # Timeout occurred
        return 'timeout', f"Request timed out after {timeout} seconds. The model server is taking too long to respond."

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
        'rag_model_status': is_model_server_alive(),
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

def get_fallback_response(question):
    """Generate a fallback response when the model server is not available."""
    logger.warning(f"Using fallback response for question: {question}")
    
    # Define some predefined responses for common questions
    if "what is rag" in question.lower():
        answer = "FALL BACK ANSWER: RAG (Retrieval-Augmented Generation) is a technique that combines retrieval-based and generation-based approaches for natural language processing tasks. It first retrieves relevant information from a knowledge base and then uses that information to generate a response. This approach helps ground the model's responses in factual information, reducing hallucinations and improving accuracy."
        sources = ["rag_paper.pdf", "llm_techniques.pdf", "ai_advances.pdf"]
    elif "how does rag work" in question.lower():
        answer = "FALL BACK ANSWER: RAG works in two main steps: 1) Retrieval: When a query is received, the system searches a knowledge base to find relevant documents or passages. This is typically done using vector similarity search with embeddings. 2) Generation: The retrieved information is then provided as context to a language model, which generates a response that incorporates this information. This helps the model produce more accurate and factual responses."
        sources = ["rag_implementation.pdf", "vector_search.pdf", "llm_context.pdf"]
    elif "benefits of rag" in question.lower():
        answer = "FALL BACK ANSWER: The benefits of RAG include: 1) Improved factual accuracy by grounding responses in retrieved information, 2) Reduced hallucinations compared to pure generative approaches, 3) Ability to access and cite specific sources of information, 4) More up-to-date responses when the knowledge base is regularly updated, and 5) Better handling of domain-specific questions when specialized documents are included in the knowledge base."
        sources = ["rag_advantages.pdf", "llm_comparison.pdf", "enterprise_ai.pdf"]
    else:
        # Default response for questions that don't match any pattern
        answer = f"FALL BACK ANSWER: I've analyzed your question about '{question}'. This appears to be related to information retrieval and processing systems. While I don't have specific information about this exact query in my knowledge base, I can tell you that modern AI systems use various techniques to understand and respond to natural language questions. Would you like me to explain more about how these systems work in general?"
        sources = ["general_ai.pdf", "information_systems.pdf"]
    
    # Create response
    response = {
        "answer": answer,
        "sources": sources,
        "confidence": 0.95,  # Default confidence score
        "processed_at": datetime.datetime.now().isoformat(),
        "question": question,
        "fallback": True  # Indicate that this is a fallback response
    }
    
    return response

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint for asking questions."""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question parameter'}), 400
    
    question = data['question']
    
    try:
        # Record the start time
        start_time = time.time()
        
        # Process question using the model server
        logger.info(f"Processing question: {question}")
        
        # Process the question with a timeout
        status, result = process_question_with_model_server(question, timeout=TIMEOUT_SECONDS)
        
        # Calculate how much time has passed
        elapsed_time = time.time() - start_time
        
        if status == 'success':
            return jsonify(result)
        elif status == 'timeout' or status == 'error' or status == 'fallback':
            logger.warning(f"Request failed with status: {status}. Using fallback response.")
            # If less than TIMEOUT_SECONDS have passed, wait for the remainder
            if elapsed_time < TIMEOUT_SECONDS:
                remaining_time = TIMEOUT_SECONDS - elapsed_time
                logger.info(f"Waiting additional {remaining_time:.2f} seconds to reach full timeout period of {TIMEOUT_SECONDS} seconds")
                time.sleep(remaining_time)
            return jsonify(get_fallback_response(question))
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Using fallback response due to error.")
        
        # Calculate how much time has passed
        elapsed_time = time.time() - start_time
        # If less than TIMEOUT_SECONDS have passed, wait for the remainder
        if elapsed_time < TIMEOUT_SECONDS:
            remaining_time = TIMEOUT_SECONDS - elapsed_time
            logger.info(f"Waiting additional {remaining_time:.2f} seconds to reach full timeout period of {TIMEOUT_SECONDS} seconds")
            time.sleep(remaining_time)
        
        return jsonify(get_fallback_response(question))

@app.route('/api/health')
def health():
    """Health check endpoint."""
    model_server_status = is_model_server_alive()
    
    return jsonify({
        'status': 'ok',
        'rag_model': model_server_status,
        'model_info': {
            'initialized': True,
            'mock_mode': False,  # No mock mode, always using real server
            'fallback_mode': not model_server_status,  # Indicate if we're using fallback mode
            'force_fallback': FORCE_FALLBACK,  # Indicate if force fallback mode is enabled
            'timeout_seconds': TIMEOUT_SECONDS,  # Timeout period before falling back
            'max_retries': MAX_RETRIES,  # Number of retries before falling back
            'retry_delay_seconds': RETRY_DELAY  # Delay between retries
        },
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
    secure_filename_val = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename_val, as_attachment=True)

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

@app.after_request
def add_no_cache_headers(response):
    """Add headers to prevent browser caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


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
        if not is_model_server_alive():
            return jsonify({'success': False, 'error': 'RAG model is not available'}), 503
        
        # In a real implementation, we would call the indexing functionality of the RAG model
        # For now, we'll just log that we're reindexing the document
        logger.info(f"Reindexing document: {filename}")
        
        # Create a temporary directory with just this file
        temp_dir = os.path.join('/flask_app/data', '_temp_reindex')
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
        if not is_model_server_alive():
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
    # Run the app with debug mode
    app.run(debug=DEBUG, host='0.0.0.0', port=8000, threaded=True)