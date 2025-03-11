import os
import sys
import json
from flask import Flask, request, jsonify
from app.models.rag_model import RAGModel

app = Flask(__name__)

# Create a dummy context
class DummyContext:
    def __init__(self):
        self.artifacts = {'app_dir': os.path.join(os.getcwd(), 'app')}

# Initialize model
model = RAGModel()
model.load_context(DummyContext())
print("Model loaded successfully")

@app.route('/invocations', methods=['POST'])
def invoke():
    """Handle model invocations"""
    try:
        # Parse input
        if not request.is_json:
            return jsonify({"error": "Input must be in JSON format"}), 400
        
        data = request.json
        
        # Extract query
        if 'dataframe_records' in data:
            if len(data['dataframe_records']) > 0 and 'query' in data['dataframe_records'][0]:
                query = data['dataframe_records'][0]['query']
            else:
                return jsonify({"error": "No query found in dataframe_records"}), 400
        elif 'inputs' in data:
            if isinstance(data['inputs'], dict) and 'query' in data['inputs']:
                query = data['inputs']['query']
            else:
                return jsonify({"error": "No query found in inputs"}), 400
        elif 'query' in data:
            query = data['query']
        else:
            # Format response according to MLflow protocol
            return jsonify({
                "error_code": "BAD_REQUEST",
                "message": "The input must be a JSON dictionary with exactly one of the input fields {'inputs', 'dataframe_records', 'dataframe_split', 'instances'}. Received dictionary with input fields: [" + ", ".join(data.keys()) + "]."
            }), 400
        
        # Process query
        result = model.predict(None, query)
        
        # Format response according to MLflow protocol
        return jsonify({"predictions": result})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port) 