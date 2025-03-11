## Running the RAG Model

There are several ways to run the RAG model:

### 1. Direct Execution

The simplest way to run the model is directly using the `run_model.py` script:

```bash
python run_model.py
```

This will start an interactive session where you can enter queries and get responses.

### 2. Flask Server

You can run the model as a Flask server using the `serve_model.py` script:

```bash
python serve_model.py
```

This will start a server on port 5003 (by default) that accepts POST requests to `/invocations` with the following formats:

```bash
# Using dataframe_records format
curl -X POST -H "Content-Type: application/json" -d '{"dataframe_records": [{"query": "Your query here"}]}' http://localhost:5003/invocations

# Using inputs format
curl -X POST -H "Content-Type: application/json" -d '{"inputs": {"query": "Your query here"}}' http://localhost:5003/invocations
```

### 3. MLflow Model Serving

You can also serve the model using MLflow:

```bash
# Register the model
python register_model.py

# Serve the model
mlflow models serve -m models:/rag_model/latest -p 5003 --no-conda
```

Note: When using MLflow to serve the model, make sure no other processes are using the specified port.

### Troubleshooting

If you encounter issues with the MLflow model serving:

1. Check if the port is already in use:
   ```bash
   lsof -i :<port_number>
   ```

2. Kill any processes using the port:
   ```bash
   lsof -i :<port_number> | awk '{print $2}' | grep -v PID | xargs kill -9
   ```

3. If the model fails to load, check the logs for errors. Common issues include:
   - Missing model files
   - Import errors
   - Memory issues when loading large models

4. For development and testing, set `USE_MOCK = True` in `app/models/rag_model.py` to use a mock implementation that doesn't require downloading large models. 