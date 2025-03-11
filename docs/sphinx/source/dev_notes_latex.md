% Auto-generated LaTeX document
\input{preamble}

\begin{document}

\section{Developer Notes for PDF RAG System}

\subsection{Overview}

This document contains development notes, technical details, and implementation guidelines for the PDF RAG (Retrieval Augmented Generation) System. It serves as an appendix to the main documentation and is intended for developers working on the project.

\subsection{Architecture Details}

\subsubsection{System Components}

The PDF RAG System consists of the following main components:

\begin{itemize}
\item **Web Interface (Flask)**
\end{itemize}
   \item Handles user interactions
   \item Manages file uploads
   \item Displays query results
   \item Provides administrative functions

\begin{itemize}
\item **Document Processing Pipeline**
\end{itemize}
   \item PDF parsing and text extraction
   \item Chunking and preprocessing
   \item Vector embedding generation
   \item Storage in vector database

\begin{itemize}
\item **Query Processing Engine**
\end{itemize}
   \item Query understanding and preprocessing
   \item Vector similarity search
   \item Context assembly
   \item Response generation with LLM

\begin{itemize}
\item **Experiment Tracking (MLflow)**
\end{itemize}
   \item Model versioning
   \item Experiment logging
   \item Performance metrics
   \item Model registry

\subsubsection{Data Flow}

\begin{itemize}
\item **Document Ingestion Flow**
\end{itemize}
   \begin{lstlisting}
Upload PDF $\rightarrow$ Extract Text $\rightarrow$ Chunk Text $\rightarrow$ Generate Embeddings $\rightarrow$ Store in Vector DB
\end{lstlisting}

\begin{itemize}
\item **Query Processing Flow**
\end{itemize}
   \begin{lstlisting}
User Query $\rightarrow$ Preprocess Query $\rightarrow$ Generate Query Embedding $\rightarrow$ Similarity Search $\rightarrow$ 
   Retrieve Relevant Chunks $\rightarrow$ Assemble Context $\rightarrow$ Generate Response $\rightarrow$ Display to User
\end{lstlisting}

\subsection{Implementation Notes}

\subsubsection{Vector Database Selection}

We evaluated several vector databases for this project:

\begin{table}[htbp]
\centering
\begin{tabular}{|l|l|l|l|}
\hline
Database & Pros & Cons & Decision \\
\hline
Chroma & Easy integration, Python-native & Limited scaling & Selected for development \\
\hline
Pinecone & Excellent scaling, managed service & Cost, external dependency & Consider for production \\
\hline
Weaviate & Rich features, good performance & More complex setup & Alternative option \\
\hline
\end{tabular}
\end{table}
\subsubsection{Embedding Models}

The system supports multiple embedding models:

\begin{itemize}
\item OpenAI's text-embedding-ada-002
\item Sentence Transformers (all-MiniLM-L6-v2)
\item BERT-based models
\end{itemize}

Current default: \texttt{all-MiniLM-L6-v2} for local deployment

\subsubsection{LLM Integration}

The system is designed to work with various LLMs:

\begin{itemize}
\item OpenAI models (GPT-3.5, GPT-4)
\item Local models via llama.cpp
\item Hugging Face models
\end{itemize}

\subsection{Development Guidelines}

\subsubsection{Code Organization}

app/
\begin{verbatim}
├── api/                  # API endpoints
│   ├── __init__.py
│   ├── documents.py      # Document management endpoints
│   └── queries.py        # Query processing endpoints
├── models/               # Core functionality
│   ├── __init__.py
│   ├── document.py       # Document processing
│   ├── embeddings.py     # Embedding generation
│   ├── retrieval.py      # Vector retrieval
│   └── llm.py            # LLM integration
├── scripts/              # Utility scripts
│   ├── convert_md_to_pdf.py
│   └── setup_sphinx.py
├── static/               # Static assets
│   ├── css/
│   ├── js/
│   └── images/
└── templates/            # HTML templates
    ├── base.html
    ├── index.html
    └── results.html
\end{verbatim}


\subsubsection{Coding Standards}

\begin{itemize}
\item **Python Style**
\end{itemize}
   \item Follow PEP 8 guidelines
   \item Use type hints for function parameters and return values
   \item Write docstrings for all functions, classes, and modules

\begin{itemize}
\item **Testing**
\end{itemize}
   \item Write unit tests for all core functionality
   \item Aim for at least 80% code coverage
   \item Include integration tests for critical paths

\begin{itemize}
\item **Documentation**
\end{itemize}
   \item Document all public APIs
   \item Keep this dev_notes.md file updated with architectural decisions
   \item Use inline comments for complex logic

\subsection{Deployment}

\subsubsection{Local Development Environment}

\begin{itemize}
\item **Setup**
\end{itemize}
   \begin{lstlisting}[language=bash]
\# Clone the repository
   git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag

   \# Create a virtual environment
   python -m venv venv
   source venv/bin/activate  \# On Windows: venv\Scripts\activate

   \# Install dependencies
   pip install -r requirements.txt

   \# Run the development server
   flask run --debug
\end{lstlisting}

\begin{itemize}
\item **Environment Variables**
\end{itemize}
   Create a \texttt{.env} file with the following variables:
   \begin{lstlisting}
FLASK\_APP=app
   FLASK\_ENV=development
   VECTOR\_DB\_PATH=./data/vectordb
   MODEL\_CACHE\_DIR=./data/models
\end{lstlisting}

\subsubsection{Production Deployment}

\begin{itemize}
\item **Docker Deployment**
\end{itemize}
   \begin{lstlisting}[language=bash]
\# Build the Docker image
   docker build -t pdfrag:latest .

   \# Run the container
   docker run -p 5000:5000 -v ./data:/app/data pdfrag:latest
\end{lstlisting}

\begin{itemize}
\item **Cloud Deployment Options**
\end{itemize}
   \item AWS Elastic Beanstalk
   \item Google Cloud Run
   \item Azure App Service

\subsection{Performance Optimization}

\subsubsection{Memory Usage}

\begin{itemize}
\item Use streaming responses for large documents
\item Implement pagination for vector search results
\item Consider chunking strategy trade-offs (size vs. context preservation)
\end{itemize}

\subsubsection{Processing Speed}

\begin{itemize}
\item Batch processing for document ingestion
\item Caching for frequent queries
\item Asynchronous processing for non-blocking operations
\end{itemize}

\subsection{Future Development}

\subsubsection{Planned Features}

\begin{itemize}
\item [ ] Multi-user support with authentication
\item [ ] Document metadata extraction and filtering
\item [ ] Custom training of embedding models
\item [ ] API rate limiting and usage tracking
\item [ ] Advanced query capabilities (filters, time ranges)
\end{itemize}

\subsubsection{Technical Debt}

\begin{itemize}
\item Refactor document processing pipeline for better modularity
\item Improve error handling and logging
\item Enhance test coverage for edge cases
\end{itemize}

\subsection{Troubleshooting}

\subsubsection{Common Issues}

\begin{itemize}
\item **PDF Extraction Failures**
\end{itemize}
   \item Check PDF format compatibility
   \item Verify OCR settings for scanned documents

\begin{itemize}
\item **Vector Database Connection Issues**
\end{itemize}
   \item Ensure proper initialization of vector store
   \item Check persistence directory permissions

\begin{itemize}
\item **LLM Integration Problems**
\end{itemize}
   \item Verify API keys and environment variables
   \item Check model availability and compatibility

\subsection{Notes from the original README.md}

\subsection{Project Overview}

This laboratory course guides you through building a complete Retrieval-Augmented Generation (RAG) system that operates entirely on local hardware. By the end of this intensive workshop, you will have created a production-ready application that processes PDF documents, generates embeddings, performs semantic search, and answers questions using local large language modelsall accessible through a Flask web interface and MLflow serving endpoint.

\subsection{Hardware Requirements}
\begin{itemize}
\item Apple MacMini with M2 Pro chip (or equivalent)
\item 16GB RAM minimum
\item 50GB available storage
\item macOS Ventura 15.3.1 or later
\end{itemize}

\subsection{Software Prerequisites}
\begin{itemize}
\item Python 3.10+
\item Docker Desktop for Mac
\item Git
\item Homebrew (recommended for macOS)
\end{itemize}

\subsection{Getting Started}

\subsubsection{Quick Start}

The easiest way to get started is to use the provided startup script:

The startup script will:
\begin{itemize}
\item Create and activate a virtual environment
\item Install required dependencies
\item Start Docker services (vector database and MLflow)
\item Check for required models and prompt you to download them if missing
\item Process any existing PDFs in the data directory
\item Deploy the RAG model to MLflow
\item Start the Flask web application
\end{itemize}

Once everything is running, you can access the web interface at: http://localhost:8000

\subsubsection{Manual Setup}

If you prefer to set up the system manually, follow these steps:

\begin{itemize}
\item **Create and activate a virtual environment**:
\end{itemize}


\begin{itemize}
\item **Install dependencies**:
\end{itemize}


\begin{itemize}
\item **Start Docker services**:
\item **Download required models**:
\end{itemize}


\begin{itemize}
\item **Process PDFs and build the vector database**:
\end{itemize}


\begin{itemize}
\item **Deploy the model to MLflow**:
\end{itemize}


\begin{itemize}
\item **Start the Flask web application**:
\end{itemize}


\subsection{Project Structure}

\begin{itemize}
\item \texttt{app/}: Core RAG application code
\end{itemize}
  \item \texttt{config/}: Configuration settings
  \item \texttt{utils/}: Utility functions for PDF processing, embedding generation, etc.
  \item \texttt{scripts/}: Helper scripts for model download, deployment, etc.
  \item \texttt{pipeline.py}: Main pipeline for processing PDFs and building the vector database
  \item \texttt{rag\textbackslash{}_app.py}: RAG application for interactive querying

\begin{itemize}
\item \texttt{flask-app/}: Web interface
\end{itemize}
  \item \texttt{app.py}: Flask application
  \item \texttt{templates/}: HTML templates
  \item \texttt{static/}: CSS, JavaScript, and images

\begin{itemize}
\item \texttt{data/}: Data storage
\end{itemize}
  \item \texttt{pdfs/}: PDF documents
  \item \texttt{vectors/}: Vector database storage

\begin{itemize}
\item \texttt{models/}: Model storage
\end{itemize}
  \item \texttt{embedding/}: Embedding models
  \item \texttt{reranker/}: Reranker models
  \item \texttt{llm/}: Large language models

\begin{itemize}
\item \texttt{docker/}: Docker configuration files
\item \texttt{mlflow/}: MLflow artifacts and backend storage
\end{itemize}

\subsection{Using the Application}

\subsubsection{Web Interface}

\begin{itemize}
\item Open your browser and navigate to http://localhost:8000
\item Use the ''Upload'' page to upload PDF documents
\item Navigate to the ''Ask'' page to query your documents
\item View and manage your documents on the ''Documents'' page
\end{itemize}

\subsubsection{Command Line Interface}

You can also use the RAG application directly from the command line:

\begin{lstlisting}[language=bash]
\section{Interactive mode}
python app/rag\_app.py --interactive

\section{Single query mode}
python app/rag\_app.py --query ''What is the main topic of the uploaded documents?''
\end{lstlisting}

\subsubsection{Processing PDFs}

To process new PDFs or rebuild the vector database:



\subsection{Documentation}

To generate documentation for the project:


\subsection{Code Review}

After a thorough review of the codebase, here are the key findings and recommendations:

\subsubsection{Strengths}

\begin{itemize}
\item **Well-structured Architecture**: The project follows a clean separation of concerns with distinct modules for PDF processing, embedding generation, vector database management, and the web interface.
\end{itemize}

\begin{itemize}
\item **Comprehensive Documentation**: The codebase includes extensive documentation generation capabilities using Sphinx, supporting multiple output formats (HTML, PDF, Markdown).
\end{itemize}

\begin{itemize}
\item **Containerization**: The use of Docker for the vector database (Qdrant) and MLflow services ensures consistent deployment across different environments.
\end{itemize}

\begin{itemize}
\item **Local LLM Integration**: The project successfully integrates local LLM inference using llama-cpp-python with Metal acceleration for Apple Silicon, providing good performance without cloud dependencies.
\end{itemize}

\begin{itemize}
\item **Robust Error Handling**: Most scripts include proper error handling and logging, making it easier to diagnose issues.
\end{itemize}

\subsubsection{Areas for Improvement}

\begin{itemize}
\item **Environment Variable Management**: Consider using a \texttt{.env} file with python-dotenv for better configuration management instead of hardcoded values in \texttt{settings.py}.
\end{itemize}

\begin{itemize}
\item **Test Coverage**: While there are some test files, the project would benefit from more comprehensive unit and integration tests, especially for critical components like the RAG pipeline.
\end{itemize}

\begin{itemize}
\item **Security Considerations**: 
\end{itemize}
   \item The Flask secret key is hardcoded and should be changed in production
   \item No authentication is implemented for the web interface
   \item Consider adding rate limiting for API endpoints

\begin{itemize}
\item **Performance Optimizations**:
\end{itemize}
   \item The PDF processing pipeline processes all documents sequentially; consider adding parallel processing for larger document collections
   \item Implement caching for frequently accessed documents and queries

\begin{itemize}
\item **Dependency Management**: 
\end{itemize}
   \item Some dependencies in requirements.txt have fixed versions that might become outdated
   \item Consider using dependency groups (e.g., dev, prod, test) for better management

\begin{itemize}
\item **Documentation Generation**:
\end{itemize}
   \item The PDF generation process occasionally times out and could be improved
   \item Consider adding more docstrings to functions for better API documentation

\subsubsection{Recommendations}

\begin{itemize}
\item **Authentication**: Implement a simple authentication system for the web interface to protect sensitive documents.
\end{itemize}

\begin{itemize}
\item **Monitoring**: Add more comprehensive monitoring for system health, model performance, and usage statistics.
\end{itemize}

\begin{itemize}
\item **Scalability**: Consider implementing a worker queue system (like Celery) for handling larger document processing tasks asynchronously.
\end{itemize}

\begin{itemize}
\item **User Experience**: Enhance the web interface with progress indicators for long-running tasks and more detailed error messages.
\end{itemize}

\begin{itemize}
\item **Model Management**: Implement a model versioning system to track changes and allow rollbacks if needed.
\end{itemize}

\begin{itemize}
\item **Multilingual Support**: Extend the system to handle documents in multiple languages by integrating multilingual embedding models.
\end{itemize}

\begin{itemize}
\item **Automated Testing**: Set up CI/CD pipelines with automated testing to ensure code quality and prevent regressions.
\end{itemize}

\subsection{Troubleshooting}

\subsubsection{Docker Services}

If Docker services are not starting properly:

\subsubsection{Model Downloads}

If you encounter issues with model downloads:


\subsubsection{PDF Processing}

If PDF processing is failing:


\subsubsection{System Status}

To check the status of all system components:


\subsection{Backup and Restore}

To backup or restore your data:



\subsection{Offline Testing}

To test the system in an offline environment:



\subsection{Lab 1: Foundation and Infrastructure}

\subsubsection{Lab 1.1: Environment Setup and Project Structure}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Configure a proper development environment for ML applications
\item Understand containerization principles with Docker
\item Establish a maintainable project structure
\end{itemize}

\paragraph{Setup Instructions:}

\begin{itemize}
\item Open Terminal and create your project directory structure:
\end{itemize}



\begin{itemize}
\item Create and activate a virtual environment:
\end{itemize}


\begin{itemize}
\item Create a requirements file:
\end{itemize}


\begin{itemize}
\item Add the following dependencies to \texttt{requirements.txt}:
\end{itemize}

\begin{itemize}
\item Install the dependencies:
\end{itemize}



**Professor's Hint:** _When working with ML libraries on Apple Silicon, use native ARM packages where possible. The torch package specified here is compiled for M-series chips. For libraries without native ARM support, Rosetta 2 will handle the translation, but with a performance penalty._

\begin{itemize}
\item Create a Docker Compose file in the root directory:
\end{itemize}


\begin{itemize}
\item Add the following configuration to \texttt{docker-compose.yml}:
\end{itemize}

\begin{itemize}
\item Start the containers to verify the setup:
\end{itemize}


\paragraph{Checkpoint Questions:}

\begin{itemize}
\item Why are we using Docker for certain components instead of running everything natively?
\item What is the purpose of volume mapping in Docker Compose?
\item What does the \texttt{restart: unless-stopped} directive do in the Docker Compose file?
\item How does containerization affect the portability versus performance tradeoff for ML systems?
\item What security implications arise from running AI models locally versus in cloud environments?
\item How would different embedding model sizes affect the system's memory footprint and performance?
\item **Additional Challenge:** Add a PostgreSQL container to the Docker Compose file as an alternative backend for MLflow instead of SQLite.
\end{itemize}

\subsubsection{Lab 1.2: Model Preparation and Configuration}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Download and prepare ML models for offline use
\item Create configuration files for application settings
\item Understand model size and performance tradeoffs
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create a configuration file for the application:
\end{itemize}

\begin{itemize}
\item Add the following to \texttt{app/config/settings.py}:
\end{itemize}



\begin{itemize}
\item Create a script to download the embedding model:
\end{itemize}


\begin{itemize}
\item Add the following to \texttt{app/scripts/download\textbackslash{}_models.py}:
\end{itemize}

\begin{itemize}
\item Run the script to download the models:
\end{itemize}



\begin{itemize}
\item Create a script to download the LLM model (manual step due to size):
\end{itemize}

\begin{itemize}
\item Run the script to download the LLM model:
\end{itemize}



**Professor's Hint:** _The LLM download is about 4GB, so it may take some time. We're using a 4-bit quantized model (Q4_0) to optimize for memory usage on the MacMini. The quality-performance tradeoff is reasonable for most use cases. For higher quality, consider the Q5_K_M variant if your system has sufficient RAM._

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item Why do we use a configuration file instead of hardcoding values in our application?
\item What is model quantization and why is it important for local LLM deployment?
\item How does the dimension of the embedding vector (384) impact our system?
\item **Additional Challenge:** Create a script to benchmark the performance of the embedding model and LLM on your local hardware. Measure throughput (tokens/second) and memory usage.
\end{itemize}

\subsection{Lab 2: PDF Processing and Embedding Pipeline}

\subsubsection{Lab 2.1: PDF Ingestion and Text Extraction}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Implement robust PDF text extraction
\item Handle various PDF formats and structures
\item Build a scalable document processing pipeline
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create a PDF ingestion utility:
\end{itemize}



\begin{itemize}
\item Add the following to \texttt{app/utils/pdf\textbackslash{}_ingestion.py}:
\end{itemize}

\begin{itemize}
\item Create a text chunking utility:
\end{itemize}


\begin{itemize}
\item Add the following to \texttt{app/utils/text\textbackslash{}_chunking.py}:
\end{itemize}

**Professor's Hint:** _When extracting text from PDFs, remember that they're essentially containers of independent objects rather than structured documents. Many PDFs, especially scientific papers with multiple columns, can be challenging to extract in reading order. Consider extracting ''blocks'' (as shown in the code) for a balance between structure preservation and accuracy._

\begin{itemize}
\item Create a test script for PDF processing:
\end{itemize}


\begin{itemize}
\item Add the following to \texttt{app/tests/test\textbackslash{}_pdf\textbackslash{}_processing.py}:
\end{itemize}

\begin{itemize}
\item Run the test script:
\end{itemize}



\paragraph{Checkpoint Questions:}

\begin{itemize}
\item What are some challenges with extracting text from PDFs?
\item Why do we use chunking with overlap instead of simply splitting the text at fixed intervals?
\item How would the choice of \texttt{chunk\textbackslash{}_size} and \texttt{chunk\textbackslash{}_overlap} impact the RAG system?
\item How do different PDF extraction techniques handle multi-column layouts, tables, and graphics?
\item What architectural changes would be needed to handle non-PDF documents like Word, PowerPoint, or HTML?
\item How might language-specific considerations affect the text extraction and chunking process for multilingual documents?
\item **Additional Challenge:** Enhance the PDF extraction to handle tables and preserve their structure using PyMuPDF's table extraction capabilities.
\end{itemize}

\subsubsection{Lab 2.2: Vector Database Setup and Embedding Generation}

\paragraph{Learning Objectives:}
\begin{itemize}
\item Implement vector embedding generation
\item Set up a vector database for semantic search
\item Design an efficient document storage system
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create an embedding utility:
\end{itemize}

\begin{itemize}
\item Add the following to \texttt{app/utils/embedding\textbackslash{}_generation.py}:
\end{itemize}

\begin{itemize}
\item Create a vector database client:
\end{itemize}



\begin{itemize}
\item Add the following to \texttt{app/utils/vector\textbackslash{}_db.py}:
\end{itemize}

**Professor's Hint:** _Vector database performance is critical for RAG applications. Qdrant offers excellent performance with minimal resource usage, making it suitable for our MacMini deployment. The cosine distance metric is used because our embeddings are normalized, making it equivalent to dot product but slightly more intuitivea score of 1.0 means perfect similarity._

\begin{itemize}
\item Create a pipeline script to orchestrate the entire process:
\item Add the following to \texttt{app/pipeline.py}:
\end{itemize}

\begin{itemize}
\item Create a test script for the vector database:
\end{itemize}


\begin{itemize}
\item Add the following to \texttt{app/tests/test\textbackslash{}_vector\textbackslash{}_db.py}:
\end{itemize}

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item Why do we normalize embeddings before storing them in the vector database?
\item What are the tradeoffs between different distance metrics (cosine, Euclidean, dot product)?
\item How does batch processing improve performance when generating embeddings or uploading to the vector database?
\item How would you modify the embedding strategy if you needed to handle documents in multiple languages?
\item How does the choice of vector dimension affect the semantic richness versus computational efficiency tradeoff?
\item What information might be lost during the chunking process, and how could this affect retrieval quality?
\item **Additional Challenge:** Implement a method to incrementally update the vector database when new PDFs are added, without reprocessing the entire corpus.
\end{itemize}

\subsection{Lab 3: Query Processing and RAG Implementation}

\subsubsection{Lab 3.1: Vector Search and Re-ranking}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Implement effective query processing
\item Build a re-ranking system for search results
\item Optimize search relevance using modern techniques
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create a query processing utility:
\end{itemize}



\begin{itemize}
\item Add the following to \texttt{app/utils/query\textbackslash{}_processing.py}:
\end{itemize}

\begin{itemize}
\item Create a re-ranking utility:
\end{itemize}


\begin{itemize}
\item Add the following to \texttt{app/utils/reranking.py}:
\end{itemize}
**Professor's Hint:** _Re-ranking is one of the most effective yet underutilized techniques in RAG systems. While vector similarity gives us a ''ballpark'' match, cross-encoders consider the query and document together to produce much more accurate relevance scores. The computational cost is higher, but since we only re-rank a small number of results, the overall impact is minimal._

\begin{itemize}
\item Create a search pipeline that combines vector search and re-ranking:
\end{itemize}


\begin{itemize}
\item Add the following to \texttt{app/utils/search.py}:
\end{itemize}

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item How does the two-stage retrieval process (vector search  re-ranking) improve result quality?
\item What are the performance implications of increasing the number of results to re-rank?
\item Why is it important to normalize query embeddings in the same way as document embeddings?
\item How might different similarity thresholds affect recall versus precision in retrieval results?
\item What architectural changes would be needed if you wanted to incorporate hybrid search (combining vector similarity with keyword matching)?
\item How would the search pipeline need to be modified to support multi-modal queries (such as image + text)?
\item **Additional Challenge:** Implement a hybrid search that combines vector similarity with BM25 keyword matching for improved retrieval of rare terms and phrases.
\end{itemize}

\subsubsection{Lab 3.2: LLM Integration and Response Generation}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Integrate a local LLM for answer generation
\item Design effective prompts for RAG systems
\item Implement context augmentation techniques
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create an LLM utility:
\end{itemize}



\begin{itemize}
\item Add the following to \texttt{app/utils/llm.py}:
\end{itemize}

**Professor's Hint:** _Prompt engineering is critical for effective RAG systems. The prompt should guide the LLM to focus on the provided context and avoid hallucination. Setting a lower temperature (e.g., 0.2) helps produce more factual, deterministic responses based on the context._

\begin{itemize}
\item Create a complete RAG application:
\item Add the following to \texttt{app/rag\textbackslash{}_app.py}:
\end{itemize}

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item How does the prompt structure influence the quality of LLM responses in a RAG system?
\item What LLM parameters (temperature, top_p, etc.) are most important for RAG applications and why?
\item How do we handle the case where no relevant documents are found in the search phase?
\item What are the ethical considerations when an LLM generates answers that contradict the retrieved context?
\item How would you modify the system to support providing citations or references in the generated responses?
\item What techniques could you implement to reduce hallucinations in the generated answers?
\item How does the choice of context window size affect the tradeoff between comprehensive context and response quality?
\item **Additional Challenge:** Implement a document citation mechanism that links specific parts of the response to the source documents, allowing users to verify information.
\end{itemize}

\subsection{Lab 4: MLflow Integration and Model Serving}

\subsubsection{Lab 4.1: MLflow Model Wrapper and Logging}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Create an MLflow model wrapper for the RAG system
\item Log models and artifacts to MLflow
\item Understand model versioning and management
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create an MLflow model wrapper:
\end{itemize}

\begin{itemize}
\item Add the following to \texttt{app/models/rag\textbackslash{}_model.py}:
\end{itemize}

\begin{itemize}
\item Create a script to log the model to MLflow:
\end{itemize}



\begin{itemize}
\item Add the following to \texttt{app/scripts/log\textbackslash{}_model.py}:
\end{itemize}

**Professor's Hint:** _MLflow's Python Function (PyFunc) model format is very flexible but doesn't handle complex dependencies well. By including the \texttt{app\textbackslash{}_dir} as an artifact, we ensure that all necessary code is available, but this approach requires the filesystem structure to be preserved. For production, consider packaging your RAG components as a Python package._

\begin{itemize}
\item Create a script to deploy the model:
\end{itemize}


\begin{itemize}
\item Add the following to \texttt{app/scripts/deploy\textbackslash{}_model.py}:
\end{itemize}

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item Why do we use MLflow for model versioning and deployment?
\item What are the advantages of packaging a model with MLflow compared to directly running the application?
\item How does the MLflow model server handle inference requests?
\item How would you design an A/B testing framework to evaluate changes to different components of the RAG pipeline?
\item What monitoring metrics would be most important for understanding RAG system performance in production?
\item How does the choice of serialization format for model artifacts affect the tradeoff between load time and storage efficiency?
\item What strategies could you implement to handle model updates without service interruption?
\item **Additional Challenge:** Create a system for A/B testing different model configurations by deploying multiple versions of the model and comparing their performance.
\end{itemize}

\subsubsection{Lab 4.2: MLflow Client and Integration Testing}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Create a client to interact with the MLflow serving endpoint
\item Implement integration testing for the end-to-end system
\item Test system performance and reliability
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create an MLflow client utility:
\item Add the following to \texttt{app/clients/mlflow\textbackslash{}_client.py}:
\item Create an integration test script:
\item Add the following to \texttt{app/tests/test\textbackslash{}_integration.py}:
\end{itemize}

**Professor's Hint:** _Integration tests are essential for ensuring the reliability of complex systems like this RAG application. The tests should cover both ''happy path'' scenarios and edge cases. Pay special attention to response timingif the system is too slow, users will find it frustrating regardless of accuracy._

\begin{itemize}
\item Create a load testing script (optional):
\item Add the following to \texttt{app/tests/load\textbackslash{}_test.py}:
\end{itemize}

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item What are the key metrics to monitor in a RAG system under load?
\item How does concurrent access impact the performance of different components (vector DB, embedding model, LLM)?
\item What options exist for scaling the system if it becomes performance-limited?
\item **Additional Challenge:** Create a benchmark suite that evaluates the system's accuracy using a set of predefined questions with known answers.
\end{itemize}

\subsubsection{Lab 4.3: Creating an MLflow Client for the Flask App}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Create a client to interact with the MLflow serving endpoint from Flask
\item Implement error handling and logging for MLflow interactions
\item Design a clean API abstraction for your web application
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create the MLflow client for the Flask app:
\end{itemize}

\begin{itemize}
\item Create a utility for pipeline triggering:
\item Add the following code to \texttt{flask-app/utils/pipeline\textbackslash{}_trigger.py}:
\end{itemize}

**Professor's Hint:** _When triggering long-running processes from a web application, it's best to run them asynchronously to avoid blocking the web server. This keeps the user interface responsive while the processing happens in the background._

\subsection{Lab 5: Flask Web Interface}

\subsubsection{Lab 5.1: Basic Flask Application Setup}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Set up a Flask web application
\item Create a user interface for the RAG system
\item Implement file upload and document management
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create a basic Flask application structure:
\item Add the following to \texttt{config.py}:
\item Add the following to \texttt{app.py}:
\end{itemize}

\subsubsection{Lab 5.2: HTML Templates and Static Files}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Create responsive HTML templates using Bootstrap
\item Implement client-side functionality with JavaScript
\item Design an intuitive user interface for RAG interactions
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create the base template:
\item Add the following to \texttt{flask-app/templates/base.html}:
\item Create the main CSS file:
\item Add the following to \texttt{flask-app/static/css/style.css}:
\item Create the index page:
\item Add the following to \texttt{flask-app/templates/index.html}:
\item Create the upload page:
\item Add the following to \texttt{flask-app/templates/upload.html}:
\item Create the documents page:
\item Add the following to \texttt{flask-app/templates/documents.html}:
\item Create the ask page:
\item Add the following to \texttt{flask-app/templates/ask.html}:
\end{itemize}

**Professor's Hint:** _Always implement robust error handling in your web interfaces, especially for operations that involve AI processing which can sometimes be unpredictable. The spinner provides important feedback to users during potentially lengthy operations, while the clear display of source documents helps users understand and trust the generated answers._

\subsubsection{Lab 5.3: Complete the Flask Application}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Implement the remaining API endpoints for the Flask application
\item Add custom filters and utility functions
\item Test the complete web interface
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Add the following additional API endpoints to \texttt{app.py}:
\item Create a modified version of the upload route to handle processing:
\item Create a startup script to launch the entire system:
\item Add the following to \texttt{startup.sh}:
\item Create a Dockerfile for the Flask app:
\item Add the following to \texttt{flask-app/Dockerfile}:
\item Update the Docker Compose file to include the Flask app:
\end{itemize}

**Professor's Hint:** _The startup script is crucial for ensuring all components start in the correct order and with proper configuration. By handling dependency checks and proper error reporting, it makes the system much more robust and user-friendly._

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item How do Flask templates and static files work together to create a responsive user interface?
\item What are the advantages of using AJAX for form submissions in a web application?
\item How can we ensure the system continues running even when disconnected from the internet?
\item What UX considerations are specific to RAG interfaces compared to traditional search interfaces?
\item How would you implement progressive loading to improve perceived performance for long-running queries?
\item What accessibility considerations should be taken into account for a RAG interface?
\item How might you design the interface to help users formulate better queries and get more accurate results?
\item **Additional Challenge:** Enhance the web interface with a history of past questions and answers, allowing users to revisit their previous queries.
\end{itemize}

\subsection{Lab 6: Final Integration and Testing}

\subsubsection{Lab 6.1: End-to-End System Integration}

\paragraph{Learning Objectives:}
\begin{itemize}
\item Integrate all components into a cohesive system
\item Test the complete RAG pipeline
\item Troubleshoot common integration issues
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create an end-to-end integration test:
\item Add the following to \texttt{app/tests/test\textbackslash{}_end\textbackslash{}_to\textbackslash{}_end.py}:
\item Create a sample PDF for testing:
\item Copy a small PDF to this directory for testing purposes.
\item Create a system status check script:
\item Add the following to \texttt{system\textbackslash{}_status.py}:
\end{itemize}

**Professor's Hint:** _Integration testing is critical for complex systems with multiple components. The system status check script helps users quickly diagnose problems and restart specific components if needed. Make this script easily accessible and user-friendly to encourage its use._

\paragraph{Consistency Testing}

To ensure the system components work together consistently, let's add a few integration tests that specifically check for compatibility between components:

\begin{itemize}
\item Create a component integration test:
\item Add the following to \texttt{app/tests/test\textbackslash{}_component\textbackslash{}_integration.py}:
\end{itemize}

This test specifically focuses on ensuring that the dimensions, data formats, and interfaces between components are consistent, which is critical for system integration.

\subsubsection{Lab 6.3: System Consistency Verification}

\subsubsection{Lab 6.2: Offline Mode Testing}

\paragraph{Learning Objectives:}

\begin{itemize}
\item Test the system with internet connectivity disabled
\item Identify and fix dependencies on external services
\item Ensure data persistence across system restarts
\end{itemize}

\paragraph{Exercises:}

\begin{itemize}
\item Create an offline mode test script:
\end{itemize}

\begin{itemize}
\item Add the following to \texttt{offline\textbackslash{}_test.sh}:
\end{itemize}

\begin{itemize}
\item Create a backup and restore script:
\end{itemize}

\begin{itemize}
\item Add the following to \texttt{backup\textbackslash{}_restore.sh}:
\end{itemize}

**Professor's Hint:** _Regular backups are essential for any system that stores important data. The backup script not only creates archives of your data and models but also ensures data consistency by stopping services before backing up and restarting them afterward._

\paragraph{Checkpoint Questions:}

\begin{itemize}
\item What happens if one of the Docker containers fails during system operation?
\item How can we make the system resilient to temporary failures?
\item Why is testing in offline mode important for a locally deployed RAG system?
\item What failure modes are unique to RAG systems compared to traditional search or pure LLM applications?
\item How would you implement graceful degradation if one component of the system fails?
\item What would be the most critical components to monitor in a production RAG system?
\item How could you design the system architecture to allow for horizontal scaling if processing demands increase?
\item What security considerations should be addressed for a system that processes potentially sensitive documents?
\item **Additional Challenge:** Implement a cron job to automatically create daily backups and retain only the 7 most recent backups.
\end{itemize}

\subsection{Final Project Submission Guidelines}

\subsubsection{Requirements}

\begin{itemize}
\item **Complete System**
\end{itemize}
   \item All components integrated and working together
   \item Fully functional in offline mode
   \item Clear documentation for usage

\begin{itemize}
\item **Technical Report (3-5 pages)**
\end{itemize}
   \item System architecture overview
   \item Component descriptions
   \item Performance analysis
   \item Limitations and future improvements

\begin{itemize}
\item **Code Repository**
\end{itemize}
   \item Well-organized codebase
   \item Proper comments and docstrings
   \item README with setup and usage instructions
   \item Requirements file with all dependencies

\begin{itemize}
\item **Demonstration Video (5-7 minutes)**
\end{itemize}
   \item System walkthrough
   \item Example usage scenarios
   \item Performance demonstration

\subsubsection{Self-Assessment Questions}

Before submitting your project, ask yourself:

\begin{itemize}
\item Can the system function completely offline after initial setup?
\item Does the RAG system correctly extract text from various PDF formats?
\item Is the vector search accurate and relevant to user queries?
\item Does the LLM generate coherent and factual responses based on the retrieved context?
\item Is the web interface intuitive and responsive?
\item Does the system handle errors gracefully?
\item Is there proper documentation for all components?
\item Can another person set up and use your system based on your documentation?
\end{itemize}

\subsection{Conclusion}

Building a local RAG system requires integration of multiple complex components, from PDF processing and vector embeddings to large language models and web interfaces. This project has given you hands-on experience with the entire pipeline, providing a foundation for developing more advanced AI applications.

The skills you've developed can be applied to many other domains, including:
\begin{itemize}
\item Custom knowledge bases for specific domains
\item Intelligent document search systems
\item Automated document analysis
\item Personalized AI assistants
\end{itemize}

Remember that local AI systems have unique advantages in terms of privacy, data control, and offline functionality. As the field continues to evolve, the ability to deploy AI systems locally will become increasingly valuable for many applications.

Good luck with your final project!

**Professor's Final Hint:** _The true test of any system is how it performs with real-world data and users. Take time to test your system with diverse PDFs and questions, and iterate based on what you learn. The most valuable systems are those that solve real problems elegantly and reliably._

\subsection{References}

\begin{itemize}
\item \href{https://flask.palletsprojects.com/}{Flask Documentation}
\item \href{https://python.langchain.com/docs/}{LangChain Documentation}
\item \href{https://docs.trychroma.com/}{Chroma Vector DB}
\item \href{https://mlflow.org/docs/latest/index.html}{MLflow Documentation}
\end{itemize}

---

*This document is a living resource and should be updated as the project evolves.* 

\end{document}
