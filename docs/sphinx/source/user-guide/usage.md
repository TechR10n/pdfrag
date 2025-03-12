# Usage Guide

This section provides a guide on how to use the PDF RAG System.

## Uploading Documents

1. Navigate to the Upload page
2. Select PDF files from your computer
3. Click the Upload button
4. Wait for the processing to complete

## PDF Ingestion Process

When you upload PDF documents, the system performs the following steps:

1. **Scanning**: The system scans the uploaded PDFs and collects metadata such as file size, page count, and last modified time.

2. **Text Extraction**: The system extracts text from each PDF using PyMuPDF, preserving the document structure as much as possible.

3. **Text Chunking**: The extracted text is split into smaller chunks for efficient processing and retrieval.

4. **Vector Embedding**: Each text chunk is converted into a vector embedding using a neural network model.

5. **Storage**: The chunks and their embeddings are stored in a vector database for fast retrieval.

You can monitor the progress of these steps in the system logs or on the Upload page.

## Querying the Knowledge Base

1. Navigate to the Query page
2. Enter your question in the text box
3. Click the Submit button
4. View the response with citations

## Managing the Knowledge Base

1. Navigate to the Management page
2. View all uploaded documents
3. Remove documents if needed
4. Update the knowledge base after changes
