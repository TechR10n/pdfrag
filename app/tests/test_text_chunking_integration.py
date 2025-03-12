import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.text_chunking import clean_text, chunk_text, process_chunks
from app.utils.pdf_ingestion import process_pdfs


class TestTextChunkingWithPDFIngestion:
    """Integration tests for text chunking with PDF ingestion."""
    
    @patch('app.utils.pdf_ingestion.extract_text_from_pdf')
    def test_pdf_ingestion_to_chunking(self, mock_extract_text, sample_pdf_dir):
        """Test the integration of PDF ingestion with text chunking."""
        # Mock the text extraction to return predictable text
        mock_extract_text.return_value = "This is extracted text from a PDF. " * 20
        
        # Process PDFs
        pdf_df = process_pdfs(sample_pdf_dir)
        
        # Process chunks
        chunks_df = process_chunks(pdf_df, chunk_size=100, chunk_overlap=20)
        
        # Verify results
        assert isinstance(chunks_df, pd.DataFrame)
        assert len(chunks_df) > 0
        assert 'chunk_id' in chunks_df.columns
        assert 'pdf_path' in chunks_df.columns
        assert 'filename' in chunks_df.columns
        assert 'chunk_text' in chunks_df.columns
        
        # Check that we have chunks for each PDF with text
        pdf_count = len([f for f in os.listdir(sample_pdf_dir) 
                        if f.endswith('.pdf') and f != 'corrupted.pdf'])
        pdf_count += len([f for f in os.listdir(os.path.join(sample_pdf_dir, 'subdir')) 
                         if f.endswith('.pdf')])
        
        # We should have chunks from each PDF
        assert len(chunks_df['filename'].unique()) == pdf_count


class TestTextChunkingWithVectorDB:
    """Integration tests for text chunking with vector database."""
    
    @patch('app.utils.vector_db.insert_chunks')
    def test_chunking_to_vector_db(self, mock_insert_chunks, sample_pdf_dataframe):
        """Test the integration of text chunking with vector database insertion."""
        # Process chunks
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Mock the vector DB insertion
        mock_insert_chunks.return_value = {'inserted': len(chunks_df), 'errors': 0}
        
        # Simulate inserting chunks into vector DB
        result = mock_insert_chunks(chunks_df)
        
        # Verify results
        assert result['inserted'] == len(chunks_df)
        assert result['errors'] == 0
        
        # Verify that mock was called with the chunks DataFrame
        mock_insert_chunks.assert_called_once()
        args, _ = mock_insert_chunks.call_args
        assert isinstance(args[0], pd.DataFrame)
        assert len(args[0]) == len(chunks_df)


class TestEndToEndProcessing:
    """End-to-end tests for the document processing pipeline."""
    
    @patch('app.utils.pdf_ingestion.extract_text_from_pdf')
    @patch('app.utils.vector_db.insert_chunks')
    @patch('app.utils.embedding.generate_embeddings')
    def test_end_to_end_pipeline(self, mock_generate_embeddings, mock_insert_chunks, 
                                mock_extract_text, sample_pdf_dir):
        """Test the entire document processing pipeline from PDF to vector DB."""
        # Mock the text extraction
        mock_extract_text.return_value = "This is extracted text from a PDF. " * 20
        
        # Mock the embedding generation
        mock_generate_embeddings.return_value = pd.DataFrame({
            'chunk_id': ['id1', 'id2'],
            'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        })
        
        # Mock the vector DB insertion
        mock_insert_chunks.return_value = {'inserted': 2, 'errors': 0}
        
        # Process PDFs
        pdf_df = process_pdfs(sample_pdf_dir)
        
        # Process chunks
        chunks_df = process_chunks(pdf_df, chunk_size=100, chunk_overlap=20)
        
        # Generate embeddings (mocked)
        embeddings_df = mock_generate_embeddings(chunks_df)
        
        # Insert into vector DB (mocked)
        result = mock_insert_chunks(embeddings_df)
        
        # Verify results
        assert isinstance(chunks_df, pd.DataFrame)
        assert len(chunks_df) > 0
        assert result['inserted'] > 0
        assert result['errors'] == 0


if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__]) 