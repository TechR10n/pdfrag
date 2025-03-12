import os
import sys
import uuid
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.text_chunking import clean_text, chunk_text, process_chunks


class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_clean_text_removes_excessive_whitespace(self, sample_text_data):
        """Test that clean_text removes excessive whitespace."""
        cleaned = clean_text(sample_text_data['whitespace_text'])
        assert "Text with excessive whitespace" in cleaned
        assert "  " not in cleaned  # No double spaces
    
    def test_clean_text_normalizes_line_breaks(self, sample_text_data):
        """Test that clean_text normalizes line breaks."""
        cleaned = clean_text(sample_text_data['whitespace_text'])
        assert "\n\n\n" not in cleaned  # No triple line breaks
        assert "and line breaks" in cleaned
    
    def test_clean_text_strips_whitespace(self, sample_text_data):
        """Test that clean_text strips whitespace from beginning and end."""
        cleaned = clean_text(sample_text_data['whitespace_text'])
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
    
    def test_clean_text_handles_empty_string(self, sample_text_data):
        """Test that clean_text handles empty string."""
        cleaned = clean_text(sample_text_data['empty_text'])
        assert cleaned == ""


class TestChunkText:
    """Tests for the chunk_text function."""
    
    def test_chunk_text_splits_text(self, sample_text_data):
        """Test that chunk_text splits text into chunks."""
        chunks = chunk_text(sample_text_data['long_text'], chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1, "Text not split into multiple chunks"
    
    def test_chunk_size_respected(self, sample_text_data):
        """Test that chunk_text respects chunk size."""
        chunks = chunk_text(sample_text_data['long_text'], chunk_size=100, chunk_overlap=20)
        # Allow for some flexibility due to how RecursiveCharacterTextSplitter works
        # It tries to split on separators, so chunks might be smaller than chunk_size
        for chunk in chunks:
            assert len(chunk) <= 120, f"Chunk size exceeds expected maximum: {len(chunk)}"
    
    def test_chunk_overlap(self, sample_text_data):
        """Test that chunk_text includes overlap between chunks."""
        chunks = chunk_text(sample_text_data['long_text'], chunk_size=100, chunk_overlap=50)
        
        # Check if there's overlap between consecutive chunks
        if len(chunks) >= 2:
            # Find a word that should be in the overlap
            overlap_found = False
            for i in range(len(chunks) - 1):
                # Get the end of the first chunk
                end_of_first = chunks[i][-50:]
                # Check if any part of it is in the beginning of the next chunk
                if any(word in chunks[i+1][:100] for word in end_of_first.split() if len(word) > 3):
                    overlap_found = True
                    break
            assert overlap_found, "No overlap found between chunks"
    
    def test_chunk_text_with_empty_string(self, sample_text_data):
        """Test that chunk_text handles empty string."""
        chunks = chunk_text(sample_text_data['empty_text'])
        assert len(chunks) == 0, "Empty text should result in no chunks"
    
    def test_chunk_text_with_short_text(self, sample_text_data):
        """Test that chunk_text handles text shorter than chunk size."""
        chunks = chunk_text(sample_text_data['short_text'], chunk_size=100)
        assert len(chunks) == 1, "Short text should result in a single chunk"
        assert chunks[0] == sample_text_data['short_text'].strip(), "Chunk should contain the entire text"
    
    def test_chunk_text_with_paragraphs(self, sample_text_data):
        """Test that chunk_text uses separators correctly with paragraphs."""
        chunks = chunk_text(sample_text_data['paragraphs'], chunk_size=50, chunk_overlap=0)
        # Should split at paragraph breaks, not mid-paragraph
        assert "Paragraph 1" in chunks[0]
        # Check that we have multiple chunks
        assert len(chunks) > 1, "Text should be split into multiple chunks"


class TestProcessChunks:
    """Tests for the process_chunks function."""
    
    def test_process_chunks_creates_dataframe(self, sample_pdf_dataframe):
        """Test that process_chunks creates a DataFrame with chunks."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        assert isinstance(chunks_df, pd.DataFrame), "Result should be a DataFrame"
        assert len(chunks_df) > 0, "DataFrame should contain chunks"
    
    def test_process_chunks_skips_empty_text(self, sample_pdf_dataframe):
        """Test that process_chunks skips documents with empty text."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        # Check that no chunks were created for the empty document
        assert not any(chunks_df['filename'] == 'empty.pdf'), "Empty document should be skipped"
    
    def test_process_chunks_limits_chunks(self, sample_pdf_dataframe):
        """Test that process_chunks limits chunks per document."""
        max_chunks = 2
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=50, chunk_overlap=10, max_chunks_per_doc=max_chunks)
        
        # Group by filename and count chunks
        chunk_counts = chunks_df.groupby('filename').size()
        
        # Check that no document has more than max_chunks
        for count in chunk_counts:
            assert count <= max_chunks, f"Document has more than {max_chunks} chunks"
    
    def test_process_chunks_generates_unique_ids(self, sample_pdf_dataframe):
        """Test that process_chunks generates unique IDs for chunks."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Check that all chunk_ids are unique
        assert len(chunks_df['chunk_id'].unique()) == len(chunks_df), "Chunk IDs should be unique"
        
        # Check that chunk_ids are valid UUIDs
        for chunk_id in chunks_df['chunk_id']:
            try:
                uuid.UUID(chunk_id)
                is_valid = True
            except ValueError:
                is_valid = False
            assert is_valid, f"Chunk ID {chunk_id} is not a valid UUID"
    
    def test_process_chunks_includes_metadata(self, sample_pdf_dataframe):
        """Test that process_chunks includes metadata from the original DataFrame."""
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Check that metadata columns are present
        assert 'pdf_path' in chunks_df.columns
        assert 'filename' in chunks_df.columns
        assert 'chunk_index' in chunks_df.columns
        assert 'chunk_text' in chunks_df.columns
        assert 'token_count' in chunks_df.columns
        
        # Check that metadata is correctly copied
        for _, row in chunks_df.iterrows():
            original_row = sample_pdf_dataframe[sample_pdf_dataframe['filename'] == row['filename']].iloc[0]
            assert row['pdf_path'] == original_row['path']
            assert row['filename'] == original_row['filename']
    
    @patch('app.utils.text_chunking.chunk_text')
    def test_process_chunks_calls_chunk_text(self, mock_chunk_text, sample_pdf_dataframe):
        """Test that process_chunks calls chunk_text with correct parameters."""
        mock_chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        
        process_chunks(sample_pdf_dataframe, chunk_size=200, chunk_overlap=30)
        
        # Check that chunk_text was called for each document with text
        assert mock_chunk_text.call_count == 2  # Two documents with text
        
        # Check that chunk_text was called with correct parameters
        mock_chunk_text.assert_any_call(sample_pdf_dataframe.iloc[0]['text'], 200, 30)
        mock_chunk_text.assert_any_call(sample_pdf_dataframe.iloc[1]['text'], 200, 30)


# Integration tests
class TestTextChunkingIntegration:
    """Integration tests for text chunking functionality."""
    
    def test_end_to_end_chunking(self, sample_pdf_dataframe):
        """Test the entire chunking process from text to DataFrame."""
        # Process chunks
        chunks_df = process_chunks(sample_pdf_dataframe, chunk_size=100, chunk_overlap=20)
        
        # Verify results
        assert isinstance(chunks_df, pd.DataFrame)
        assert len(chunks_df) > 1
        assert 'chunk_id' in chunks_df.columns
        assert 'pdf_path' in chunks_df.columns
        assert 'filename' in chunks_df.columns
        assert 'chunk_index' in chunks_df.columns
        assert 'chunk_text' in chunks_df.columns
        assert 'token_count' in chunks_df.columns
        
        # Check that chunk indices are sequential within each document
        for filename in chunks_df['filename'].unique():
            doc_chunks = chunks_df[chunks_df['filename'] == filename]
            assert list(doc_chunks['chunk_index']) == list(range(len(doc_chunks)))
        
        # Check that token counts are reasonable
        for _, row in chunks_df.iterrows():
            assert row['token_count'] > 0
            assert row['token_count'] == len(row['chunk_text'].split())


if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__]) 