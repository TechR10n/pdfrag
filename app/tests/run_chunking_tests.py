#!/usr/bin/env python3
"""
Script to run all text chunking tests.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_tests():
    """Run all text chunking tests."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define test files
    test_files = [
        os.path.join(script_dir, "test_text_chunking.py"),
        os.path.join(script_dir, "test_text_chunking_integration.py"),
    ]
    
    # Run tests
    print("Running text chunking tests...")
    exit_code = pytest.main(["-xvs"] + test_files)
    
    return exit_code

def generate_test_data():
    """Generate test data for text chunking tests."""
    from app.tests.test_data_generator import (
        generate_test_pdfs,
        generate_test_dataframe,
        generate_test_chunks_dataframe,
        generate_test_embeddings
    )
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define output directory
    output_dir = os.path.join(script_dir, "data", "test_pdfs")
    
    # Generate test PDFs
    pdf_paths = generate_test_pdfs(output_dir, count=3)
    print(f"Generated {len(pdf_paths)} test PDFs in {output_dir}")
    
    # Generate test DataFrame
    pdf_df = generate_test_dataframe(pdf_paths)
    print(f"Generated DataFrame with {len(pdf_df)} PDFs")
    
    # Generate test chunks
    chunks_df = generate_test_chunks_dataframe(pdf_df)
    print(f"Generated {len(chunks_df)} chunks")
    
    # Generate test embeddings
    embeddings_df = generate_test_embeddings(chunks_df)
    print(f"Generated embeddings with dimension {len(embeddings_df['embedding'][0])}")
    
    return {
        'pdf_paths': pdf_paths,
        'pdf_df': pdf_df,
        'chunks_df': chunks_df,
        'embeddings_df': embeddings_df
    }

def main():
    """Main function."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run text chunking tests.")
    parser.add_argument("--generate-data", action="store_true", help="Generate test data")
    parser.add_argument("--run-tests", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    # Default to running tests if no arguments are provided
    if not args.generate_data and not args.run_tests:
        args.run_tests = True
    
    # Generate test data if requested
    if args.generate_data:
        generate_test_data()
    
    # Run tests if requested
    if args.run_tests:
        exit_code = run_tests()
        sys.exit(exit_code)

if __name__ == "__main__":
    main() 