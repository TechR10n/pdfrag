#!/usr/bin/env python3
"""
Script to convert SVG files to PDF format for better LaTeX compatibility.
"""

import sys
import subprocess
from pathlib import Path
import os

def convert_svg_to_pdf(svg_file):
    """
    Convert an SVG file to PDF format using Inkscape.
    
    Args:
        svg_file: Path to the SVG file
    
    Returns:
        Path to the generated PDF file
    """
    svg_path = Path(svg_file)
    if not svg_path.exists():
        print(f"Error: {svg_path} not found.")
        return None
    
    pdf_path = svg_path.with_suffix('.pdf')
    
    try:
        # Try using Inkscape to convert SVG to PDF
        print(f"Converting {svg_path} to {pdf_path}...")
        subprocess.run(
            ['inkscape', '--export-filename=' + str(pdf_path), str(svg_path)],
            check=True
        )
        print(f"PDF generated at {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"Error during SVG to PDF conversion: {e}")
        return None
    except FileNotFoundError:
        print("Inkscape not found. Please install Inkscape or ensure it's in your PATH.")
        return None

def find_and_convert_svg_files(directory='.'):
    """
    Find all SVG files in the given directory and its subdirectories,
    and convert them to PDF format.
    
    Args:
        directory: Directory to search for SVG files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} not found.")
        return
    
    svg_files = list(dir_path.glob('**/*.svg'))
    if not svg_files:
        print(f"No SVG files found in {dir_path}")
        return
    
    print(f"Found {len(svg_files)} SVG files.")
    
    for svg_file in svg_files:
        convert_svg_to_pdf(svg_file)

def update_markdown_file(markdown_file):
    """
    Update a Markdown file to use PDF images instead of SVG.
    
    Args:
        markdown_file: Path to the Markdown file
    """
    md_path = Path(markdown_file)
    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        return
    
    try:
        with open(md_path, 'r') as file:
            content = file.read()
        
        # Replace SVG image references with PDF
        updated_content = content.replace('.svg)', '.pdf)')
        
        # Write the updated content back to the file
        with open(md_path, 'w') as file:
            file.write(updated_content)
        
        print(f"Updated {md_path} to use PDF images.")
    except Exception as e:
        print(f"Error updating Markdown file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_svg_to_pdf.py [directory] [markdown_file]")
        sys.exit(1)
    
    directory = sys.argv[1]
    find_and_convert_svg_files(directory)
    
    if len(sys.argv) > 2:
        markdown_file = sys.argv[2]
        update_markdown_file(markdown_file) 