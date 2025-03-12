#!/usr/bin/env python3
"""
Script to scrape code from app directory, Docker files, and shell scripts.
The contents are concatenated into eval_code.md with file paths as headers.
"""

import os
import glob
from pathlib import Path

def is_binary_file(file_path):
    """Check if a file is binary by reading its first few thousand bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(4096)
            return b'\0' in chunk  # Binary files typically contain null bytes
    except Exception:
        return True  # If we can't read it, assume it's binary

def should_include_file(file_path):
    """Determine if a file should be included in the output."""
    # Skip binary files, hidden files, and certain directories
    if (is_binary_file(file_path) or 
        os.path.basename(file_path).startswith('.') or
        '__pycache__' in file_path or
        'venv' in file_path or
        '.git' in file_path):
        return False
    
    # Include files from app directory
    if file_path.startswith('app/'):
        return True
    
    # Include files from flask-app directory
    if file_path.startswith('flask-app/'):
        return True
    
    # Include Docker files
    if ('Dockerfile' in file_path or 
        'docker-compose' in file_path):
        return True
    
    # Include shell scripts
    if file_path.endswith('.sh'):
        return True
    
    return False

def main():
    # Get the project root directory
    project_root = os.getcwd()
    output_file = os.path.join(project_root, 'eval_code.md')
    
    # Initialize the output file
    with open(output_file, 'w') as f:
        f.write("# Code Evaluation File\n\n")
        f.write("This file contains code from the project for evaluation purposes.\n\n")
    
    # Find all files in the app directory
    app_files = []
    for root, dirs, files in os.walk(os.path.join(project_root, 'app')):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, project_root)
            app_files.append(rel_path)
    
    # Find all files in the flask-app directory
    flask_app_files = []
    if os.path.exists(os.path.join(project_root, 'flask-app')):
        for root, dirs, files in os.walk(os.path.join(project_root, 'flask-app')):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_root)
                flask_app_files.append(rel_path)
    
    # Find all Docker files
    docker_files = []
    # Root Dockerfiles
    for file in glob.glob(os.path.join(project_root, 'Dockerfile*')):
        rel_path = os.path.relpath(file, project_root)
        docker_files.append(rel_path)
    
    # Dockerfiles in subdirectories
    for file in glob.glob(os.path.join(project_root, '**', 'Dockerfile*'), recursive=True):
        rel_path = os.path.relpath(file, project_root)
        docker_files.append(rel_path)
    
    # Docker compose files
    for file in glob.glob(os.path.join(project_root, '**/docker-compose*.yml'), recursive=True):
        rel_path = os.path.relpath(file, project_root)
        docker_files.append(rel_path)
    
    if os.path.exists(os.path.join(project_root, 'docker-compose.yml')):
        docker_files.append('docker-compose.yml')
    
    # Find all shell scripts
    sh_files = []
    for file in glob.glob(os.path.join(project_root, '*.sh')):
        rel_path = os.path.relpath(file, project_root)
        sh_files.append(rel_path)
    
    # Also check for shell scripts in the app directory
    for file in glob.glob(os.path.join(project_root, 'app', '**', '*.sh'), recursive=True):
        rel_path = os.path.relpath(file, project_root)
        sh_files.append(rel_path)
    
    # Combine all files
    all_files = app_files + flask_app_files + docker_files + sh_files
    
    # Remove duplicates and sort
    all_files = sorted(set(all_files))
    
    # Process each file
    total_files = len(all_files)
    processed_files = 0
    skipped_files = 0
    max_file_size = 1024 * 1024  # 1MB limit
    
    print(f"Found {total_files} files to process")
    
    for file_path in all_files:
        processed_files += 1
        if processed_files % 10 == 0 or processed_files == total_files:
            print(f"Progress: {processed_files}/{total_files} files processed")
            
        abs_path = os.path.join(project_root, file_path)
        
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            skipped_files += 1
            continue
            
        # Skip files that are too large
        file_size = os.path.getsize(abs_path)
        if file_size > max_file_size:
            print(f"Skipping {file_path}: File too large ({file_size/1024:.1f} KB)")
            skipped_files += 1
            continue
            
        if not should_include_file(file_path):
            skipped_files += 1
            continue
        
        try:
            # Try to read with utf-8 encoding first
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # If utf-8 fails, try with latin-1 encoding (which can read any byte sequence)
                with open(abs_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    print(f"Warning: {file_path} was read with latin-1 encoding")
            
            # Determine file extension for markdown code block
            _, ext = os.path.splitext(file_path)
            if ext:
                ext = ext[1:]  # Remove the dot
            
            # Map extensions to markdown code block language
            lang_map = {
                'py': 'python',
                'sh': 'bash',
                'yml': 'yaml',
                'yaml': 'yaml',
                'md': 'markdown',
                'js': 'javascript',
                'html': 'html',
                'css': 'css',
                'json': 'json',
                'txt': 'text'
            }
            
            lang = lang_map.get(ext, '')
            
            # Special case for Dockerfiles
            if 'Dockerfile' in file_path:
                lang = 'dockerfile'
            
            # Append to the output file
            with open(output_file, 'a') as f:
                f.write(f"## {file_path}\n\n")
                f.write(f"```{lang}\n")
                f.write(content)
                if not content.endswith('\n'):
                    f.write('\n')
                f.write("```\n\n")
            
            print(f"Processed: {file_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nCompleted! Output saved to {output_file}")
    print(f"Summary: {processed_files} files processed, {processed_files - skipped_files} files included, {skipped_files} files skipped")

if __name__ == "__main__":
    main() 