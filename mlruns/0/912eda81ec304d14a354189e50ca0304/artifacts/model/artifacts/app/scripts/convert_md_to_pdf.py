import sys
import subprocess
from pathlib import Path
import argparse
import os
import re

def main():
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF with academic styling')
    parser.add_argument('input_file', help='Input markdown file')
    parser.add_argument('--title', default='Laboratory Handbook: Building a Local RAG System with Flask and MLflow', help='Document title')
    parser.add_argument('--author', default='Ryan Hammang', help='Author name')
    parser.add_argument('--date', default='March 11, 2025', help='Document date (default: today)')
    parser.add_argument('--abstract', default='Creating a local RAG system with Flask and MLflow.', 
                        help='Document abstract')
    parser.add_argument('--affiliation', default='Organization or Institution Name', 
                        help='Author affiliation')
    parser.add_argument('--documentclass', default='acmart', 
                        choices=['IEEEtran', 'acmart', 'article', 'llncs', 'elsarticle'],
                        help='LaTeX document class to use')
    parser.add_argument('--classoption', default='screen,review,acmlarge', 
                        help='Options for the document class')
    parser.add_argument('--latex-engine', default='pdflatex',
                        choices=['pdflatex', 'xelatex', 'lualatex'],
                        help='LaTeX engine to use for PDF generation')
    parser.add_argument('--keep-tex', action='store_true',
                        help='Keep the intermediate .tex file')
    parser.add_argument('--tex-only', action='store_true',
                        help='Only generate the .tex file, do not compile to PDF')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    tex_file = input_file.with_suffix('.tex')
    pdf_file = input_file.with_suffix('.pdf')
    preamble_path = Path('app/scripts/preamble.tex')
    if not preamble_path.exists():
        print(f"Error: preamble.tex not found.")
        sys.exit(1)

    # Step 1: Convert Markdown to LaTeX using Pandoc
    pandoc_command = [
        'pandoc',
        str(input_file),
        '-o',
        str(tex_file),
        '--standalone',  # Create a complete LaTeX document
        '--listings',
        '--no-highlight',  # Disable syntax highlighting to avoid special character issues
        '--wrap=preserve',  # Preserve line wrapping
        '-V',
        f'documentclass={args.documentclass}',
    ]
    
    # Only add classoption if it's not empty
    if args.classoption:
        pandoc_command.extend(['-V', f'classoption={args.classoption}'])
    
    pandoc_command.extend([
        '--include-in-header',
        str(preamble_path),
        # Front matter metadata
        '-M', f'title={args.title}',
        '-M', f'author={args.author}',
        '-M', f'date={args.date}',
        '-M', f'abstract={args.abstract}'
    ])
    
    # Add the appropriate affiliation/institute parameter based on document class
    if args.documentclass == 'llncs':
        pandoc_command.extend(['-M', f'institute={args.affiliation}'])
    else:
        pandoc_command.extend(['-M', f'affiliation={args.affiliation}'])

    print(f"Converting {input_file} to {tex_file} using Pandoc...")
    try:
        subprocess.run(pandoc_command, check=True)
        print(f"LaTeX file generated at {tex_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during Markdown to LaTeX conversion: {e}")
        sys.exit(1)
    
    # Modify the generated .tex file
    print("Modifying the generated .tex file...")
    try:
        with open(tex_file, 'r') as file:
            tex_content = file.read()
        
        # Remove \usepackage{amsmath,amssymb} line
        tex_content = re.sub(r'\\usepackage\{amsmath,amssymb\}', '', tex_content)
        
        # Fix special characters in tabular environments
        # Look for tabular environments and escape underscores and other special characters
        def fix_tabular_content(match):
            tabular_content = match.group(0)
            # Escape underscores in tabular content
            tabular_content = tabular_content.replace('_', '\\_')
            return tabular_content
        
        tex_content = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', fix_tabular_content, tex_content, flags=re.DOTALL)
        
        # For ACM, completely restructure the document to ensure abstract is before \maketitle
        if args.documentclass == 'acmart':
            # Extract the abstract text
            abstract_text = args.abstract  # Default to command line argument
            
            # Try to find abstract in the document
            abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex_content, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
            
            # Remove any existing abstract
            tex_content = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', tex_content, flags=re.DOTALL)
            
            # Find the \begin{document} and \maketitle positions
            begin_doc_match = re.search(r'\\begin\{document\}', tex_content)
            maketitle_match = re.search(r'\\maketitle', tex_content)
            
            if begin_doc_match and maketitle_match:
                # Split the content
                begin_doc_pos = begin_doc_match.end()
                maketitle_pos = maketitle_match.start()
                
                # Reconstruct the document with abstract before \maketitle
                new_content = (
                    tex_content[:begin_doc_pos] + 
                    '\n\n\\begin{abstract}\n' + abstract_text + '\n\\end{abstract}\n\n' +
                    tex_content[begin_doc_pos:maketitle_pos] +
                    '\\maketitle\n\n' +
                    tex_content[maketitle_match.end():]
                )
                
                tex_content = new_content
        
        with open(tex_file, 'w') as file:
            file.write(tex_content)
        
        print("LaTeX file successfully modified.")
    except Exception as e:
        print(f"Error modifying the LaTeX file: {e}")
        sys.exit(1)
    
    # Exit if user only wants the .tex file
    if args.tex_only:
        print("Skipping PDF generation as requested (--tex-only flag used).")
        return

    # Step 2: Compile LaTeX to PDF using the local LaTeX engine
    print(f"Compiling {tex_file} to {pdf_file} using {args.latex_engine}...")
    
    # Change to the directory containing the .tex file for proper relative path handling
    working_dir = tex_file.parent
    tex_filename = tex_file.name
    
    # Run LaTeX engine twice to resolve references
    for i in range(2):
        try:
            subprocess.run(
                [args.latex_engine, 
                 '-shell-escape',  # Enable shell escape for SVG processing
                 tex_filename], 
                check=True,
                cwd=str(working_dir)  # Set working directory
            )
        except subprocess.CalledProcessError as e:
            print(f"Error during LaTeX to PDF compilation: {e}")
            sys.exit(1)
    
    print(f"PDF generated at {pdf_file}")
    
    # Clean up intermediate files unless --keep-tex is specified
    if not args.keep_tex:
        print("Cleaning up intermediate files...")
        extensions_to_remove = ['.aux', '.log', '.out', '.toc']
        if not args.keep_tex:
            extensions_to_remove.append('.tex')
        
        for ext in extensions_to_remove:
            temp_file = input_file.with_suffix(ext)
            if temp_file.exists():
                os.remove(temp_file)
                print(f"Removed {temp_file}")

if __name__ == "__main__":
    main()