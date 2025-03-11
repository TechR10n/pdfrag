#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def fix_directory_tree(content):
    """Convert ASCII directory trees to LaTeX-friendly format."""
    # Replace ASCII characters with LaTeX commands
    replacements = {
        '├': '\\textbar--',
        '│': '\\textbar\\phantom{--}',
        '└': '\\textbar\\_',
        '─': '-',
    }
    
    def replace_tree_chars(match):
        line = match.group(0)
        for char, repl in replacements.items():
            line = line.replace(char, repl)
        return line
    
    # Find directory trees (indented blocks with ASCII characters)
    return re.sub(
        r'^([ \t]*[│├└].+$\n?)+',
        lambda m: '\\begin{verbatim}\n' + m.group(0) + '\\end{verbatim}\n',
        content,
        flags=re.MULTILINE
    )

def fix_tables(content):
    """Convert markdown tables to LaTeX tables."""
    def process_table(match):
        lines = match.group(0).strip().split('\n')
        if not lines:
            return match.group(0)
            
        # Count columns from header
        num_cols = len(lines[0].split('|')) - 2  # -2 for empty edges
        
        # Start table environment
        result = ['\\begin{table}[htbp]', '\\centering', '\\begin{tabular}{|' + 'l|' * num_cols + '}']
        result.append('\\hline')
        
        for i, line in enumerate(lines):
            # Skip separator line
            if i == 1 and all(c in '|-' for c in line.strip('| ')):
                continue
                
            # Process cells
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            # Escape special characters in cells
            cells = [
                cell.replace('_', '\\_')
                    .replace('&', '\\&')
                    .replace('%', '\\%')
                    .replace('#', '\\#')
                    .replace('$', '\\$')
                for cell in cells
            ]
            result.append(' & '.join(cells) + ' \\\\')
            result.append('\\hline')
        
        result.extend(['\\end{tabular}', '\\end{table}'])
        return '\n'.join(result)
    
    # Find and process tables
    return re.sub(
        r'^\|.+\|$\n\|[-|\s]+\|\n(\|.+\|$\n?)+',
        process_table,
        content,
        flags=re.MULTILINE
    )

def fix_code_blocks(content):
    """Convert markdown code blocks to LaTeX listings."""
    def process_code_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        
        # Special handling for directory trees
        if '├' in code or '│' in code or '└' in code:
            return fix_directory_tree(code)
            
        # Escape special characters in code
        code = (
            code.strip()
            .replace('_', '\\_')
            .replace('&', '\\&')
            .replace('%', '\\%')
            .replace('#', '\\#')
            .replace('$', '\\$')
        )
        
        # Regular code block
        return (
            '\\begin{lstlisting}' +
            ('[language=' + lang + ']' if lang else '') +
            '\n' + code + '\n\\end{lstlisting}'
        )
    
    return re.sub(
        r'```(\w+)?\n(.*?)```',
        process_code_block,
        content,
        flags=re.DOTALL
    )

def fix_inline_code(content):
    """Fix inline code and special characters."""
    # Fix inline code with special characters
    content = re.sub(
        r'`([^`]*)`',
        lambda m: '\\texttt{' + (
            m.group(1)
            .replace('_', '\\_')
            .replace('$', '\\$')
            .replace('&', '\\&')
            .replace('#', '\\#')
            .replace('%', '\\%')
            .replace('{', '\\{')
            .replace('}', '\\}')
            .replace('~', '\\~{}')
            .replace('^', '\\^{}')
            .replace('\\', '\\textbackslash{}')
        ) + '}',
        content
    )
    
    # Fix arrows and other special characters
    replacements = {
        '→': '$\\rightarrow$',
        '←': '$\\leftarrow$',
        '↑': '$\\uparrow$',
        '↓': '$\\downarrow$',
        '≤': '$\\leq$',
        '≥': '$\\geq$',
        '≠': '$\\neq$',
        '×': '$\\times$',
        '…': '\\ldots{}',
        ''': "'",
        ''': "'",
        '"': "``",
        '"': "''",
        '—': '---',
        '–': '--',
    }
    
    for char, repl in replacements.items():
        content = content.replace(char, repl)
    
    # Fix URLs
    content = re.sub(
        r'\[(.*?)\]\((.*?)\)',
        lambda m: '\\href{' + m.group(2).replace('%', '\\%') + '}{' + m.group(1) + '}',
        content
    )
    
    return content

def fix_images(content):
    """Fix image references."""
    def process_image(match):
        alt_text = match.group(1)
        path = match.group(2)
        
        # Convert SVG path to PDF
        if path.endswith('.svg'):
            path = path.replace('.svg', '.pdf')
            
        # Make path relative to the document
        if path.startswith('../../'):
            path = path[6:]
            
        # Handle image references
        if path.startswith('puml/'):
            path = '_images/' + path.replace('puml/', '')
            
        return (
            '\\begin{figure}[htbp]\n'
            '\\centering\n'
            '\\includegraphics[width=0.8\\textwidth]{' + path + '}\n'
            '\\caption{' + alt_text + '}\n'
            '\\label{fig:' + alt_text.lower().replace(' ', '-') + '}\n'
            '\\end{figure}'
        )
    
    # First convert any image links to proper markdown image syntax
    content = re.sub(
        r'\[([^]]+)\]\((.*?\.(?:svg|pdf))\)',
        r'![\1](\2)',
        content
    )
    
    # Then convert all images to figures
    content = re.sub(
        r'!\[(.*?)\]\((.*?)\)',
        process_image,
        content
    )
    
    # Remove any extra exclamation marks before figures
    content = re.sub(
        r'!\s*\\begin{figure}',
        r'\\begin{figure}',
        content
    )
    
    return content

def fix_lists(content):
    """Fix markdown lists to use proper LaTeX itemize/enumerate environments."""
    # Convert unordered lists
    content = re.sub(
        r'(?m)^(\s*)-\s',
        r'\1\\item ',
        content
    )
    
    # Convert ordered lists
    content = re.sub(
        r'(?m)^(\s*)\d+\.\s',
        r'\1\\item ',
        content
    )
    
    # Wrap lists in proper environments
    content = re.sub(
        r'(?sm)^\\item(.*?)(?=^[^\\]|\Z)',
        r'\\begin{itemize}\n\\item\1\\end{itemize}\n',
        content
    )
    
    return content

def fix_headings(content):
    """Convert markdown headings to LaTeX sections."""
    replacements = [
        (r'^#\s+(.+)$', r'\\section{\1}'),
        (r'^##\s+(.+)$', r'\\subsection{\1}'),
        (r'^###\s+(.+)$', r'\\subsubsection{\1}'),
        (r'^####\s+(.+)$', r'\\paragraph{\1}'),
        (r'^#####\s+(.+)$', r'\\subparagraph{\1}'),
    ]
    
    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content, flags=re.MULTILINE)
    
    return content

def fix_markdown_for_latex(content):
    """Fix markdown content for LaTeX processing."""
    # Process in specific order - images first to avoid interference with other elements
    content = fix_images(content)
    content = fix_headings(content)
    content = fix_tables(content)
    content = fix_code_blocks(content)
    content = fix_inline_code(content)
    content = fix_lists(content)
    
    # Fix include directives
    content = re.sub(
        r'\{include\} (.*?)\.md',
        r'\\input{\1}',
        content
    )
    
    # Add LaTeX document class and preamble if not present
    if not content.startswith('\\documentclass'):
        content = (
            '% Auto-generated LaTeX document\n'
            '\\input{preamble}\n\n'
            '\\begin{document}\n\n' +
            content +
            '\n\n\\end{document}\n'
        )
    
    return content

def process_file(file_path):
    """Process a markdown file and fix it for LaTeX."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found")
        return False
    
    try:
        content = path.read_text()
        fixed_content = fix_markdown_for_latex(content)
        
        # Create a new file with _latex suffix
        new_path = path.parent / (path.stem + '_latex' + path.suffix)
        new_path.write_text(fixed_content)
        print(f"Created LaTeX-compatible version at {new_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_markdown.py <markdown_file>")
        sys.exit(1)
    
    success = process_file(sys.argv[1])
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 