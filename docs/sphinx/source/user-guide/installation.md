# Installation

This section covers how to install and set up the PDF RAG System.

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the configuration:
   ```bash
   cp app/config/config.example.py app/config/config.py
   # Edit app/config/config.py with your settings
   ```

5. Run the application:
   ```bash
   python app/main.py
   ```

The application should now be running at http://localhost:5000.
