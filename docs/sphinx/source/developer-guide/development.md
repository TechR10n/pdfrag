# Development Guide

This document contains development notes and additional information for developers working on the PDF RAG System.

## Project Structure

```
project/
├── app/
│   ├── api/
│   ├── models/
│   ├── scripts/
│   ├── static/
│   └── templates/
├── docs/
│   ├── puml/
│   └── sphinx/
└── tests/
```

## Setup Script for Ubuntu

I've created a comprehensive setup script (`setup_ubuntu.sh`) for your PDFrag system designed specifically for Ubuntu Linux users. Here's what the script does:

1. **System Checks**:
   - Verifies the user is not running as root (for security)
   - Checks if the system is running Ubuntu

2. **System Dependencies**:
   - Updates package lists
   - Installs required system packages:
     - Python 3 and development tools
     - Git, curl, wget
     - Build tools (build-essential, cmake)
     - Cairo graphics library (for PDF processing)
     - Docker and Docker Compose

3. **Docker Configuration**:
   - Ensures Docker service is running and enabled
   - Adds the current user to the docker group to avoid using sudo

4. **Repository Setup**:
   - Clones the repository if not already in it
   - Sets up a Python virtual environment
   - Installs Python dependencies from requirements.txt and requirements-dev.txt

5. **Directory Structure**:
   - Creates necessary data and model directories

6. **Environment Configuration**:
   - Creates a .env file from the template
   - Prompts for Hugging Face token
   - Generates a random Flask secret key
   - Sets PlantUML path if available

7. **Model Download**:
   - Downloads all required models using the download_models.sh script
   - Verifies models were downloaded successfully

8. **Docker Build**:
   - Builds the Docker containers

9. **Final Instructions**:
   - Provides clear instructions on how to start the system and access the web interface
   - Offers troubleshooting tips

### Usage Instructions

To use the script, a new user would:

1. Save the script as `setup_ubuntu.sh`
2. Make it executable: `chmod +x setup_ubuntu.sh`
3. Run it: `./setup_ubuntu.sh`
4. Follow the prompts, especially for the Hugging Face token
5. Once setup is complete, start the system with `./startup.sh`

### Important Notes

1. The script checks if the user recently added to the docker group needs to log out and back in for changes to take effect.
2. It handles the case where a user might run the script from outside the repository by cloning it first.
3. The script requires an internet connection to download packages and models.
4. A Hugging Face token with access to Meta Llama models is required.

Let me know if you need any clarifications or have any questions about the setup script!


## Development Environment

### Prerequisites

- Python 3.8+
- pip
- virtualenv or conda

### Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Run the development server

## Coding Standards

- Follow PEP 8 for Python code
- Use docstrings for all functions and classes
- Write unit tests for new features

## Deployment

### Local Deployment

Instructions for local deployment...

### Production Deployment

Instructions for production deployment...

## Testing

I'll explain how to run the text chunking tests as part of your routine testing process and how to set them up for future CI integration.

### Running Text Chunking Tests as Part of Routine Testing

You already have a good testing infrastructure set up with the `run_tests.py` script in your project root. To include the text chunking tests in your routine testing, you have a few options:

#### Option 1: Using the existing run_tests.py script

The simplest approach is to use your existing `run_tests.py` script, which should automatically discover and run all test files in the `app/tests` directory, including your new text chunking tests:

```bash
# Run all tests including text chunking tests
./run_tests.py

# Run with coverage
./run_tests.py --coverage
```

#### Option 2: Using the specialized run_chunking_tests.py script

If you want to run only the text chunking tests, you can use the specialized script you've created:

```bash
# Run just the text chunking tests
python app/tests/run_chunking_tests.py
```

#### Option 3: Combining both approaches

For a comprehensive testing routine, you might want to:

1. First run the specialized chunking tests to verify that component
2. Then run all tests to ensure everything works together

```bash
# Run text chunking tests first
python app/tests/run_chunking_tests.py

# Then run all tests
./run_tests.py
```

### Setting Up for CI Integration

When you're ready to set up CI, here's how you can integrate your tests:

#### GitHub Actions Example

Here's a sample GitHub Actions workflow file you could create at `.github/workflows/tests.yml`:

```yaml
name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
        
    - name: Generate test data
      run: |
        python app/tests/run_chunking_tests.py --generate-data
        
    - name: Run tests with pytest and coverage
      run: |
        ./run_tests.py --coverage
        
    - name: Upload coverage report
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report
        path: htmlcov/
```

### GitLab CI Example

For GitLab CI, you could create a `.gitlab-ci.yml` file:

```yaml
image: python:3.10

before_script:
  - pip install -r requirements.txt
  - pip install -r requirements-dev.txt

stages:
  - test

test:
  stage: test
  script:
    - python app/tests/run_chunking_tests.py --generate-data
    - ./run_tests.py --coverage
  artifacts:
    paths:
      - htmlcov/
```

#### Jenkins Pipeline Example

For Jenkins, you could create a `Jenkinsfile`:

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install -r requirements-dev.txt'
            }
        }
        stage('Generate Test Data') {
            steps {
                sh 'python app/tests/run_chunking_tests.py --generate-data'
            }
        }
        stage('Test') {
            steps {
                sh './run_tests.py --coverage'
            }
        }
    }
    post {
        always {
            publishHTML(target: [
                allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: true,
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }
    }
}
```

### Best Practices for CI Testing

1. **Test Markers**: Use your pytest markers to categorize tests. This allows you to run different test suites in different CI stages:
   ```bash
   # In CI, you might run fast tests first
   pytest app/tests -m "not slow"
   
   # Then run slow tests in a separate stage
   pytest app/tests -m "slow"
   ```

2. **Test Data Generation**: Your `test_data_generator.py` is excellent for CI. Make sure to generate test data at the beginning of your CI pipeline.

3. **Parallelization**: For larger test suites, consider using pytest-xdist to run tests in parallel:
   ```bash
   # Run tests using 4 parallel processes
   pytest app/tests -n 4
   ```

4. **Test Reports**: Generate JUnit XML reports for better CI integration:
   ```bash
   pytest app/tests --junitxml=test-results.xml
   ```

5. **Caching**: Use CI caching for pip dependencies to speed up your builds.

6. **Environment Variables**: Use environment variables for sensitive configuration in CI.

### Modifying run_tests.py to Include Specific Test Categories

You might want to update your main `run_tests.py` script to include specific options for running text chunking tests:

```python
# Add to your argument parser in run_tests.py
parser.add_argument("--chunking", action="store_true", help="Run only text chunking tests")

# Then in your main function
if args.chunking:
    test_files = [
        "app/tests/test_text_chunking.py",
        "app/tests/test_text_chunking_integration.py"
    ]
    pytest.main(["-xvs"] + test_files)
```

This would allow you to run:

```bash
./run_tests.py --chunking
```

By implementing these suggestions, you'll have a robust testing strategy that includes your text chunking tests in both manual testing workflows and future CI pipelines.

## Future Improvements

- List of planned improvements and features
- Known limitations and how they might be addressed

