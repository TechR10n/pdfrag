#!/usr/bin/env python3
"""
Test runner script for the PDF RAG System.
"""

import os
import sys
import argparse
import subprocess

def run_unittest(args):
    """Run tests using unittest."""
    if args.test_file:
        cmd = [sys.executable, "-m", "unittest", args.test_file]
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "app/tests"]
    
    return subprocess.run(cmd).returncode

def run_pytest(args):
    """Run tests using pytest."""
    cmd = ["pytest"]
    
    if args.test_file:
        cmd.append(args.test_file)
    else:
        cmd.append("app/tests")
    
    if args.coverage:
        cmd.extend(["--cov=app", "--cov-report=term", "--cov-report=html"])
    
    if args.verbose:
        cmd.append("-v")
    
    # Add markers if specified
    if args.unit:
        cmd.append("-m unit")
    elif args.integration:
        cmd.append("-m integration")
    elif args.api:
        cmd.append("-m api")
    elif args.model:
        cmd.append("-m model")
    elif args.pdf:
        cmd.append("-m pdf")
    
    # Include slow tests if specified
    if args.runslow:
        cmd.append("--runslow")
    
    return subprocess.run(cmd).returncode

def main():
    parser = argparse.ArgumentParser(description="Run tests for PDF RAG System")
    parser.add_argument("--framework", choices=["unittest", "pytest"], default="pytest",
                        help="Testing framework to use (default: pytest)")
    parser.add_argument("--test-file", help="Specific test file to run")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report (pytest only)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Add marker options
    marker_group = parser.add_argument_group("test markers")
    marker_group.add_argument("--unit", action="store_true", help="Run only unit tests")
    marker_group.add_argument("--integration", action="store_true", help="Run only integration tests")
    marker_group.add_argument("--api", action="store_true", help="Run only API tests")
    marker_group.add_argument("--model", action="store_true", help="Run only model tests")
    marker_group.add_argument("--pdf", action="store_true", help="Run only PDF tests")
    marker_group.add_argument("--runslow", action="store_true", help="Include slow tests")
    
    args = parser.parse_args()
    
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Run tests with the selected framework
    if args.framework == "unittest":
        return run_unittest(args)
    else:
        return run_pytest(args)

if __name__ == "__main__":
    sys.exit(main()) 