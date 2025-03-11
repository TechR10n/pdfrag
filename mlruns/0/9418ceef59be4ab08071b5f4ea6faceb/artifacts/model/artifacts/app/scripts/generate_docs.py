#!/usr/bin/env python3
"""
Python wrapper for the generate_docs.sh script.
This script is provided for users who accidentally try to run the shell script with Python.
"""

import os
import sys
import subprocess

def main():
    """Main function to parse arguments and call the shell script."""
    print("This is a Python wrapper for the generate_docs.sh shell script.")
    print("Redirecting to the shell script...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the shell script
    shell_script = os.path.join(script_dir, "generate_docs.sh")
    
    # Make sure the shell script is executable
    os.chmod(shell_script, 0o755)
    
    # Construct the command
    cmd = [shell_script]
    
    # Add any arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    # Print the command being run
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run the shell script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: The shell script exited with code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Could not find the shell script at {shell_script}")
        sys.exit(1)

if __name__ == "__main__":
    main() 