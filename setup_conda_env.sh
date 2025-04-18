#!/bin/bash

# This script creates and activates a Conda/Mamba environment for the PyTorch Docs Search project
# using the environment.yml file

# Exit immediately if a command exits with a non-zero status
set -e

# Check if mamba is available, otherwise use conda
if command -v mamba &> /dev/null; then
    PACKAGE_MANAGER="mamba"
    echo "Using Mamba for faster dependency resolution"
elif command -v conda &> /dev/null; then
    PACKAGE_MANAGER="conda"
    echo "Using Conda (consider installing Mamba for faster dependency resolution)"
else
    echo "Error: Neither conda nor mamba is installed or in PATH."
    echo "Please install Miniconda/Mamba before running this script."
    exit 1
fi

# Source conda.sh to enable conda/mamba command
source "$(conda info --base)/etc/profile.d/conda.sh"

# Initialize mamba if it's being used
if [ "$PACKAGE_MANAGER" = "mamba" ]; then
    eval "$(mamba shell hook --shell bash)"
fi

# Check if the environment.yml file exists
if [ ! -f "environment.yml" ]; then
    echo "Error: environment.yml file not found in the current directory."
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "pytorch_docs_search"; then
    echo "The 'pytorch_docs_search' environment already exists."
    echo "Do you want to update it? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Updating environment from environment.yml..."
        $PACKAGE_MANAGER env update -f environment.yml
    else
        echo "Skipping environment update."
    fi
else
    echo "Creating new environment from environment.yml..."
    # Set a longer timeout for environment creation
    export CONDA_SOLVE_TIMEOUT=600  # 10-minute timeout instead of the default
    
    # Create the environment with increased verbosity
    $PACKAGE_MANAGER env create -f environment.yml -v
fi

# Activate the environment
$PACKAGE_MANAGER activate pytorch_docs_search

# Create a test script if it doesn't exist
if [ ! -f "test_conda_env.py" ]; then
    cat > test_conda_env.py << 'EOF'
#!/usr/bin/env python3
"""Test that the Conda environment is set up correctly."""

import sys
import importlib

def test_imports():
    """Test importing the required packages."""
    packages = [
        "numpy", "torch", "chromadb", "openai",
        "tqdm", "dotenv", "psutil", "pytest", "flask"
    ]
    
    success = True
    print("Testing package imports:")
    
    for package in packages:
        try:
            # Handle special cases
            if package == "dotenv":
                package = "python_dotenv"
                module = importlib.import_module("dotenv")
            else:
                module = importlib.import_module(package)
            
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {package}: {version}")
        except ImportError as e:
            success = False
            print(f"  ✗ {package}: Not found - {e}")
    
    return success

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    if test_imports():
        print("\nEnvironment test passed successfully!")
        sys.exit(0)
    else:
        print("\nEnvironment test failed. Some packages are missing.")
        sys.exit(1)
EOF
    echo "Created test_conda_env.py"
fi

# Verify installation by running the test script
echo -e "\nVerifying environment setup..."
python test_conda_env.py

# Print success message
if [ $? -eq 0 ]; then
    echo -e "\nSetup completed successfully!"
    echo "You can activate the environment with: $PACKAGE_MANAGER activate pytorch_docs_search"
else
    echo -e "\nSetup encountered issues. Please check the error messages above."
fi