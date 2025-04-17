#!/bin/bash

# This script creates and activates a Conda environment for the PyTorch Docs Search project
# using the environment.yml file

# Exit immediately if a command exits with a non-zero status
set -e

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda before running this script."
    exit 1
fi

# Source conda.sh to enable conda command
source "$(conda info --base)/etc/profile.d/conda.sh"

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
        conda env update -f environment.yml
    else
        echo "Skipping environment update."
    fi
else
    echo "Creating new environment from environment.yml..."
    conda env create -f environment.yml
fi

# Activate the environment
conda activate pytorch_docs_search

# Verify installation by running the test script
echo -e "\nVerifying environment setup..."
python test_conda_env.py

# Print success message
if [ $? -eq 0 ]; then
    echo -e "\nSetup completed successfully!"
    echo "You can activate the environment with: conda activate pytorch_docs_search"
else
    echo -e "\nSetup encountered issues. Please check the error messages above."
fi