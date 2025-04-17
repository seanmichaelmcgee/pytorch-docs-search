#!/bin/bash

# This script activates the Conda environment and runs the test script

# Deactivate any active environment
conda deactivate 2>/dev/null || true

# Get conda executable path
CONDA_BASE=$(conda info --base)
echo "Conda base: $CONDA_BASE"

# Source conda.sh to enable conda command
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the pytorch_docs_search environment
echo "Activating pytorch_docs_search environment..."
conda activate pytorch_docs_search

# Run the test script
echo "Running test script..."
python test_conda_env.py

# Store the exit code
exit_code=$?

# Deactivate the conda environment
echo "Deactivating environment..."
conda deactivate

# Exit with the same code as the test script
echo "Tests completed with exit code: $exit_code"
exit $exit_code