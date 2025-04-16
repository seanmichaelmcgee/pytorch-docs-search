#!/usr/bin/env python3

import json
import subprocess
import sys

# Query to search for
query = "How to implement batch normalization in PyTorch"

# Prepare input data
input_data = {
    "query": query
}

# Convert to JSON
input_json = json.dumps(input_data)

print(f"Input JSON: {input_json}")

# Run the tool
try:
    result = subprocess.run(
        ["python", "../scripts/claude-code-tool.py"],
        input=input_json,
        text=True,
        capture_output=True
    )
    
    print(f"Return code: {result.returncode}")
    
    if result.returncode == 0:
        print("\nOutput:")
        print(result.stdout)
    else:
        print("\nError:")
        print(result.stderr)
except Exception as e:
    print(f"Error: {str(e)}")