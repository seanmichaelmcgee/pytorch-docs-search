#!/usr/bin/env python3
"""Test the new environment setup."""

import os
import sys
import numpy as np
import torch
import chromadb

def test_environment():
    """Test that the environment is set up correctly."""
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ChromaDB version: {chromadb.__version__}")
    
    # Test ChromaDB
    try:
        client = chromadb.Client()
        print("ChromaDB client created successfully")
        
        # Try to create a collection
        collection = client.create_collection("test_collection")
        print("ChromaDB collection created successfully")
        
        # Try to add a document
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["test1"]
        )
        print("Document added to collection successfully")
        
        # Try a simple query
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        print("Query successful")
        print(f"Query results: {results}")
        
        # Clean up
        client.delete_collection("test_collection")
        print("Collection deleted successfully")
    
    except Exception as e:
        print(f"ChromaDB test failed: {e}")
    
    # Test PyTorch
    try:
        # Create a simple tensor
        x = torch.rand(3, 3)
        print(f"PyTorch tensor created: {x}")
        
        # Check if CUDA is available
        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"PyTorch test failed: {e}")
    
    print("Environment test completed successfully")

if __name__ == "__main__":
    test_environment()