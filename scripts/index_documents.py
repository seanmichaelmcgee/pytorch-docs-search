#!/usr/bin/env python3

import os
import json
import glob
from tqdm import tqdm
from typing import List, Dict, Any
import argparse

# Import our modules
from scripts.document_processing.parser import DocumentParser
from scripts.document_processing.chunker import DocumentChunker
from scripts.config import CHUNK_SIZE, OVERLAP_SIZE, INDEXED_FILE

def process_documents(docs_dir: str, output_file: str = INDEXED_FILE) -> List[Dict[str, Any]]:
    """Process all documents in the directory and save chunks."""
    # Initialize our components
    parser = DocumentParser()
    chunker = DocumentChunker(CHUNK_SIZE, OVERLAP_SIZE)
    
    # Find all markdown and Python files
    file_patterns = ['**/*.md', '**/*.markdown', '**/*.py']
    all_files = []
    
    for pattern in file_patterns:
        matched_files = glob.glob(os.path.join(docs_dir, pattern), recursive=True)
        all_files.extend(matched_files)
    
    print(f"Found {len(all_files)} documentation files to process")
    
    # Process each file
    all_chunks = []
    
    for filepath in tqdm(all_files, desc="Processing documents"):
        # Parse the file into sections
        sections = parser.parse_file(filepath)
        
        # Chunk the sections
        chunks = chunker.process_sections(sections)
        
        # Add to our collection
        all_chunks.extend(chunks)
    
    print(f"Generated {len(all_chunks)} chunks from {len(all_files)} files")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the chunks to a file (without embeddings for now)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Saved chunks to {output_file}")
    
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PyTorch documentation files into chunks")
    parser.add_argument("--docs-dir", type=str, default="./pytorch_docs",
                        help="Directory containing documentation files")
    parser.add_argument("--output-file", type=str, default=INDEXED_FILE,
                        help="Output JSON file to save chunks")
    args = parser.parse_args()
    
    chunks = process_documents(args.docs_dir, args.output_file)
    print(f"Successfully processed {len(chunks)} chunks")