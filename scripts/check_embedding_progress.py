#!/usr/bin/env python3
"""
Check the progress of the embedding generation process.
This script only monitors the status without doing any processing or finalization.
"""

import os
import sys
import json
import glob
from pathlib import Path

def load_chunks(file_path):
    """Load chunks from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Check the progress of embedding generation."""
    # Set up file paths
    index_file = "./data/pytorch_indexed_chunks.json"
    merged_file = "./data/pytorch_indexed_chunks.json.merged"
    
    # Load count of original chunks
    try:
        print(f"Loading all chunks from {index_file}")
        all_chunks = load_chunks(index_file)
        total_chunks = len(all_chunks)
        print(f"Total chunks in original file: {total_chunks}")
    except Exception as e:
        print(f"Error loading original chunks: {e}")
        return 1
    
    # Load previously processed chunks
    try:
        previously_processed = load_chunks(merged_file)
        print(f"Previously processed chunks: {len(previously_processed)}")
    except Exception as e:
        print(f"No previously processed chunks found: {e}")
        previously_processed = []
    
    # Find all part files and count chunks
    part_files = sorted(glob.glob(f"{index_file}.part*"))
    print(f"Found {len(part_files)} part files")
    
    # Count chunks in each part file
    all_processed_chunks = len(previously_processed)
    new_part_chunks = 0
    
    for part_file in part_files:
        try:
            chunks = load_chunks(part_file)
            chunk_count = len(chunks)
            print(f"  - {part_file}: {chunk_count} chunks")
            
            # Only count chunks from new part files (part5 and above)
            part_num = int(part_file.split("part")[-1])
            if part_num >= 5:
                new_part_chunks += chunk_count
        except Exception as e:
            print(f"  - Error loading {part_file}: {e}")
    
    all_processed_chunks += new_part_chunks
    print(f"\nTotal processed chunks so far: {all_processed_chunks} / {total_chunks}")
    print(f"Progress: {all_processed_chunks/total_chunks:.1%}")
    
    remaining = total_chunks - all_processed_chunks
    if remaining > 0:
        print(f"Remaining chunks: {remaining}")
        print("The embedding generation process is still running.")
        print("Check back later to see the progress.")
    else:
        print("All chunks have been processed!")
        print("When you're ready to finalize, you can run a script to merge all part files and load everything into ChromaDB.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())