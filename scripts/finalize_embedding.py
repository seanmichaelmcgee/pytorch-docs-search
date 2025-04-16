#!/usr/bin/env python3
"""
Finalize the embedding process by:
1. Merging all part files into a single file
2. Loading all chunks into ChromaDB
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.database.chroma_manager import ChromaManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chunks(file_path):
    """Load chunks from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_chunks(chunks, file_path):
    """Save chunks to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {file_path}")
    print(f"Saved {len(chunks)} chunks to {file_path}")

def merge_part_files(base_file):
    """Merge all part files into a single file."""
    print(f"Merging all part files for {base_file}")
    
    # Find all part files including the new ones
    part_files = sorted(glob.glob(f"{base_file}.part*"))
    if not part_files:
        print("No part files found to merge.")
        return None
    
    print(f"Found {len(part_files)} part files")
    
    # Load all chunks from all part files
    all_chunks = []
    for part_file in part_files:
        print(f"Loading chunks from {part_file}")
        try:
            chunks = load_chunks(part_file)
            all_chunks.extend(chunks)
            print(f"  - {len(chunks)} chunks loaded")
        except Exception as e:
            print(f"Error loading {part_file}: {e}")
    
    print(f"Total chunks loaded: {len(all_chunks)}")
    
    # Save to the final merged file
    merged_file = f"{base_file}.merged.full"
    save_chunks(all_chunks, merged_file)
    
    print(f"Successfully merged all parts to {merged_file}")
    return merged_file

def load_into_chromadb(input_file, db_dir, collection_name):
    """Load all chunks into ChromaDB."""
    print(f"Loading all chunks from {input_file} into ChromaDB collection {collection_name}")
    
    # Initialize ChromaDB manager
    db_manager = ChromaManager(db_dir, collection_name)
    
    # Reset the collection and load all chunks
    db_manager.load_from_file(input_file, reset=True)
    
    # Verify database status
    stats = db_manager.get_stats()
    print("\nCollection Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    
    print("  Chunk types:")
    for chunk_type, count in stats['chunk_types'].items():
        print(f"    - {chunk_type}: {count}")
    
    print("  Top 10 sources:")
    for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {source}: {count}")
    if len(stats['sources']) > 10:
        print(f"    - ... and {len(stats['sources']) - 10} more sources")

def main():
    """Finalize the embedding process."""
    # Set up file paths
    index_file = "./data/pytorch_indexed_chunks.json"
    db_dir = "./data/chroma_db"
    collection_name = "pytorch_docs"
    
    # Check if all chunks have been processed
    all_chunks = load_chunks(index_file)
    total_chunks = len(all_chunks)
    
    part_files = sorted(glob.glob(f"{index_file}.part*"))
    processed_chunks = 0
    for part_file in part_files:
        try:
            chunks = load_chunks(part_file)
            processed_chunks += len(chunks)
        except Exception:
            pass
    
    print(f"Total processed chunks: {processed_chunks} / {total_chunks}")
    
    if processed_chunks < total_chunks:
        print("Warning: Not all chunks have been processed yet.")
        proceed = input("Do you want to proceed anyway? (y/N): ")
        if proceed.lower() != 'y':
            print("Aborting.")
            return 1
    
    # Merge all part files into a single file
    merged_file = merge_part_files(index_file)
    
    if merged_file:
        # Load all chunks into ChromaDB
        load_into_chromadb(merged_file, db_dir, collection_name)
        
        print("\nProcess completed successfully!")
        print(f"All {processed_chunks} PyTorch documentation chunks have been loaded into ChromaDB.")
        return 0
    else:
        print("Failed to merge part files.")
        return 1

if __name__ == "__main__":
    sys.exit(main())