#!/usr/bin/env python3
"""
Continue loading the remaining PyTorch documentation chunks into ChromaDB.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.database.chroma_manager import ChromaManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_chunks_in_file(filepath):
    """Count the number of chunks in a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return len(chunks)

def main():
    """Continue loading PyTorch documentation chunks into ChromaDB."""
    # Set up the file paths
    merged_file = "./data/pytorch_indexed_chunks.json.merged"
    db_dir = "./data/chroma_db"
    collection_name = "pytorch_docs"
    
    # Verify the merged file exists
    if not os.path.exists(merged_file):
        print(f"Error: Merged file {merged_file} not found.")
        return 1
    
    # Initialize ChromaDB manager
    manager = ChromaManager(db_dir, collection_name)
    
    # Get current database stats
    db_stats = manager.get_stats()
    current_chunks = db_stats['total_chunks']
    print(f"\nCurrent database status:")
    print(f"  Collection: {collection_name}")
    print(f"  Total chunks loaded: {current_chunks}")
    
    # Count total chunks in the merged file
    total_chunks = count_chunks_in_file(merged_file)
    print(f"  Total chunks in merged file: {total_chunks}")
    
    if current_chunks >= total_chunks:
        print("All chunks are already loaded into the database.")
        return 0
    
    # Load the chunks from the merged file
    with open(merged_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    # Skip the chunks that are already loaded
    remaining_chunks = all_chunks[current_chunks:]
    print(f"Loading remaining {len(remaining_chunks)} chunks into ChromaDB...")
    
    # Load the remaining chunks without resetting the collection
    manager.add_chunks(remaining_chunks)
    
    # Verify database status after loading
    db_stats = manager.get_stats()
    final_chunks = db_stats['total_chunks']
    print(f"\nFinal database status:")
    print(f"  Collection: {collection_name}")
    print(f"  Total chunks loaded: {final_chunks}")
    print(f"  Chunk types:")
    for chunk_type, count in db_stats['chunk_types'].items():
        print(f"    - {chunk_type}: {count}")
    
    print(f"  Top 10 sources:")
    for source, count in sorted(db_stats['sources'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {source}: {count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())