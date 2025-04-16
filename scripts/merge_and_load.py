#!/usr/bin/env python3
"""
Merge part files and load embeddings into ChromaDB.
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path

def merge_part_files(base_file):
    """Merge part files into a single final file."""
    print(f"Merging part files for {base_file}")
    
    part_files = sorted(glob.glob(f"{base_file}.part*"))
    if not part_files:
        print("No part files found to merge.")
        return False
    
    # Load all chunks from all part files
    all_chunks = []
    for part_file in part_files:
        print(f"Loading chunks from {part_file}")
        with open(part_file, 'r') as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    
    print(f"Total chunks loaded: {len(all_chunks)}")
    
    # Save to the final file
    merged_file = f"{base_file}.merged"
    with open(merged_file, 'w') as f:
        json.dump(all_chunks, f)
    
    print(f"Successfully merged all parts to {merged_file}")
    return merged_file

def load_into_chromadb(input_file, db_dir, collection_name):
    """Load the embeddings into ChromaDB."""
    print(f"Loading embeddings from {input_file} into ChromaDB collection {collection_name}")
    
    # Use subprocess to call the script
    cmd = [
        sys.executable, "-m", "scripts.load_to_database",
        "--input-file", input_file,
        "--db-dir", db_dir,
        "--collection", collection_name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("ChromaDB Load Output:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("Error loading into ChromaDB:")
        print(result.stderr)
        return False
    
    return True

def main():
    """Main function."""
    # Set up the file paths
    pytorch_file = "./data/pytorch_indexed_chunks.json"
    db_dir = "./data/chroma_db"
    collection_name = "pytorch_docs"
    
    # Merge part files
    merged_file = merge_part_files(pytorch_file)
    if merged_file:
        # Load into ChromaDB
        success = load_into_chromadb(merged_file, db_dir, collection_name)
        if success:
            print("Successfully loaded PyTorch documentation embeddings into ChromaDB!")
            
            # Verify database status
            print("\nVerifying database status...")
            cmd = [
                sys.executable, "-c",
                "from scripts.database.chroma_manager import ChromaManager; "
                f"manager = ChromaManager('{db_dir}', '{collection_name}'); "
                "print('Collection statistics:'); "
                "stats = manager.get_stats(); "
                "print(f'  Total chunks: {stats[\"total_chunks\"]}'); "
                "print(f'  Chunk types: {stats[\"chunk_types\"]}'); "
                "print('  Top 5 sources:'); "
                "for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:5]: "
                "    print(f'    - {source}: {count}')"
            ]
            subprocess.run(cmd)
            
            return 0
        else:
            print("Failed to load embeddings into ChromaDB.")
            return 1
    else:
        print("Failed to merge part files.")
        return 1

if __name__ == "__main__":
    sys.exit(main())