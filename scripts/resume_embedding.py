#!/usr/bin/env python3
"""
Resume the embedding generation process from where it left off.
This script will:
1. Check which chunks already have embeddings
2. Generate embeddings for the remaining chunks
3. Save the new chunks with embeddings to part files
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embedding.generator import EmbeddingGenerator

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

def main():
    """Resume embedding generation from where it left off."""
    # Set up file paths
    index_file = "./data/pytorch_indexed_chunks.json"
    
    # Load all original chunks
    print(f"Loading all chunks from {index_file}")
    all_chunks = load_chunks(index_file)
    print(f"Total chunks in original file: {len(all_chunks)}")
    
    # Find all existing part files
    part_files = sorted(glob.glob(f"{index_file}.part*"))
    print(f"Found {len(part_files)} existing part files")
    
    # Collect all chunk IDs that already have embeddings
    processed_ids = set()
    for part_file in part_files:
        try:
            chunks = load_chunks(part_file)
            for chunk in chunks:
                processed_ids.add(chunk["id"])
            print(f"  - {part_file}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  - Error loading {part_file}: {e}")
    
    print(f"Found {len(processed_ids)} chunks that already have embeddings")
    
    # Find chunks that still need embeddings
    remaining_chunks = [chunk for chunk in all_chunks if chunk["id"] not in processed_ids]
    print(f"Remaining chunks to process: {len(remaining_chunks)}")
    
    if not remaining_chunks:
        print("All chunks already have embeddings. Nothing to do.")
        return 0
    
    # Initialize embedding generator
    generator = EmbeddingGenerator()
    
    # Process remaining chunks in batches
    batch_size = 500
    last_part_num = len(part_files)
    
    for i in range(0, len(remaining_chunks), batch_size):
        batch_num = i // batch_size + last_part_num + 1
        end_idx = min(i + batch_size, len(remaining_chunks))
        batch_chunks = remaining_chunks[i:end_idx]
        
        print(f"Processing batch {batch_num} ({len(batch_chunks)} chunks)")
        
        # Generate embeddings for this batch
        batch_with_embeddings = generator.embed_chunks(batch_chunks)
        
        # Save this batch to a part file
        part_file = f"{index_file}.part{batch_num}"
        save_chunks(batch_with_embeddings, part_file)
        
        print(f"Completed batch {batch_num}")
    
    print("\nAll batches processed successfully!")
    print(f"You can now run the finalize_embedding.py script to merge all part files and load them into ChromaDB.")
    return 0

if __name__ == "__main__":
    sys.exit(main())