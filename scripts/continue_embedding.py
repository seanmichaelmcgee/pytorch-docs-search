#!/usr/bin/env python3
"""
Continue generating embeddings for the remaining PyTorch documentation chunks.
This script will:
1. Load all chunks from the original indexed file
2. Load the chunks that already have embeddings from the merged file
3. Generate embeddings for the remaining chunks
4. Save the new chunks with embeddings to part files
5. Merge all part files including the new ones
6. Load all chunks into ChromaDB
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embedding.generator import EmbeddingGenerator
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
    import glob
    
    print(f"Merging all part files for {base_file}")
    
    # Find all part files including the new ones
    part_files = sorted(glob.glob(f"{base_file}.part*"))
    if not part_files:
        print("No part files found to merge.")
        return None
    
    # Load all chunks from all part files
    all_chunks = []
    for part_file in part_files:
        print(f"Loading chunks from {part_file}")
        all_chunks.extend(load_chunks(part_file))
    
    print(f"Total chunks loaded: {len(all_chunks)}")
    
    # Save to the final merged file
    merged_file = f"{base_file}.merged"
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
    """Main function to continue the embedding generation process."""
    # Set up file paths
    index_file = "./data/pytorch_indexed_chunks.json"
    merged_file = "./data/pytorch_indexed_chunks.json.merged"
    db_dir = "./data/chroma_db"
    collection_name = "pytorch_docs"
    
    # Load all original chunks
    print(f"Loading all chunks from {index_file}")
    all_chunks = load_chunks(index_file)
    print(f"Total chunks in original file: {len(all_chunks)}")
    
    # Load chunks that already have embeddings
    print(f"Loading chunks with existing embeddings from {merged_file}")
    processed_chunks = load_chunks(merged_file)
    print(f"Chunks with existing embeddings: {len(processed_chunks)}")
    
    # Extract IDs of chunks that already have embeddings
    processed_ids = set(chunk["id"] for chunk in processed_chunks)
    
    # Find chunks that need embeddings
    remaining_chunks = [chunk for chunk in all_chunks if chunk["id"] not in processed_ids]
    print(f"Remaining chunks to process: {len(remaining_chunks)}")
    
    if not remaining_chunks:
        print("All chunks already have embeddings. Nothing to do.")
        return 0
    
    # Initialize embedding generator
    generator = EmbeddingGenerator()
    
    # Process remaining chunks in batches
    batch_size = 500
    start_part = len(processed_chunks) // batch_size + 1
    
    for i in range(0, len(remaining_chunks), batch_size):
        batch_num = i // batch_size + start_part
        end_idx = min(i + batch_size, len(remaining_chunks))
        batch_chunks = remaining_chunks[i:end_idx]
        
        print(f"Processing batch {batch_num} ({len(batch_chunks)} chunks)")
        
        # Generate embeddings for this batch
        batch_with_embeddings = generator.embed_chunks(batch_chunks)
        
        # Save this batch to a part file
        part_file = f"{index_file}.part{batch_num}"
        save_chunks(batch_with_embeddings, part_file)
        
        print(f"Completed batch {batch_num}")
    
    # Merge all part files including the new ones
    print("\nAll batches processed. Merging all part files...")
    merged_file = merge_part_files(index_file)
    
    if merged_file:
        # Load all chunks into ChromaDB
        print("\nLoading all chunks into ChromaDB...")
        load_into_chromadb(merged_file, db_dir, collection_name)
        
        print("\nProcess completed successfully!")
        print(f"All PyTorch documentation chunks have been processed and loaded into ChromaDB.")
        return 0
    else:
        print("Failed to merge part files.")
        return 1

if __name__ == "__main__":
    sys.exit(main())