#!/usr/bin/env python3

import argparse
import os
from scripts.database.chroma_manager import ChromaManager
from scripts.config import INDEXED_FILE, DB_DIR, COLLECTION_NAME

def main():
    parser = argparse.ArgumentParser(description="Load document chunks into ChromaDB")
    parser.add_argument("--input-file", type=str, default=INDEXED_FILE,
                        help="Input JSON file with document chunks and embeddings")
    parser.add_argument("--db-dir", type=str, default=DB_DIR,
                        help="Directory for ChromaDB storage")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME,
                        help="Name of the ChromaDB collection")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for loading data")
    parser.add_argument("--no-reset", action="store_true",
                        help="Don't reset the collection before loading")
    args = parser.parse_args()
    
    # Initialize ChromaDB manager
    db_manager = ChromaManager(args.db_dir, args.collection)
    
    # Load chunks into database
    db_manager.load_from_file(
        args.input_file,
        reset=not args.no_reset,
        batch_size=args.batch_size
    )
    
    # Display collection statistics
    stats = db_manager.get_stats()
    print("\nCollection Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    
    print("  Chunk types:")
    for chunk_type, count in stats['chunk_types'].items():
        print(f"    - {chunk_type}: {count}")
    
    print("  Sources:")
    for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {source}: {count}")
    if len(stats['sources']) > 10:
        print(f"    - ... and {len(stats['sources']) - 10} more sources")

if __name__ == "__main__":
    main()