#!/usr/bin/env python3
"""
Check the status of the ChromaDB database.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.database.chroma_manager import ChromaManager

def main():
    """Check the status of the ChromaDB database."""
    db_dir = "./data/chroma_db"
    collections = ["pytorch_docs", "betaband_docs"]
    
    for collection_name in collections:
        try:
            manager = ChromaManager(db_dir, collection_name)
            stats = manager.get_stats()
            
            print(f"\nCollection: {collection_name}")
            print("  Collection statistics:")
            print(f"    Total chunks: {stats['total_chunks']}")
            
            print("    Chunk types:")
            for chunk_type, count in stats['chunk_types'].items():
                print(f"      - {chunk_type}: {count}")
            
            print("    Top 10 sources:")
            for i, (source, count) in enumerate(sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:10]):
                print(f"      - {source}: {count}")
            
            source_count = len(stats['sources'])
            if source_count > 10:
                print(f"      - ... and {source_count - 10} more sources")
                
        except Exception as e:
            print(f"Error accessing collection {collection_name}: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())