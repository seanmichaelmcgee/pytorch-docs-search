#!/usr/bin/env python3

import os
import json
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Any

from scripts.embedding.generator import EmbeddingGenerator
from scripts.config import EMBEDDING_MODEL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migration")

def migrate_embeddings(input_file: str, output_file: str, batch_size: int = 10):
    """Migrate embeddings from previous model to new model.
    
    Args:
        input_file: JSON file with chunks using old embeddings
        output_file: Where to save chunks with new embeddings
        batch_size: Batch size for embedding generation
    """
    logger.info(f"Loading chunks from {input_file}...")
    print(f"Loading chunks from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks with embeddings from old model")
    print(f"Loaded {len(chunks)} chunks with embeddings from old model")
    
    # Check if embeddings exist and get dimensionality
    if "embedding" in chunks[0]:
        old_dimensions = len(chunks[0]["embedding"])
        logger.info(f"Old embedding dimensions: {old_dimensions}")
        print(f"Old embedding dimensions: {old_dimensions}")
    
    # Create embedding generator with the new model and caching enabled
    generator = EmbeddingGenerator(model=EMBEDDING_MODEL)
    
    # Re-embed all chunks
    logger.info(f"Generating new embeddings with {EMBEDDING_MODEL}...")
    print(f"Generating new embeddings with {EMBEDDING_MODEL}...")
    updated_chunks = generator.embed_chunks(chunks, batch_size)
    
    # Save updated chunks
    logger.info(f"Saving {len(updated_chunks)} chunks with new embeddings to {output_file}")
    print(f"Saving {len(updated_chunks)} chunks with new embeddings to {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_chunks, f)
    
    # If output dimensions differ, report it
    if "embedding" in updated_chunks[0]:
        new_dimensions = len(updated_chunks[0]["embedding"])
        logger.info(f"New embedding dimensions: {new_dimensions}")
        print(f"New embedding dimensions: {new_dimensions}")
        
        if old_dimensions != new_dimensions:
            logger.info(f"Dimensionality changed: {old_dimensions} → {new_dimensions}")
            print(f"Dimensionality changed: {old_dimensions} → {new_dimensions}")
    
    logger.info(f"Migration complete. New embeddings saved to {output_file}")
    print(f"Migration complete. New embeddings saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Migrate embeddings to new model")
    parser.add_argument("--input-file", required=True, type=str,
                        help="Input JSON file with old embeddings")
    parser.add_argument("--output-file", required=True, type=str,
                        help="Output JSON file for new embeddings")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for embedding generation")
    args = parser.parse_args()
    
    migrate_embeddings(args.input_file, args.output_file, args.batch_size)

if __name__ == "__main__":
    main()