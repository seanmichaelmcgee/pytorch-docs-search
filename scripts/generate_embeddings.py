#!/usr/bin/env python3

import os
import argparse
from scripts.embedding.generator import EmbeddingGenerator
from scripts.config import INDEXED_FILE

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for document chunks")
    parser.add_argument("--input-file", type=str, default=INDEXED_FILE,
                        help="Input JSON file with document chunks")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output JSON file to save chunks with embeddings (defaults to overwriting input)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Batch size for embedding generation")
    args = parser.parse_args()
    
    # Default to overwriting the input file if no output file is specified
    output_file = args.output_file if args.output_file else args.input_file
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    chunks = generator.process_chunks_file(args.input_file, output_file)
    
    print(f"Successfully generated embeddings for {len(chunks)} chunks")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()