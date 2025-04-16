#!/usr/bin/env python3

import os
import json
import time
import argparse
import numpy as np
import logging
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

from scripts.config import OPENAI_API_KEY, EMBEDDING_MODEL
from scripts.embedding.generator import EmbeddingGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")

# Test queries for code semantics
CODE_QUERIES = [
    "How to implement a custom autograd function in PyTorch",
    "Creating a multi-input neural network with PyTorch",
    "Implementing batch normalization in a custom layer",
    "How to use torch.nn.functional directly",
    "Converting between torch.Tensor and numpy arrays",
    "Implementing custom loss functions in PyTorch",
    "How to use DataLoader with custom Dataset",
    "Saving and loading model checkpoints in PyTorch",
    "Using torch.distributed for multi-GPU training",
    "Implementing a custom optimizer in PyTorch"
]

# Test queries for conceptual understanding
CONCEPT_QUERIES = [
    "What is autograd in PyTorch?",
    "Difference between nn.Module and nn.functional",
    "How backpropagation works in PyTorch",
    "What are hooks in PyTorch?",
    "When to use DataParallel vs DistributedDataParallel",
    "Differences between PyTorch and TensorFlow",
    "Understanding GPU memory management in PyTorch",
    "What is quantization in PyTorch?",
    "How to profile PyTorch models for performance",
    "Explain gradient accumulation in PyTorch"
]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    return np.dot(a, b) / (a_norm * b_norm)

def benchmark_model(model_name: str) -> Dict[str, Any]:
    """Benchmark embedding model on PyTorch-specific queries."""
    generator = EmbeddingGenerator(model=model_name, use_cache=False)
    
    results = {
        "model": model_name,
        "code_similarities": [],
        "concept_similarities": [],
        "cross_similarities": [],
        "timing": {"total": 0, "count": 0}
    }
    
    # Generate embeddings for code queries
    logger.info(f"Generating embeddings for code queries with {model_name}...")
    print(f"Generating embeddings for code queries with {model_name}...")
    start_time = time.time()
    code_embeddings = generator._batch_embed(CODE_QUERIES)
    code_time = time.time() - start_time
    results["timing"]["total"] += code_time
    results["timing"]["count"] += len(CODE_QUERIES)
    logger.info(f"Generated {len(code_embeddings)} code query embeddings in {code_time:.2f} seconds")
    
    # Generate embeddings for concept queries
    logger.info(f"Generating embeddings for concept queries with {model_name}...")
    print(f"Generating embeddings for concept queries with {model_name}...")
    start_time = time.time()
    concept_embeddings = generator._batch_embed(CONCEPT_QUERIES)
    concept_time = time.time() - start_time
    results["timing"]["total"] += concept_time
    results["timing"]["count"] += len(CONCEPT_QUERIES)
    logger.info(f"Generated {len(concept_embeddings)} concept query embeddings in {concept_time:.2f} seconds")
    
    # Calculate similarities within code queries
    for i in range(len(CODE_QUERIES)):
        for j in range(i+1, len(CODE_QUERIES)):
            sim = cosine_similarity(code_embeddings[i], code_embeddings[j])
            results["code_similarities"].append({
                "query1": CODE_QUERIES[i],
                "query2": CODE_QUERIES[j],
                "similarity": sim
            })
    
    # Calculate similarities within concept queries
    for i in range(len(CONCEPT_QUERIES)):
        for j in range(i+1, len(CONCEPT_QUERIES)):
            sim = cosine_similarity(concept_embeddings[i], concept_embeddings[j])
            results["concept_similarities"].append({
                "query1": CONCEPT_QUERIES[i],
                "query2": CONCEPT_QUERIES[j],
                "similarity": sim
            })
    
    # Calculate cross-domain similarities
    for i in range(len(CODE_QUERIES)):
        for j in range(len(CONCEPT_QUERIES)):
            sim = cosine_similarity(code_embeddings[i], concept_embeddings[j])
            results["cross_similarities"].append({
                "code_query": CODE_QUERIES[i],
                "concept_query": CONCEPT_QUERIES[j],
                "similarity": sim
            })
    
    # Add summary statistics
    results["summary"] = {
        "avg_code_similarity": np.mean([item["similarity"] for item in results["code_similarities"]]),
        "avg_concept_similarity": np.mean([item["similarity"] for item in results["concept_similarities"]]),
        "avg_cross_similarity": np.mean([item["similarity"] for item in results["cross_similarities"]]),
        "avg_embedding_time": results["timing"]["total"] / results["timing"]["count"],
        "dimensions": len(code_embeddings[0]) if code_embeddings and len(code_embeddings) > 0 else 0
    }
    
    return results

def find_notable_similarities(results: Dict[str, Any]) -> Dict[str, Any]:
    """Find highest and lowest similarity pairs for analysis."""
    code_similarities = results["code_similarities"]
    concept_similarities = results["concept_similarities"]
    cross_similarities = results["cross_similarities"]
    
    # Find highest and lowest similarity pairs within code queries
    sorted_code = sorted(code_similarities, key=lambda x: x["similarity"], reverse=True)
    highest_code = sorted_code[:3]
    lowest_code = sorted_code[-3:]
    
    # Find highest and lowest similarity pairs within concept queries
    sorted_concept = sorted(concept_similarities, key=lambda x: x["similarity"], reverse=True)
    highest_concept = sorted_concept[:3]
    lowest_concept = sorted_concept[-3:]
    
    # Find highest and lowest cross-domain similarities
    sorted_cross = sorted(cross_similarities, key=lambda x: x["similarity"], reverse=True)
    highest_cross = sorted_cross[:3]
    lowest_cross = sorted_cross[-3:]
    
    return {
        "most_similar_code_queries": highest_code,
        "least_similar_code_queries": lowest_code,
        "most_similar_concept_queries": highest_concept,
        "least_similar_concept_queries": lowest_concept,
        "most_similar_cross_domain": highest_cross,
        "least_similar_cross_domain": lowest_cross
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models for PyTorch documentation search")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["text-embedding-ada-002", "text-embedding-3-large"],
                        help="Models to benchmark")
    parser.add_argument("--output", type=str, default="./data/benchmark_results.json",
                        help="Output file for benchmark results")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    all_results = {}
    
    for model in args.models:
        print(f"\n=== Benchmarking {model} ===")
        results = benchmark_model(model)
        all_results[model] = results
        
        # Find notable similarities
        notable = find_notable_similarities(results)
        all_results[model]["notable_pairs"] = notable
        
        # Print summary
        summary = results["summary"]
        print(f"\nResults for {model}:")
        print(f"  Vector dimensions: {summary['dimensions']}")
        print(f"  Average code similarity: {summary['avg_code_similarity']:.4f}")
        print(f"  Average concept similarity: {summary['avg_concept_similarity']:.4f}")
        print(f"  Average cross-domain similarity: {summary['avg_cross_similarity']:.4f}")
        print(f"  Average embedding time: {summary['avg_embedding_time']:.4f} seconds")
        
        # Print most similar code queries
        print("\nMost similar code queries:")
        for pair in notable["most_similar_code_queries"]:
            print(f"  {pair['similarity']:.4f}: \"{pair['query1']}\" and \"{pair['query2']}\"")
    
    # Compare models
    if len(args.models) > 1:
        print("\n=== Model Comparison ===")
        for metric in ["avg_code_similarity", "avg_concept_similarity", "avg_cross_similarity", "avg_embedding_time"]:
            print(f"\n{metric}:")
            for model in args.models:
                value = all_results[model]["summary"][metric]
                print(f"  {model}: {value:.4f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBenchmark results saved to {args.output}")

if __name__ == "__main__":
    main()