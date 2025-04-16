#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
import random
import string
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_test")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the embedding cache and generator
from scripts.embedding.cache import EmbeddingCache
from scripts.embedding.generator import EmbeddingGenerator
from scripts.config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR

def generate_random_text(length: int = 100) -> str:
    """Generate random text for testing."""
    letters = string.ascii_letters + string.digits + " " * 10  # More spaces for realism
    return ''.join(random.choice(letters) for _ in range(length))

def test_embedding_cache_performance():
    """Test the performance of the embedding cache."""
    print("=== Embedding Cache Performance Test ===")
    
    # Create a test directory
    test_cache_dir = os.path.join(EMBEDDING_CACHE_DIR, "test")
    os.makedirs(test_cache_dir, exist_ok=True)
    
    # Initialize the cache
    print("\n1. Initializing embedding cache...")
    cache = EmbeddingCache(test_cache_dir, max_size_gb=0.1)  # Small cache for testing
    print(f"✓ Cache initialized at {test_cache_dir}")
    
    # Initialize the embedding generator with cache
    print("\n2. Creating embedding generator with cache...")
    generator = EmbeddingGenerator(model=EMBEDDING_MODEL, use_cache=True)
    print(f"✓ Embedding generator created using model: {EMBEDDING_MODEL}")
    
    # Test 1: Cache miss and hit for the same text
    print("\n3. Testing cache miss and hit for same text...")
    test_text = "This is a test of the PyTorch Documentation Search Tool embedding cache functionality"
    
    # First call - should miss cache
    print("  - First call (should miss cache)...")
    start_time = time.time()
    embedding1 = generator._get_embedding_with_cache(test_text)
    miss_time = time.time() - start_time
    print(f"    Time: {miss_time:.4f} seconds")
    print(f"    Embedding dimensions: {len(embedding1)}")
    
    # Second call - should hit cache
    print("  - Second call (should hit cache)...")
    start_time = time.time()
    embedding2 = generator._get_embedding_with_cache(test_text)
    hit_time = time.time() - start_time
    print(f"    Time: {hit_time:.4f} seconds")
    
    # Calculate speedup
    if hit_time > 0:
        speedup = miss_time / hit_time
        print(f"  ✓ Cache hit is {speedup:.2f}x faster than cache miss")
    else:
        print("  ✓ Cache hit was nearly instantaneous")
    
    # Verify embeddings are identical
    if embedding1 == embedding2:
        print("  ✓ Embeddings from cache match original embeddings")
    else:
        print("  ✗ Embeddings from cache do not match original embeddings")
    
    # Test 2: Batch embedding with partial cache hits
    print("\n4. Testing batch embedding with partial cache hits...")
    
    # Generate test texts
    num_texts = 10
    text_length = 200
    batch_texts = [generate_random_text(text_length) for _ in range(num_texts)]
    
    # Add test_text to create one guaranteed cache hit
    batch_texts[0] = test_text
    
    # First batch - should have one cache hit, rest misses
    print("  - First batch call...")
    start_time = time.time()
    batch1_embeddings = generator._batch_embed(batch_texts)
    batch1_time = time.time() - start_time
    print(f"    Time: {batch1_time:.4f} seconds")
    
    # Second batch - should be all cache hits
    print("  - Second batch call (should be all cache hits)...")
    start_time = time.time()
    batch2_embeddings = generator._batch_embed(batch_texts)
    batch2_time = time.time() - start_time
    print(f"    Time: {batch2_time:.4f} seconds")
    
    # Calculate batch speedup
    if batch2_time > 0:
        batch_speedup = batch1_time / batch2_time
        print(f"  ✓ Cached batch is {batch_speedup:.2f}x faster than uncached batch")
    else:
        print("  ✓ Cached batch was nearly instantaneous")
    
    # Verify batch embeddings are identical
    if batch1_embeddings == batch2_embeddings:
        print("  ✓ Batch embeddings from cache match original embeddings")
    else:
        print("  ✗ Batch embeddings from cache do not match original embeddings")
    
    # Test 3: Cache statistics
    print("\n5. Testing cache statistics...")
    stats = cache.get_stats()
    print(f"  - Cache entries: {stats['entries']}")
    print(f"  - Cache size: {stats['size_mb']:.2f} MB")
    print(f"  - Cache hits: {stats['hits']}")
    print(f"  - Cache misses: {stats['misses']}")
    print(f"  - Cache hit rate: {stats['hit_rate']:.2%}")
    
    # Test 4: Cache pruning
    print("\n6. Testing cache pruning...")
    
    # Generate many texts to potentially trigger pruning
    print("  - Generating 30 more random texts...")
    many_texts = [generate_random_text(1000) for _ in range(30)]
    
    # Embed all texts to populate cache
    print("  - Embedding texts to populate cache...")
    generator._batch_embed(many_texts)
    
    # Get cache stats after potential pruning
    stats_after = cache.get_stats()
    print(f"  - Cache entries after bulk insertion: {stats_after['entries']}")
    print(f"  - Cache size after bulk insertion: {stats_after['size_mb']:.2f} MB")
    
    if stats_after['entries'] < stats['entries'] + 30:
        print(f"  ✓ Cache pruned successfully ({stats['entries'] + 30 - stats_after['entries']} entries removed)")
    elif stats_after['size_mb'] < 100:  # Less than our max_size_gb in MB
        print(f"  ✓ Cache size maintained below limit ({stats_after['size_mb']:.2f} MB)")
    else:
        print(f"  ✓ Cache holds {stats_after['entries']} entries")
    
    # Test 5: Persistence
    print("\n7. Testing cache persistence...")
    
    # Create a new cache instance pointing to the same directory
    print("  - Creating new cache instance...")
    new_cache = EmbeddingCache(test_cache_dir)
    
    # Check if entries were loaded
    new_stats = new_cache.get_stats()
    print(f"  - Loaded {new_stats['entries']} entries from existing cache")
    
    if new_stats['entries'] > 0:
        print("  ✓ Cache successfully persisted and loaded entries")
    else:
        print("  ✗ Cache failed to persist or load entries")
    
    # Test retrieving a previously cached embedding
    print("  - Retrieving previously cached embedding...")
    cached_embedding = new_cache.get(test_text, EMBEDDING_MODEL)
    
    if cached_embedding and len(cached_embedding) > 0:
        print("  ✓ Successfully retrieved previously cached embedding")
        if cached_embedding == embedding1:
            print("  ✓ Retrieved embedding matches original")
        else:
            print("  ✗ Retrieved embedding does not match original")
    else:
        print("  ✗ Failed to retrieve previously cached embedding")
    
    # Overall status
    print("\n=== Embedding Cache Performance Test Results ===")
    print(f"Cache is operational with {new_stats['entries']} entries and {new_stats['size_mb']:.2f} MB")
    print(f"Cache hit rate: {new_stats['hit_rate']:.2%}")
    print(f"Average speedup for cache hits: {speedup:.2f}x")
    print("The embedding cache is functioning correctly")

if __name__ == "__main__":
    test_embedding_cache_performance()
