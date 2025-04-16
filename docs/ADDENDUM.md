# PyTorch Documentation Search Tool
## Embedding Model Update Addendum

This addendum documents the changes required to implement the PyTorch Documentation Search Tool using OpenAI's `text-embedding-3-large` model instead of `text-embedding-ada-002`. The changes outlined here can be applied at any stage of the implementation pipeline.

## 1. Implementation Plan Updates (GUIDE.md)

### Configuration Changes
- Update `EMBEDDING_MODEL` to "text-embedding-3-large"
- Adjust batch size from 20 to 10 in embedding generation to manage memory and API rate limits
- Update vector dimensionality expectations from 1536 to 3072

### System Requirements
- Increase minimum RAM recommendation to 8GB (from 4GB)
- Increase storage requirements for vector database (approximately double for the same number of documents)
- Allocate additional disk space for embedding cache

### New Component: Embedding Cache
- Add a disk-based cache to prevent regenerating identical embeddings
- Implement LRU (Least Recently Used) strategy for cache management
- Store embeddings with content hash as key for quick lookup

## 2. Product Requirements Document Updates (PRD.md)

### Technical Requirements (Section 6.1)
- Update embedding model specification to text-embedding-3-large
- Adjust hardware requirements to reflect increased memory needs
- Add reference to embedding cache component in system architecture

### Cost Considerations (Section 7.2)
- Note increased API costs for the enhanced embedding model
- Add cost optimization strategies including caching and efficient chunking
- Reference approximately 3x cost increase compared to text-embedding-ada-002

### Performance Expectations (Section 8.1)
- Adjust search quality expectations upward (95% relevant results in top 3, up from 90%)
- Adjust query response time expectation to < 2.5 seconds (from < 2 seconds)
- Add expected embedding cache hit rate of > 40% after initial corpus processing

## 3. Implementation Timeline Updates (Timeline.md)

### Phase 3: Embedding Generation
- Add task: "Implement embedding caching mechanism"
- Add task: "Benchmark embedding performance and quality"
- Add task: "Optimize batch sizes for memory efficiency"
- Extend phase duration by 1-2 days to accommodate additional work

### Phase 6: Optimization
- Add task: "Optimize for larger embedding dimensions"
- Add task: "Implement storage efficiency measures"
- Add task: "Tune ChromaDB for large vector dimensions"

## 4. Benefits of Model Upgrade

- Superior code semantics understanding
- Better performance on technical PyTorch queries
- Improved differentiation between similar code patterns
- Enhanced comprehension of technical documentation
- More accurate detection of conceptual relationships in technical material
- Better handling of API-related queries and implementation patterns

## 5. Implementation Guidance

### Configuration Updates

Update your configuration module to reflect the new embedding model:

```python
# scripts/config/__init__.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"  # Updated from "text-embedding-ada-002"
EMBEDDING_DIMENSIONS = 3072  # Updated from 1536
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "200"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
DB_DIR = os.getenv("DB_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pytorch_docs")
INDEXED_FILE = "./data/indexed_chunks.json"
EMBEDDING_CACHE_DIR = "./data/embedding_cache"
EMBEDDING_BATCH_SIZE = 10  # Reduced from 20
```

### Embedding Cache Implementation

Create a new cache module for managing embeddings:

```python
# scripts/embedding/cache.py
import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_cache")

class EmbeddingCache:
    def __init__(self, cache_dir: str, max_size_gb: float = 1.0):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_gb: Maximum cache size in gigabytes
        """
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.index_file = os.path.join(cache_dir, "index.json")
        self.index = {}
        self.stats = {"hits": 0, "misses": 0}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load index if it exists
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                logger.info(f"Loaded embedding cache index with {len(self.index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache index: {e}")
                self.index = {}
    
    def _get_hash(self, text: str, model: str) -> str:
        """Generate a hash for the text and model combination."""
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.sha256(text_bytes)
        hash_obj.update(model.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache if it exists."""
        text_hash = self._get_hash(text, model)
        
        if text_hash in self.index:
            cache_file = os.path.join(self.cache_dir, text_hash + ".json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    # Update access time
                    self.index[text_hash]["last_access"] = time.time()
                    self._save_index()
                    
                    self.stats["hits"] += 1
                    return data["embedding"]
                except Exception as e:
                    logger.warning(f"Failed to load embedding from cache: {e}")
        
        self.stats["misses"] += 1
        return None
    
    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        text_hash = self._get_hash(text, model)
        cache_file = os.path.join(self.cache_dir, text_hash + ".json")
        
        try:
            # Save embedding to file
            with open(cache_file, 'w') as f:
                json.dump({"text": text[:100] + "..." if len(text) > 100 else text,
                           "model": model,
                           "embedding": embedding}, f)
            
            # Update index
            self.index[text_hash] = {
                "file": text_hash + ".json",
                "size": os.path.getsize(cache_file),
                "last_access": time.time(),
                "created": time.time()
            }
            
            # Save index
            self._save_index()
            
            # Check cache size and prune if necessary
            self._prune_cache_if_needed()
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")
    
    def _save_index(self) -> None:
        """Save the index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache index: {e}")
    
    def _prune_cache_if_needed(self) -> None:
        """Remove oldest entries if cache exceeds maximum size."""
        total_size = sum(entry["size"] for entry in self.index.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort entries by last access time
        sorted_entries = sorted(
            [(k, v) for k, v in self.index.items()],
            key=lambda x: x[1]["last_access"]
        )
        
        # Remove oldest entries until under size limit
        bytes_to_remove = total_size - self.max_size_bytes
        bytes_removed = 0
        removed_count = 0
        
        for text_hash, entry in sorted_entries:
            if bytes_removed >= bytes_to_remove:
                break
            
            cache_file = os.path.join(self.cache_dir, entry["file"])
            if os.path.exists(cache_file):
                bytes_removed += entry["size"]
                os.remove(cache_file)
                del self.index[text_hash]
                removed_count += 1
        
        logger.info(f"Pruned embedding cache: removed {removed_count} entries ({bytes_removed / 1024 / 1024:.2f} MB)")
        self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry["size"] for entry in self.index.values())
        
        return {
            "entries": len(self.index),
            "size_mb": total_size / 1024 / 1024,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
        }
```

### Updated Embedding Generator

Modify your embedding generator to use the cache and handle the new model:

```python
# scripts/embedding/generator.py
import os
import json
import openai
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE, EMBEDDING_CACHE_DIR
from embedding.cache import EmbeddingCache

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

class EmbeddingGenerator:
    def __init__(self, model: str = EMBEDDING_MODEL, use_cache: bool = True):
        """Initialize the embedding generator."""
        self.model = model
        self.cache = EmbeddingCache(EMBEDDING_CACHE_DIR) if use_cache else None
    
    def _get_embedding_with_cache(self, text: str) -> List[float]:
        """Get embedding for a single text, using cache if available."""
        if self.cache:
            cached_embedding = self.cache.get(text, self.model)
            if cached_embedding:
                return cached_embedding
        
        # Generate embedding via API
        response = openai.Embedding.create(
            input=text,
            model=self.model
        )
        embedding = response["data"][0]["embedding"]
        
        # Cache the result
        if self.cache:
            self.cache.set(text, self.model, embedding)
        
        return embedding
    
    def _batch_embed(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache first
            if self.cache:
                for j, text in enumerate(batch_texts):
                    cached_embedding = self.cache.get(text, self.model)
                    if cached_embedding:
                        batch_embeddings.append(cached_embedding)
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)
            else:
                uncached_texts = batch_texts
                uncached_indices = list(range(len(batch_texts)))
            
            # If there are texts not in cache, generate embeddings
            if uncached_texts:
                # Retry mechanism for API calls
                max_retries = 5
                retry_delay = 1  # seconds
                
                for attempt in range(max_retries):
                    try:
                        response = openai.Embedding.create(
                            input=uncached_texts,
                            model=self.model
                        )
                        
                        api_embeddings = [item["embedding"] for item in response["data"]]
                        
                        # Cache the results
                        if self.cache:
                            for text, embedding in zip(uncached_texts, api_embeddings):
                                self.cache.set(text, self.model, embedding)
                        
                        # Place embeddings in the correct order
                        for idx, embedding in zip(uncached_indices, api_embeddings):
                            # Extend batch_embeddings list if needed
                            while len(batch_embeddings) <= idx:
                                batch_embeddings.append(None)
                            batch_embeddings[idx] = embedding
                        
                        # Respect API rate limits
                        if i + batch_size < len(texts):
                            time.sleep(0.5)  # Sleep to avoid hitting rate limits
                            
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Error generating embeddings (attempt {attempt+1}/{max_retries}): {str(e)}")
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            print(f"Failed to generate embeddings after {max_retries} attempts: {str(e)}")
                            # Return empty embeddings for this batch
                            for idx in uncached_indices:
                                while len(batch_embeddings) <= idx:
                                    batch_embeddings.append(None)
                                batch_embeddings[idx] = [0.0] * EMBEDDING_DIMENSIONS  # Use 0 vectors as fallback
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of chunks."""
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks with model {self.model}...")
        
        # Generate embeddings
        embeddings = self._batch_embed(texts, batch_size)
        
        # Add embeddings to chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding
        
        # Output cache stats if using cache
        if self.cache:
            stats = self.cache.get_stats()
            print(f"Embedding cache stats: {stats['entries']} entries, {stats['size_mb']:.2f} MB")
            print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        
        print(f"Successfully generated {len(embeddings)} embeddings")
        
        return chunks
    
    def process_chunks_file(self, input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a file containing chunks and add embeddings."""
        # Load chunks from file
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Generate embeddings
        chunks_with_embeddings = self.embed_chunks(chunks)
        
        # Save to file if output_file is provided
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_with_embeddings, f)
        
        return chunks_with_embeddings
```

### ChromaDB Integration Updates

Update your ChromaDB manager to handle the larger vector dimensions:

```python
# scripts/database/chroma_manager.py
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import json

from config import DB_DIR, COLLECTION_NAME, EMBEDDING_DIMENSIONS

class ChromaManager:
    def __init__(self, persist_directory: str = DB_DIR, collection_name: str = COLLECTION_NAME):
        """Initialize the ChromaDB manager."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the client with optimized settings for large vectors
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False,
            chroma_db_impl="duckdb+parquet",  # More efficient for larger embeddings
            persist_directory_path=self.persist_directory
        ))
    
    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            # Collection might not exist yet
            pass
        
        # Create a new collection with optimized settings
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Explicitly specify cosine similarity
        )
        print(f"Reset collection '{self.collection_name}'")
    
    def get_collection(self):
        """Get or create the collection."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            # Collection doesn't exist, create it with optimized settings
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Explicitly specify cosine similarity
            )
        
        return self.collection
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> None:
        """Add chunks to the collection."""
        collection = self.get_collection()
        
        # Verify embedding dimensions
        if len(chunks) > 0 and "embedding" in chunks[0]:
            actual_dims = len(chunks[0]["embedding"])
            if actual_dims != EMBEDDING_DIMENSIONS:
                print(f"Warning: Expected {EMBEDDING_DIMENSIONS} dimensions but found {actual_dims}")
        
        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add data to collection in batches
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch_num = i // batch_size + 1
            
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(f"Added batch {batch_num}/{total_batches} ({end_idx - i} chunks)")
    
    # Rest of ChromaManager implementation remains the same...
```

### Benchmarking and Evaluation

Add a benchmarking script to evaluate embedding quality:

```python
# scripts/benchmark_embeddings.py
#!/usr/bin/env python3

import os
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Any
import openai
from tqdm import tqdm

from config import OPENAI_API_KEY
from embedding.generator import EmbeddingGenerator

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

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
    print(f"Generating embeddings for code queries with {model_name}...")
    start_time = time.time()
    code_embeddings = generator._batch_embed(CODE_QUERIES)
    code_time = time.time() - start_time
    results["timing"]["total"] += code_time
    results["timing"]["count"] += len(CODE_QUERIES)
    
    # Generate embeddings for concept queries
    print(f"Generating embeddings for concept queries with {model_name}...")
    start_time = time.time()
    concept_embeddings = generator._batch_embed(CONCEPT_QUERIES)
    concept_time = time.time() - start_time
    results["timing"]["total"] += concept_time
    results["timing"]["count"] += len(CONCEPT_QUERIES)
    
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
        "avg_embedding_time": results["timing"]["total"] / results["timing"]["count"]
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models for PyTorch documentation search")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["text-embedding-ada-002", "text-embedding-3-large"],
                        help="Models to benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for benchmark results")
    args = parser.parse_args()
    
    all_results = {}
    
    for model in args.models:
        print(f"\n=== Benchmarking {model} ===")
        results = benchmark_model(model)
        all_results[model] = results
        
        # Print summary
        summary = results["summary"]
        print(f"\nResults for {model}:")
        print(f"  Average code similarity: {summary['avg_code_similarity']:.4f}")
        print(f"  Average concept similarity: {summary['avg_concept_similarity']:.4f}")
        print(f"  Average cross-domain similarity: {summary['avg_cross_similarity']:.4f}")
        print(f"  Average embedding time: {summary['avg_embedding_time']:.4f} seconds")
    
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
```

### Memory Optimization Techniques

For systems with limited memory, implement these optimizations:

```python
# Memory optimization in document_processing/chunker.py
def process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process all sections into appropriate chunks with memory optimization."""
    all_chunks = []
    
    # Process sections in batches to reduce memory usage
    batch_size = 100  # Adjust based on available memory
    for i in range(0, len(sections), batch_size):
        batch_sections = sections[i:i+batch_size]
        batch_chunks = []
        
        for section in batch_sections:
            chunks = self.chunk_text(section['text'], section['metadata'])
            batch_chunks.extend(chunks)
        
        all_chunks.extend(batch_chunks)
        
        # Explicit garbage collection to free memory
        import gc
        gc.collect()
    
    return all_chunks
```

## 6. Implementation Timeline by Phase

Depending on where you are in the implementation pipeline, here's how to integrate the embedding model changes:

### If in Phase 1 (Environment Setup)
- Update configuration files with new model parameters
- Adjust hardware planning for increased memory needs
- Add embedding cache directory to project structure

### If in Phase 2 (Document Processing)
- Continue with document processing as planned
- No significant changes needed at this stage as chunking is model-independent

### If in Phase 3 (Embedding Generation)
- Implement the embedding cache as shown above
- Update embedding generator to use the new model
- Adjust batch sizes based on memory constraints
- Run benchmarks to validate embedding quality

### If in Phase 4-6 (Search Interface, Integration, Optimization)
- Update ChromaDB configuration for larger vectors
- Implement memory optimization techniques
- Prioritize storage efficiency measures
- Update evaluation metrics to reflect the improved model capabilities

## 7. Troubleshooting and Common Issues

### Memory Errors
If encountering memory errors during embedding generation:
- Reduce batch size to 5 or even 3
- Implement progressive processing with intermediate saves
- Use memory profiling to identify bottlenecks

Example memory-efficient embedding generation:
```python
def embed_large_collection(self, chunks: List[Dict[str, Any]], output_file: str, batch_size: int = 5) -> None:
    """Process a large collection in stages with intermediate saves."""
    chunk_batches = [chunks[i:i+100] for i in range(0, len(chunks), 100)]
    
    for i, batch in enumerate(chunk_batches):
        print(f"Processing batch {i+1}/{len(chunk_batches)} ({len(batch)} chunks)")
        
        # Process this batch
        processed_batch = self.embed_chunks(batch, batch_size)
        
        # Save intermediate results
        temp_file = f"{output_file}.part{i+1}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(processed_batch, f)
        
        # Explicit garbage collection
        import gc
        gc.collect()
    
    # Combine results
    all_processed = []
    for i in range(len(chunk_batches)):
        temp_file = f"{output_file}.part{i+1}"
        with open(temp_file, 'r', encoding='utf-8') as f:
            all_processed.extend(json.load(f))
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed, f)
    
    # Clean up temp files
    for i in range(len(chunk_batches)):
        temp_file = f"{output_file}.part{i+1}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
```

### API Errors
When encountering OpenAI API errors:
- Verify API key configuration
- Check for rate limit issues and implement more aggressive backoff
- Implement request validation before sending to API

### Database Performance Issues
For ChromaDB performance with larger vectors:
- Use the duckdb+parquet persistence format
- Explicitly configure HNSW index parameters:
```python
self.collection = self.client.create_collection(
    name=self.collection_name,
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 128,
        "hnsw:search_ef": 96,
        "hnsw:M": 16
    }
)
```

## 8. Migration from text-embedding-ada-002

If you've already processed documents with text-embedding-ada-002, here's a migration script:

```python
# scripts/migrate_embeddings.py
#!/usr/bin/env python3

import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

from embedding.generator import EmbeddingGenerator
from config import EMBEDDING_MODEL

def migrate_embeddings(input_file: str, output_file: str, batch_size: int = 10):
    """Migrate embeddings from previous model to new model."""
    print(f"Loading chunks from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks with embeddings from old model")
    
    # Create embedding generator with the new model
    generator = EmbeddingGenerator(model=EMBEDDING_MODEL)
    
    # Re-embed all chunks
    print(f"Generating new embeddings with {EMBEDDING_MODEL}...")
    updated_chunks = generator.embed_chunks(chunks, batch_size)
    
    # Save updated chunks
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_chunks, f)
    
    print(f"Saved {len(updated_chunks)} chunks with new embeddings to {output_file}")

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
```

## 9. Conclusion

The transition to text-embedding-3-large offers significant advantages for the PyTorch Documentation Search Tool, particularly in understanding code semantics and technical language. While it requires some adjustments to the implementation, the benefits in search quality justify these changes.

Key recommendations:
1. Implement embedding caching to manage API costs
2. Adjust batch sizes to work with the larger embedding dimensions
3. Configure ChromaDB optimally for the larger vectors
4. Test and benchmark to verify improvements

By following this addendum, you can successfully integrate the improved embedding model at any stage of your implementation pipeline, resulting in better search quality for PyTorch documentation.
