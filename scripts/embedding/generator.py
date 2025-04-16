import os
import json
import time
import logging
import gc
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse 

from scripts.config import (
    OPENAI_API_KEY, 
    EMBEDDING_MODEL, 
    EMBEDDING_DIMENSIONS, 
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_CACHE_MAX_SIZE_GB
)
from scripts.embedding.cache import EmbeddingCache

# Setup logger
logger = logging.getLogger("embedding_generator")

def get_optimal_batch_size():
    """Dynamically adjust batch size based on document characteristics and available memory"""
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        # Simple heuristic - smaller batches for less memory
        if available_memory < 2:  # Less than 2GB available
            logger.info(f"Low memory available ({available_memory:.2f}GB), using small batch size of 5")
            return 5
        elif available_memory < 4:  # Less than 4GB available
            logger.info(f"Limited memory available ({available_memory:.2f}GB), using moderate batch size of 10")
            return 10
        elif available_memory < 8:  # Less than 8GB available
            logger.info(f"Moderate memory available ({available_memory:.2f}GB), using default batch size of 20")
            return 20
        else:
            logger.info(f"Sufficient memory available ({available_memory:.2f}GB), using larger batch size of 50")
            return 50
    except ImportError:
        logger.info("psutil not available, using conservative default batch size")
        return 20  # Default conservative batch size if psutil unavailable

class EmbeddingGenerator:
    def __init__(self, model: str = EMBEDDING_MODEL, use_cache: bool = True):
        """Initialize the embedding generator.
        
        Args:
            model: The OpenAI embedding model to use
            use_cache: Whether to use the embedding cache
        """
        self.model = model
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize cache if enabled
        if use_cache:
            self.cache = EmbeddingCache(
                EMBEDDING_CACHE_DIR, 
                max_size_gb=EMBEDDING_CACHE_MAX_SIZE_GB
            )
            logger.info(f"Embedding cache initialized at {EMBEDDING_CACHE_DIR}")
        else:
            self.cache = None
            logger.info("Embedding cache disabled")
    
    def _get_embedding_with_cache(self, text: str) -> List[float]:
        """Get embedding for a single text, using cache if available."""
        if self.cache:
            cached_embedding = self.cache.get(text, self.model)
            if cached_embedding:
                return cached_embedding
        
        # Generate embedding via API
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            if self.cache:
                self.cache.set(text, self.model, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            # Return zeros as fallback
            return [0.0] * EMBEDDING_DIMENSIONS
    
    def _batch_embed(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for a batch of texts.
        
        Uses caching and handles larger embedding dimensions with appropriate
        batch sizes for the text-embedding-3-large model.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed in each API call (if None, determined dynamically)
        
        Returns:
            List of embeddings, one for each input text
        """
        # Determine optimal batch size if not provided
        if batch_size is None:
            batch_size = get_optimal_batch_size()
        all_embeddings = []
        
        # Process in batches to avoid API limits and memory issues
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
                        response = self.client.embeddings.create(
                            input=uncached_texts,
                            model=self.model
                        )
                        
                        api_embeddings = [item.embedding for item in response.data]
                        
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
                            time.sleep(1.0)  # Increased sleep time for larger model
                            
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Error generating embeddings (attempt {attempt+1}/{max_retries}): {str(e)}")
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            logger.error(f"Failed to generate embeddings after {max_retries} attempts: {str(e)}")
                            # Return zero embeddings for this batch as fallback
                            for idx in uncached_indices:
                                while len(batch_embeddings) <= idx:
                                    batch_embeddings.append(None)
                                batch_embeddings[idx] = [0.0] * EMBEDDING_DIMENSIONS
            
            # Ensure all positions have embeddings
            for j in range(len(batch_texts)):
                if j >= len(batch_embeddings) or batch_embeddings[j] is None:
                    batch_embeddings.append([0.0] * EMBEDDING_DIMENSIONS)
            
            all_embeddings.extend(batch_embeddings[:len(batch_texts)])
            
            # Explicitly run garbage collection to free memory after each batch
            # This is important for large embeddings
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunks to embed
            batch_size: Number of chunks to embed in each API call (if None, determined dynamically)
            
        Returns:
            List of chunks with embeddings added
        """
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Determine optimal batch size if not provided
        if batch_size is None:
            batch_size = get_optimal_batch_size()
        
        logger.info(f"Generating embeddings for {len(texts)} chunks using model {self.model} with batch size {batch_size}...")
        print(f"Generating embeddings for {len(texts)} chunks using model {self.model} with batch size {batch_size}...")
        
        # For large sets, process in smaller batches to manage memory
        embeddings = []
        
        # Process in manageable segments (also dynamically sized)
        segment_size = min(500, batch_size * 25)  # Scale segment size with batch size
        for i in range(0, len(texts), segment_size):
            end_idx = min(i + segment_size, len(texts))
            logger.info(f"Processing segment {i//segment_size + 1}/{(len(texts)-1)//segment_size + 1} ({i} to {end_idx})")
            print(f"Processing segment {i//segment_size + 1}/{(len(texts)-1)//segment_size + 1} ({i} to {end_idx})")
            
            segment_texts = texts[i:end_idx]
            segment_embeddings = self._batch_embed(segment_texts, batch_size)
            embeddings.extend(segment_embeddings)
            
            # Free memory
            gc.collect()
        
        # Add embeddings to chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding
        
        # Output cache stats if using cache
        if self.cache:
            stats = self.cache.get_stats()
            logger.info(f"Embedding cache stats: {stats['entries']} entries, {stats['size_mb']:.2f} MB")
            logger.info(f"Cache hit rate: {stats['hit_rate']:.2%}")
            print(f"Embedding cache stats: {stats['entries']} entries, {stats['size_mb']:.2f} MB")
            print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        print(f"Successfully generated {len(embeddings)} embeddings")
        
        return chunks
    
    def process_chunks_file(self, input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a file containing chunks and add embeddings.
        
        Args:
            input_file: JSON file containing chunks
            output_file: Where to save chunks with embeddings (defaults to input_file)
            
        Returns:
            List of chunks with embeddings
        """
        logger.info(f"Loading chunks from {input_file}")
        
        # Load chunks from file
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks for embedding generation")
        
        # For very large files, process in batches to avoid memory issues
        if len(chunks) > 1000:
            return self._process_large_chunks_file(chunks, output_file or input_file)
        
        # Generate embeddings
        chunks_with_embeddings = self.embed_chunks(chunks)
        
        # Save to file if output_file is provided
        output_path = output_file or input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f)
            
        logger.info(f"Saved {len(chunks_with_embeddings)} chunks with embeddings to {output_path}")
        
        return chunks_with_embeddings
    
    def _process_large_chunks_file(self, chunks: List[Dict[str, Any]], output_file: str) -> List[Dict[str, Any]]:
        """Process a large collection in stages with intermediate saves."""
        # Determine optimal segment size based on memory availability
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            # Adjust batch size based on available memory
            if available_memory < 4:  # Less than 4GB available
                batch_size = 100
            elif available_memory < 8:  # Less than 8GB available
                batch_size = 250
            else:
                batch_size = 500
            logger.info(f"Memory-optimized batch size: {batch_size} chunks (available memory: {available_memory:.2f}GB)")
        except ImportError:
            batch_size = 250  # Conservative default
            logger.info(f"Using conservative default batch size: {batch_size} chunks (psutil not available)")
            
        chunk_batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        
        logger.info(f"Processing {len(chunks)} chunks in {len(chunk_batches)} batches")
        print(f"Processing {len(chunks)} chunks in {len(chunk_batches)} batches")
        
        for i, batch in enumerate(chunk_batches):
            logger.info(f"Processing batch {i+1}/{len(chunk_batches)} ({len(batch)} chunks)")
            print(f"Processing batch {i+1}/{len(chunk_batches)} ({len(batch)} chunks)")
            
            # Process this batch
            processed_batch = self.embed_chunks(batch)
            
            # Save intermediate results
            temp_file = f"{output_file}.part{i+1}"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(processed_batch, f)
            
            logger.info(f"Saved intermediate results to {temp_file}")
            
            # Explicit garbage collection
            gc.collect()
        
        # Combine results
        all_processed = []
        for i in range(len(chunk_batches)):
            temp_file = f"{output_file}.part{i+1}"
            with open(temp_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_processed.extend(batch_data)
        
        # Save final results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_processed, f)
        
        logger.info(f"Combined all batches and saved to {output_file}")
        
        # Clean up temp files
        for i in range(len(chunk_batches)):
            temp_file = f"{output_file}.part{i+1}"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        logger.info("Cleaned up temporary files")
        
        return all_processed