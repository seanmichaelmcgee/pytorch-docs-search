import os
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any

# Get the logger
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
                json.dump({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "model": model,
                    "embedding": embedding
                }, f)
            
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
        
        if removed_count > 0:
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