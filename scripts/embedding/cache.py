import os
import json
import hashlib
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from scripts.config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

# Get the logger
logger = logging.getLogger("embedding_cache")

class EmbeddingCache:
    def __init__(self, cache_dir: str, max_size_gb: float = 1.0):
        """Initialize embedding cache with versioning support.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_gb: Maximum cache size in gigabytes
        """
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.index_file = os.path.join(cache_dir, "index.json")
        self.index = {}
        self.stats = {"hits": 0, "misses": 0, "drift_detections": 0, "invalidations": 0}
        self.version_file = os.path.join(cache_dir, "version.json")
        self.version_info = self._load_version_info()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load index if it exists
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                logger.info(f"Loaded embedding cache index with {len(self.index)} entries")
                
                # Check for version mismatch
                if self._is_version_changed():
                    logger.warning("Embedding model version changed, cache will be invalidated gradually")
                    self.stats["version_mismatch"] = True
            except Exception as e:
                logger.warning(f"Failed to load embedding cache index: {e}")
                self.index = {}
    
    def _load_version_info(self) -> Dict[str, Any]:
        """Load version information or create default."""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load version info: {e}")
        
        # Default version info with current embedding model
        version_info = {
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "last_updated": time.time(),
            "drift_detected": False,
            "checksum": self._compute_model_checksum(),
            "embedding_samples": {},
            "drift_checks": 0
        }
        
        # Save the new version info
        with open(self.version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
            
        return version_info
    
    def _compute_model_checksum(self) -> str:
        """Compute a checksum that represents the current model configuration."""
        config_str = f"{EMBEDDING_MODEL}:{EMBEDDING_DIMENSIONS}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _is_version_changed(self) -> bool:
        """Check if the embedding model has changed since cache creation."""
        current_checksum = self._compute_model_checksum()
        stored_checksum = self.version_info.get("checksum", "")
        
        # If checksums don't match, version has changed
        if current_checksum != stored_checksum:
            logger.warning(f"Embedding model changed: {self.version_info.get('model')} -> {EMBEDDING_MODEL}")
            return True
        
        # If dimensions don't match, version has changed
        if self.version_info.get("dimensions") != EMBEDDING_DIMENSIONS:
            logger.warning(f"Embedding dimensions changed: {self.version_info.get('dimensions')} -> {EMBEDDING_DIMENSIONS}")
            return True
        
        return False
    
    def _update_version_info(self) -> None:
        """Update the version information file."""
        self.version_info["model"] = EMBEDDING_MODEL
        self.version_info["dimensions"] = EMBEDDING_DIMENSIONS
        self.version_info["last_updated"] = time.time()
        self.version_info["checksum"] = self._compute_model_checksum()
        
        with open(self.version_file, 'w') as f:
            json.dump(self.version_info, f, indent=2)
    
    def _verify_embedding_dimensions(self, embedding: List[float]) -> bool:
        """Verify that embedding dimensions match expected dimensions."""
        if len(embedding) != EMBEDDING_DIMENSIONS:
            logger.warning(f"Embedding dimensions mismatch: got {len(embedding)}, expected {EMBEDDING_DIMENSIONS}")
            return False
        return True
    
    def _detect_embedding_drift(self, text: str, model: str, embedding: List[float]) -> Tuple[bool, float]:
        """Detect if embeddings from the same model have drifted over time.
        
        This is done by periodically comparing embeddings of the same text
        generated at different times.
        
        Args:
            text: The text used to generate the embedding
            model: The model used to generate the embedding
            embedding: The new embedding
            
        Returns:
            Tuple of (drift_detected, drift_magnitude)
        """
        # Only check drift on a small percentage of operations
        drift_check_probability = 0.01  # 1% chance of checking for drift
        if np.random.random() > drift_check_probability:
            return False, 0.0
        
        # Hash text to use as a stable key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Get embedding samples for drift detection
        samples = self.version_info.get("embedding_samples", {})
        
        # Check if we have a sample for this text
        if text_hash in samples:
            old_embedding = samples[text_hash].get("embedding")
            old_model = samples[text_hash].get("model")
            
            # Only compare if old sample used the same model
            if old_model == model and old_embedding and len(old_embedding) == len(embedding):
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(old_embedding, embedding))
                magnitude_a = sum(a ** 2 for a in old_embedding) ** 0.5
                magnitude_b = sum(b ** 2 for b in embedding) ** 0.5
                
                if magnitude_a > 0 and magnitude_b > 0:
                    similarity = dot_product / (magnitude_a * magnitude_b)
                    drift_magnitude = 1.0 - similarity
                    
                    # Check for significant drift
                    if drift_magnitude > 0.05:  # More than 5% difference
                        logger.warning(f"Embedding drift detected: {drift_magnitude:.4f} magnitude change")
                        
                        # Update version info to note drift was detected
                        self.version_info["drift_detected"] = True
                        self.version_info["last_drift_magnitude"] = drift_magnitude
                        self.version_info["last_drift_time"] = time.time()
                        self._update_version_info()
                        
                        self.stats["drift_detections"] += 1
                        return True, drift_magnitude
        
        # Store this embedding as a sample for future drift detection
        # Only store a limited number of samples to save space
        if len(samples) < 100:  # Store up to 100 sample embeddings
            # Use a subset of the embedding to save space
            embedding_sample = embedding[:32]  # Store only first 32 dimensions
            
            samples[text_hash] = {
                "model": model,
                "timestamp": time.time(),
                "embedding": embedding_sample
            }
            
            self.version_info["embedding_samples"] = samples
            self.version_info["drift_checks"] = self.version_info.get("drift_checks", 0) + 1
            self._update_version_info()
        
        return False, 0.0
    
    def _get_hash(self, text: str, model: str) -> str:
        """Generate a hash for the text and model combination."""
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.sha256(text_bytes)
        hash_obj.update(model.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache with version checking."""
        text_hash = self._get_hash(text, model)
        
        if text_hash in self.index:
            # Version mismatch detected - validate cached embedding
            if self._is_version_changed() or self.version_info.get("drift_detected", False):
                cache_file = os.path.join(self.cache_dir, text_hash + ".json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                        
                        # Check embedding dimensions
                        cached_model = data.get("model", "")
                        cached_embedding = data.get("embedding", [])
                        
                        # Invalidate if models don't match or dimensions are wrong
                        if (cached_model != model or 
                                not self._verify_embedding_dimensions(cached_embedding)):
                            logger.info(f"Invalidating cache entry due to version mismatch or dimension error")
                            # Remove from index and return None to force regeneration
                            del self.index[text_hash]
                            self.stats["invalidations"] += 1
                            self.stats["misses"] += 1
                            return None
                    except Exception as e:
                        logger.warning(f"Failed to validate cache entry: {e}")
                        self.stats["misses"] += 1
                        return None
            
            # Normal cache hit path
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
        """Save embedding to cache with version information."""
        # Verify embedding dimensions
        if not self._verify_embedding_dimensions(embedding):
            # Don't cache invalid embeddings
            logger.warning("Not caching embedding with incorrect dimensions")
            return
        
        # Update version info if necessary
        if self._is_version_changed():
            self._update_version_info()
        
        # Detect embedding drift
        drift_detected, drift_magnitude = self._detect_embedding_drift(text, model, embedding)
        
        text_hash = self._get_hash(text, model)
        cache_file = os.path.join(self.cache_dir, text_hash + ".json")
        
        try:
            # Save embedding to file with version metadata
            with open(cache_file, 'w') as f:
                json.dump({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "model": model,
                    "embedding": embedding,
                    "version": {
                        "model": EMBEDDING_MODEL,
                        "dimensions": EMBEDDING_DIMENSIONS,
                        "timestamp": time.time(),
                        "checksum": self._compute_model_checksum()
                    }
                }, f)
            
            # Update index
            self.index[text_hash] = {
                "file": text_hash + ".json",
                "size": os.path.getsize(cache_file),
                "last_access": time.time(),
                "created": time.time(),
                "model": model,
                "dimensions": len(embedding)
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
        
        # Prioritize removing entries from old versions if version changed
        if self._is_version_changed():
            current_model = EMBEDDING_MODEL
            current_dimensions = EMBEDDING_DIMENSIONS
            
            # First try to remove entries from different models or dimensions
            for text_hash, entry in sorted_entries:
                if bytes_removed >= bytes_to_remove:
                    break
                
                if entry.get("model") != current_model or entry.get("dimensions") != current_dimensions:
                    cache_file = os.path.join(self.cache_dir, entry["file"])
                    if os.path.exists(cache_file):
                        bytes_removed += entry["size"]
                        os.remove(cache_file)
                        del self.index[text_hash]
                        removed_count += 1
        
        # If we still need to remove more, use standard LRU approach
        if bytes_removed < bytes_to_remove:
            for text_hash, entry in sorted_entries:
                if bytes_removed >= bytes_to_remove:
                    break
                
                if text_hash not in self.index:  # Skip already removed entries
                    continue
                
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
        """Get cache statistics with version information."""
        total_size = sum(entry["size"] for entry in self.index.values())
        
        # Count entries by model
        model_counts = {}
        for entry in self.index.values():
            model_name = entry.get("model", "unknown")
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        # Collect version statistics
        version_stats = {
            "current_model": EMBEDDING_MODEL,
            "current_dimensions": EMBEDDING_DIMENSIONS,
            "version_changed": self._is_version_changed(),
            "drift_detected": self.version_info.get("drift_detected", False),
            "last_drift_magnitude": self.version_info.get("last_drift_magnitude", 0),
            "last_updated": self.version_info.get("last_updated", 0)
        }
        
        return {
            "entries": len(self.index),
            "size_mb": total_size / 1024 / 1024,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "invalidations": self.stats.get("invalidations", 0),
            "drift_detections": self.stats.get("drift_detections", 0),
            "hit_rate": self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0,
            "model_counts": model_counts,
            "version": version_stats
        }