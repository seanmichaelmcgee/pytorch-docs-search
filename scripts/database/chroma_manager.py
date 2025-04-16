import os
import logging
import chromadb
from typing import List, Dict, Any, Optional, Union
import json
import gc

from scripts.config import DB_DIR, COLLECTION_NAME, EMBEDDING_DIMENSIONS

# Setup logger
logger = logging.getLogger("chroma_manager")

def get_optimal_batch_size():
    """Dynamically adjust batch size based on available memory for ChromaDB operations"""
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        # Simple heuristic - smaller batches for less memory
        if available_memory < 2:  # Less than 2GB available
            logger.info(f"Low memory available ({available_memory:.2f}GB), using small batch size of 10")
            return 10
        elif available_memory < 4:  # Less than 4GB available
            logger.info(f"Limited memory available ({available_memory:.2f}GB), using moderate batch size of 25")
            return 25
        elif available_memory < 8:  # Less than 8GB available
            logger.info(f"Moderate memory available ({available_memory:.2f}GB), using default batch size of 50")
            return 50
        else:
            logger.info(f"Sufficient memory available ({available_memory:.2f}GB), using larger batch size of 100")
            return 100
    except ImportError:
        logger.info("psutil not available, using conservative default batch size")
        return 25  # Default conservative batch size if psutil unavailable

class ChromaManager:
    def __init__(self, persist_directory: str = DB_DIR, collection_name: str = COLLECTION_NAME):
        """Initialize the ChromaDB manager for handling large embedding vectors.
        
        This implementation is optimized for the text-embedding-3-large model with
        3072-dimensional embeddings.
        
        Args:
            persist_directory: Directory to store the ChromaDB files
            collection_name: Name of the collection in ChromaDB
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the client with optimized settings for large vectors
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
    def _ensure_vector_format(self, embedding):
        """Ensure vector is in the correct format for ChromaDB.
        
        Handles type conversion, dimension checking, and ensures consistent
        vector format for ChromaDB operations.
        
        Args:
            embedding: Vector embedding in any format
            
        Returns:
            Properly formatted embedding vector
        """
        # Handle empty or None embeddings
        if not embedding:
            logger.warning("Empty embedding received, using zero vector")
            return [0.0] * EMBEDDING_DIMENSIONS
            
        # Handle NumPy arrays and other array-like objects
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
            
        # Ensure all values are Python floats
        try:
            embedding = [float(x) for x in embedding]
        except (TypeError, ValueError) as e:
            logger.error(f"Error converting embedding values to float: {e}")
            return [0.0] * EMBEDDING_DIMENSIONS
        
        # Verify dimensions
        if len(embedding) != EMBEDDING_DIMENSIONS:
            logger.warning(f"Expected {EMBEDDING_DIMENSIONS} dimensions but got {len(embedding)}")
            # Pad or truncate if necessary
            if len(embedding) < EMBEDDING_DIMENSIONS:
                logger.info(f"Padding embedding from {len(embedding)} to {EMBEDDING_DIMENSIONS} dimensions")
                embedding.extend([0.0] * (EMBEDDING_DIMENSIONS - len(embedding)))
            else:
                logger.info(f"Truncating embedding from {len(embedding)} to {EMBEDDING_DIMENSIONS} dimensions")
                embedding = embedding[:EMBEDDING_DIMENSIONS]
                
        return embedding
    
    def reset_collection(self) -> None:
        """Delete and recreate the collection with optimized settings for large vectors."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection '{self.collection_name}'")
        except Exception as e:
            # Collection might not exist yet
            logger.info(f"No existing collection to delete: {str(e)}")
        
        # Create a new collection with improved HNSW index settings
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",          # Use cosine similarity for text embeddings
                "hnsw:construction_ef": 200,     # Increased from 128 for better recall
                "hnsw:search_ef": 128,           # Increased from 96 for better search quality
                "hnsw:M": 16                     # Keep as is for memory efficiency
            }
        )
        logger.info(f"Created new collection '{self.collection_name}' with optimized settings")
        print(f"Reset collection '{self.collection_name}' with optimized settings for large vectors")
    
    def get_collection(self):
        """Get or create the collection with optimized settings."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection '{self.collection_name}'")
        except Exception as e:
            # Collection doesn't exist, create it with optimized settings
            logger.info(f"Creating new collection: {str(e)}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,  # Increased from 128 for better recall
                    "hnsw:search_ef": 128,        # Increased from 96 for better search quality
                    "hnsw:M": 16                  # Keep as is for memory efficiency
                }
            )
            logger.info(f"Created new collection '{self.collection_name}'")
        
        return self.collection
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: Optional[int] = None) -> None:
        """Add chunks to the collection.
        
        Optimized for adding large embedding vectors by using memory-aware batch sizing
        and explicitly freeing memory after each batch.
        
        Args:
            chunks: List of chunks with embeddings to add
            batch_size: Number of chunks to add in each batch (if None, determined dynamically)
        """
        # Determine optimal batch size if not provided
        if batch_size is None:
            batch_size = get_optimal_batch_size()
        collection = self.get_collection()
        
        # Verify embedding dimensions
        if chunks and "embedding" in chunks[0]:
            actual_dims = len(chunks[0]["embedding"])
            if actual_dims != EMBEDDING_DIMENSIONS and actual_dims > 0:
                logger.warning(f"Warning: Expected {EMBEDDING_DIMENSIONS} dimensions but found {actual_dims}")
                print(f"Warning: Expected {EMBEDDING_DIMENSIONS} dimensions but found {actual_dims}")
        
        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add data to collection in smaller batches for large vectors
        total_batches = (len(chunks) - 1) // batch_size + 1
        logger.info(f"Adding {len(chunks)} chunks in {total_batches} batches")
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch_num = i // batch_size + 1
            
            # Process each embedding to ensure correct format
            batch_embeddings = []
            for j, emb in enumerate(embeddings[i:end_idx]):
                # Apply robust type handling to each embedding
                processed_emb = self._ensure_vector_format(emb)
                batch_embeddings.append(processed_emb)
            
            try:
                collection.add(
                    ids=ids[i:end_idx],
                    embeddings=batch_embeddings,
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                logger.info(f"Added batch {batch_num}/{total_batches} ({end_idx - i} chunks)")
                print(f"Added batch {batch_num}/{total_batches} ({end_idx - i} chunks)")
                
                # Free memory after each batch
                if batch_num % 5 == 0:  
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error adding batch {batch_num}: {str(e)}")
                print(f"Error adding batch {batch_num}: {str(e)}")
    
    def load_from_file(self, filepath: str, reset: bool = True, batch_size: Optional[int] = None) -> None:
        """Load chunks from a file into ChromaDB.
        
        Args:
            filepath: Path to the JSON file containing chunks with embeddings
            reset: Whether to reset the collection before loading
            batch_size: Number of chunks to add in each batch (if None, determined dynamically)
        """
        logger.info(f"Loading chunks from {filepath}...")
        print(f"Loading chunks from {filepath}...")
        
        # Load the chunks
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks from file")
        print(f"Loaded {len(chunks)} chunks from file")
        
        # Reset collection if requested
        if reset:
            self.reset_collection()
        
        # For very large collections, process in stages
        if len(chunks) > 1000:
            # Process in batches of 1000 to avoid memory issues
            segment_size = 1000
            for i in range(0, len(chunks), segment_size):
                end_idx = min(i + segment_size, len(chunks))
                logger.info(f"Processing segment {i//segment_size + 1}/{(len(chunks)-1)//segment_size + 1} ({i} to {end_idx})")
                print(f"Processing segment {i//segment_size + 1}/{(len(chunks)-1)//segment_size + 1} ({i} to {end_idx})")
                
                segment_chunks = chunks[i:end_idx]
                self.add_chunks(segment_chunks, batch_size)
                
                # Free memory
                gc.collect()
        else:
            # Add chunks to collection directly
            self.add_chunks(chunks, batch_size)
        
        logger.info(f"Successfully loaded {len(chunks)} chunks into ChromaDB")
        print(f"Successfully loaded {len(chunks)} chunks into ChromaDB")
    
    def query(self, query_embedding: List[float], 
              n_results: int = 5, 
              filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the collection.
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            filters: Optional filters to apply to the query
            
        Returns:
            Dictionary containing query results
        """
        collection = self.get_collection()
        
        # Ensure query embedding has the correct format
        query_embedding = self._ensure_vector_format(query_embedding)
        
        # Prepare query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Add filters if provided
        if filters:
            query_params["where"] = filters
        
        # Execute query
        try:
            results = collection.query(**query_params)
            
            # Convert to dictionary for compatibility with both old and new ChromaDB versions
            # Check if results is already a dictionary or a ChromaDB object
            if hasattr(results, "keys"):
                # If it's a dictionary-like object (new ChromaDB)
                result_dict = {
                    "ids": results["ids"],
                    "documents": results["documents"],
                    "metadatas": results["metadatas"],
                    "distances": results["distances"]
                }
            else:
                # For older versions or different return types
                logger.warning("Non-dictionary result type from ChromaDB, using custom conversion")
                result_dict = {
                    "ids": [results.ids] if hasattr(results, "ids") else [[]],
                    "documents": [results.documents] if hasattr(results, "documents") else [[]],
                    "metadatas": [results.metadatas] if hasattr(results, "metadatas") else [[]],
                    "distances": [results.distances] if hasattr(results, "distances") else [[]]
                }
            
            # Safe way to get result count
            result_count = 0
            if "ids" in result_dict and result_dict["ids"]:
                if isinstance(result_dict["ids"], list) and result_dict["ids"]:
                    if isinstance(result_dict["ids"][0], list):
                        result_count = len(result_dict["ids"][0])
                    else:
                        result_count = len(result_dict["ids"])
            
            logger.info(f"Query returned {result_count} results")
            return result_dict
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            # Return empty results as fallback
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        collection = self.get_collection()
        
        # Get all metadata (this could be memory-intensive for large collections)
        try:
            # For large collections, get a count first and potentially use paging
            count = collection.count()
            if count > 10000:
                logger.warning(f"Large collection detected ({count} items), calculating stats from sample")
                results = collection.get(limit=10000)  # Get a sample
                is_sample = True
            else:
                results = collection.get()
                is_sample = False
                
            # Calculate statistics
            total_chunks = count if is_sample else len(results["ids"])
            
            # Count by chunk type
            chunk_types = {}
            if "metadatas" in results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    chunk_type = metadata.get("chunk_type", "unknown")
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Count by source
            sources = {}
            if "metadatas" in results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    source = metadata.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1
            
            # If using a sample, scale up the counts
            if is_sample and total_chunks > 0:
                scale_factor = count / len(results["ids"])
                for key in chunk_types:
                    chunk_types[key] = int(chunk_types[key] * scale_factor)
                for key in sources:
                    sources[key] = int(sources[key] * scale_factor)
            
            return {
                "total_chunks": total_chunks,
                "chunk_types": chunk_types,
                "sources": sources,
                "is_sample": is_sample if is_sample else False
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "total_chunks": 0,
                "chunk_types": {},
                "sources": {},
                "error": str(e)
            }