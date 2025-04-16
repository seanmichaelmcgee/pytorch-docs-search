import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import json

from scripts.config import DB_DIR, COLLECTION_NAME

class ChromaManager:
    def __init__(self, persist_directory: str = DB_DIR, collection_name: str = COLLECTION_NAME):
        """Initialize the ChromaDB manager."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the client
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
    
    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            # Collection might not exist yet
            pass
        
        # Create a new collection
        self.collection = self.client.create_collection(self.collection_name)
        print(f"Reset collection '{self.collection_name}'")
    
    def get_collection(self):
        """Get or create the collection."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(self.collection_name)
        
        return self.collection
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Add chunks to the collection."""
        collection = self.get_collection()
        
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
    
    def load_from_file(self, filepath: str, reset: bool = True, batch_size: int = 100) -> None:
        """Load chunks from a file into ChromaDB."""
        print(f"Loading chunks from {filepath}...")
        
        # Load the chunks
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} chunks from file")
        
        # Reset collection if requested
        if reset:
            self.reset_collection()
        
        # Add chunks to collection
        self.add_chunks(chunks, batch_size)
        
        print(f"Successfully loaded {len(chunks)} chunks into ChromaDB")
    
    def query(self, query_embedding: List[float], 
              n_results: int = 5, 
              filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the collection."""
        collection = self.get_collection()
        
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
        results = collection.query(**query_params)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        collection = self.get_collection()
        
        # Get all metadata
        results = collection.get()
        
        # Calculate statistics
        total_chunks = len(results["ids"]) if "ids" in results else 0
        
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
        
        return {
            "total_chunks": total_chunks,
            "chunk_types": chunk_types,
            "sources": sources
        }