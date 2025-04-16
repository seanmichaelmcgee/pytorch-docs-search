import os
import json
import openai
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from scripts.config import OPENAI_API_KEY, EMBEDDING_MODEL

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

class EmbeddingGenerator:
    def __init__(self, model: str = EMBEDDING_MODEL):
        """Initialize the embedding generator."""
        self.model = model
    
    def _batch_embed(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Retry mechanism for API calls
            max_retries = 5
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = openai.Embedding.create(
                        input=batch_texts,
                        model=self.model
                    )
                    batch_embeddings = [item["embedding"] for item in response["data"]]
                    all_embeddings.extend(batch_embeddings)
                    
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
                        all_embeddings.extend([[] for _ in batch_texts])
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 20) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of chunks."""
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings
        embeddings = self._batch_embed(texts, batch_size)
        
        # Add embeddings to chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding
        
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