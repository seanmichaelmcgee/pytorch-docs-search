import os
import logging
from typing import List, Dict, Any
import gc
from openai import OpenAI

from scripts.config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from scripts.embedding.cache import EmbeddingCache

# Setup logger
logger = logging.getLogger("query_processor")

class QueryProcessor:
    def __init__(self, model: str = EMBEDDING_MODEL, use_cache: bool = True):
        """Initialize the query processor for handling search queries.
        
        Args:
            model: The OpenAI embedding model to use
            use_cache: Whether to use cache for query embeddings
        """
        self.model = model
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Cache for storing query embeddings (reuses the embedding cache)
        if use_cache:
            from scripts.config import EMBEDDING_CACHE_DIR, EMBEDDING_CACHE_MAX_SIZE_GB
            # Ensure queries subdirectory exists
            query_cache_dir = os.path.join(EMBEDDING_CACHE_DIR, 'queries')
            os.makedirs(query_cache_dir, exist_ok=True)
            self.cache = EmbeddingCache(
                query_cache_dir,
                max_size_gb=EMBEDDING_CACHE_MAX_SIZE_GB / 10  # Use smaller cache for queries
            )
            logger.info(f"Query embedding cache initialized at {query_cache_dir}")
        else:
            self.cache = None
    
    def generate_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query using OpenAI's API.
        
        Uses cache if available to avoid redundant API calls for the same query.
        
        Args:
            query: The search query text
            
        Returns:
            Embedding vector for the query
            
        Raises:
            Exception: If the embedding generation fails and no fallback is available
        """
        # Check cache first
        if self.cache:
            cached_embedding = self.cache.get(query, self.model)
            if cached_embedding:
                logger.info(f"Using cached embedding for query: {query[:50]}...")
                return cached_embedding
        
        # Generate embedding via API
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.model
            )
            embedding = response.data[0].embedding
            
            # Cache the embedding
            if self.cache:
                self.cache.set(query, self.model, embedding)
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            # If we can't generate an embedding, raise the exception
            # This will be handled by the search interface
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query to extract intent and generate embedding.
        
        Args:
            query: The search query text
            
        Returns:
            Dictionary with processed query information:
            - query: The original query text
            - embedding: The embedding vector
            - is_code_query: Whether the query is likely looking for code
        """
        # Basic preprocessing
        query = query.strip()
        
        # Detect query type (code vs. concept)
        is_code_query = self._is_code_query(query)
        logger.info(f"Query '{query[:50]}...' classified as " + 
                  ("code query" if is_code_query else "concept query"))
        
        # Generate embedding
        embedding = self.generate_embedding(query)
        
        # Verify embedding dimensions
        if len(embedding) != EMBEDDING_DIMENSIONS:
            logger.warning(f"Expected {EMBEDDING_DIMENSIONS} dimensions but got {len(embedding)}")
        
        # Free memory (especially important for large embedding vectors)
        gc.collect()
        
        return {
            "query": query,
            "embedding": embedding,
            "is_code_query": is_code_query
        }
    
    def _is_code_query(self, query: str) -> bool:
        """Determine if a query is likely looking for code examples.
        
        This function analyzes the query text to determine if the user is looking
        for code examples or implementation details vs. conceptual information.
        
        Args:
            query: The search query text
            
        Returns:
            True if the query appears to be looking for code, False otherwise
        """
        # Look for code-related keywords
        code_indicators = [
            "code", "example", "implementation", "function", "class",
            "method", "snippet", "syntax", "API", "parameter", "argument",
            "return", "import", "module", "library", "package",
            "how to", "how do i", "sample", "usage", "call", "invoke",
            "instantiate", "create", "initialize", "define"
        ]
        
        query_lower = query.lower()
        
        # Check for code indicators
        for indicator in code_indicators:
            if indicator.lower() in query_lower:
                return True
        
        # Check for Python/PyTorch code patterns
        code_patterns = [
            "def ", "class ", "import ", "from ", "torch.", "nn.",
            "self.", "->", "=>", "==", "!=", "+=", "-=", "*=", "/=",
            "():", "@", "if __name__", "__init__", "super()", 
            ".cuda()", ".to(", ".backward(", ".forward(", ".parameters()",
            "optimizer.", "loss.", "dataset.", "dataloader", "model.",
            "tensor(", ".view(", ".reshape("
        ]
        
        for pattern in code_patterns:
            if pattern in query:
                return True
        
        return False