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
        
        # Initialize OpenAI client with error handling for compatibility issues
        try:
            # First attempt: standard initialization
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
        except TypeError as e:
            # If 'proxies' parameter error occurs, try with a basic HTTP client
            if "unexpected keyword argument 'proxies'" in str(e):
                import httpx
                logger.info("Creating custom HTTP client for OpenAI compatibility")
                # Create a simple HTTP client without the problematic parameters
                http_client = httpx.Client(timeout=60.0)
                self.client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
                logger.info("OpenAI client initialized with custom HTTP client")
            else:
                # For other TypeError issues, log and re-raise
                logger.error(f"Unexpected TypeError initializing OpenAI client: {str(e)}")
                raise
        
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
        """Process a query to extract intent and generate embedding with confidence scoring.
        
        Args:
            query: The search query text
            
        Returns:
            Dictionary with processed query information:
            - query: The original query text
            - embedding: The embedding vector
            - is_code_query: Whether the query is likely looking for code
            - intent_confidence: Confidence score for the classification (0.0-1.0)
        """
        # Basic preprocessing
        query = query.strip()
        
        # Generate embedding
        embedding = self.generate_embedding(query)
        
        # Compute intent score with confidence
        intent_data = self._compute_query_intent(query)
        is_code_query = intent_data["is_code_query"]
        confidence = intent_data["confidence"]
        
        # Log query classification with confidence score
        logger.info(f"Query '{query[:50]}...' classified as " + 
                  ("code query" if is_code_query else "concept query") +
                  f" with confidence {confidence:.2f}")
        
        # Verify embedding dimensions
        if len(embedding) != EMBEDDING_DIMENSIONS:
            logger.warning(f"Expected {EMBEDDING_DIMENSIONS} dimensions but got {len(embedding)}")
        
        # Free memory (especially important for large embedding vectors)
        gc.collect()
        
        return {
            "query": query,
            "embedding": embedding,
            "is_code_query": is_code_query,
            "intent_confidence": confidence
        }
    
    def _compute_query_intent(self, query: str) -> Dict[str, Any]:
        """Determine if a query is looking for code with confidence scoring.
        
        Args:
            query: The search query text
            
        Returns:
            Dictionary with query intent classification:
            - is_code_query: Whether the query is likely looking for code
            - confidence: Confidence score for the classification (0.0-1.0)
        """
        query_lower = query.lower()
        score = 0.0
        max_score = 0.0
        
        # Code indicator patterns with weights
        code_indicators = {
            # Strong indicators (weight 2.0)
            "code": 2.0, "example": 2.0, "implementation": 2.0,
            "function": 2.0, "class": 2.0, "method": 2.0, 
            "snippet": 2.0, "syntax": 2.0, 
            
            # Medium indicators (weight 1.5)
            "parameter": 1.5, "argument": 1.5, "return": 1.5, 
            "import": 1.5, "module": 1.5, "api": 1.5,
            
            # Weaker indicators (weight 1.0)
            "library": 1.0, "package": 1.0, "how to": 1.0, 
            "how do i": 1.0, "sample": 1.0, "usage": 1.0,
            "call": 1.0, "invoke": 1.0, "instantiate": 1.0,
            "create": 1.0, "initialize": 1.0, "define": 1.0
        }
        
        # Concept indicator patterns with negative weights
        concept_indicators = {
            "what is": -1.5, "explain": -1.5, "difference between": -1.5,
            "why": -1.5, "concept": -1.5, "understand": -1.5,
            "meaning": -1.5, "how does": -1.5, "when to use": -1.0,
            "purpose": -1.0, "versus": -1.0, "vs": -1.0,
            "compare": -1.0, "comparison": -1.0, "trade-off": -1.0,
            "tradeoff": -1.0, "limitation": -1.0, "advantage": -1.0,
            "disadvantage": -1.0, "problem": -1.0, "issue": -1.0
        }
        
        # Calculate maximum possible score
        max_score = sum(weight for weight in code_indicators.values())
        
        # Check for code indicators
        for indicator, weight in code_indicators.items():
            if indicator in query_lower:
                score += weight
        
        # Check for concept indicators (negative weight)
        for indicator, weight in concept_indicators.items():
            if indicator in query_lower:
                score += weight  # Weight is negative
        
        # Check for comparative queries (special handling)
        comparative_patterns = ["vs", "versus", "compared to", "or", "better than"]
        if any(pattern in query_lower for pattern in comparative_patterns):
            # Lower the confidence for comparative queries
            score *= 0.8
        
        # Check for negative queries (special handling)
        negative_patterns = ["not ", "without", "instead of", "avoid"]
        if any(pattern in query_lower for pattern in negative_patterns):
            # Reduce confidence for negative queries
            score *= 0.7
        
        # Python/PyTorch code patterns (strong indicators - weight 2.5)
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
                score += 2.5
                max_score += 2.5  # Add to max possible score
        
        # Normalize the score to confidence between 0.0 and 1.0
        # We center the range so the midpoint (0.5) represents uncertainty
        normalized_score = (score + max_score) / (2 * max_score + 0.001)
        
        # A score > 0.5 indicates code intent, with confidence proportional to distance from 0.5
        return {
            "is_code_query": normalized_score > 0.5,
            "confidence": normalized_score 
        }
    
    def _is_code_query(self, query: str) -> bool:
        """Legacy method for backward compatibility."""
        return self._compute_query_intent(query)["is_code_query"]