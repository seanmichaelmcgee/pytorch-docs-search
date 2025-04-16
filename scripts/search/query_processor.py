import openai
from typing import List, Dict, Any

from scripts.config import OPENAI_API_KEY, EMBEDDING_MODEL

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

class QueryProcessor:
    def __init__(self, model: str = EMBEDDING_MODEL):
        """Initialize the query processor."""
        self.model = model
    
    def generate_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query using OpenAI's API."""
        try:
            response = openai.Embedding.create(
                input=query,
                model=self.model
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query to extract intent and generate embedding."""
        # Basic preprocessing
        query = query.strip()
        
        # Detect query type (code vs. concept)
        is_code_query = self._is_code_query(query)
        
        # Generate embedding
        embedding = self.generate_embedding(query)
        
        return {
            "query": query,
            "embedding": embedding,
            "is_code_query": is_code_query
        }
    
    def _is_code_query(self, query: str) -> bool:
        """Determine if a query is likely looking for code examples."""
        # Look for code-related keywords
        code_indicators = [
            "code", "example", "implementation", "function", "class",
            "method", "snippet", "syntax", "API", "parameter", "argument",
            "return", "import", "module", "library", "package",
            "how to", "how do i", "sample"
        ]
        
        query_lower = query.lower()
        
        # Check for code indicators
        for indicator in code_indicators:
            if indicator.lower() in query_lower:
                return True
        
        # Check for Python code patterns
        code_patterns = [
            "def ", "class ", "import ", "from ", "torch.", "nn.",
            "self.", "->", "=>", "==", "!=", "+=", "-=", "*=", "/=",
            "():", "@", "if __name__", "__init__", "super()", 
        ]
        
        for pattern in code_patterns:
            if pattern in query:
                return True
        
        return False