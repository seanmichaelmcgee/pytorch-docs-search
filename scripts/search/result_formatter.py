from typing import List, Dict, Any

class ResultFormatter:
    def __init__(self):
        """Initialize the result formatter."""
        pass
    
    def format_results(self, results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format ChromaDB results into a structured response."""
        formatted_results = []
        
        # Extract results from ChromaDB response
        # Handle both old and new ChromaDB response formats
        if results is None:
            return {
                "results": [],
                "query": query,
                "count": 0
            }
            
        if isinstance(results.get('documents'), list):
            # New ChromaDB returns flat lists directly
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            distances = results.get('distances', [])
        else:
            # Handle older format that had nested lists
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
        
        # Format each result
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Create a snippet (first 200 chars)
            snippet = doc[:200] + "..." if len(doc) > 200 else doc
            
            # Convert distance to similarity score (1.0 is exact match)
            similarity = 1.0 - distance
            
            # Add result with rich metadata
            formatted_results.append({
                "title": metadata.get("title", f"Result {i+1}"),
                "snippet": snippet,
                "source": metadata.get("source", ""),
                "chunk_type": metadata.get("chunk_type", "unknown"),
                "score": float(similarity)
            })
        
        # Return the formatted response
        return {
            "results": formatted_results,
            "query": query,
            "count": len(formatted_results)
        }
    
    def rank_results(self, results: Dict[str, Any], is_code_query: bool) -> Dict[str, Any]:
        """Rank and filter results based on query type."""
        if "results" not in results:
            return results
        
        # Get the results
        formatted_results = results["results"]
        
        # Adjust scores based on query type
        for result in formatted_results:
            base_score = result["score"]
            
            # Boost code results for code queries
            if is_code_query and result.get("chunk_type") == "code":
                result["score"] = min(1.0, base_score * 1.2)  # 20% boost, max 1.0
            
            # Boost text results for concept queries
            elif not is_code_query and result.get("chunk_type") == "text":
                result["score"] = min(1.0, base_score * 1.1)  # 10% boost, max 1.0
        
        # Re-sort by score
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update results
        results["results"] = formatted_results
        
        return results