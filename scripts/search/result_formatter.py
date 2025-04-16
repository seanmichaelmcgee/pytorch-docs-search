from typing import List, Dict, Any
import logging

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
            # Adjust snippet length based on number of results
            # Fewer results can have longer snippets, more results need shorter snippets
            max_snippet_length = 300 if len(documents) <= 3 else 150
            snippet = doc[:max_snippet_length] + "..." if len(doc) > max_snippet_length else doc
            
            # Convert distance to similarity score (1.0 is exact match)
            # Handle potential non-scalar distance values (e.g., lists)
            if isinstance(distance, (int, float)):
                similarity = 1.0 - distance
            else:
                # If distance is not a scalar, use a default score of 0.5
                similarity = 0.5
            
            # Handle metadata formats - could be dict or could be something else
            if isinstance(metadata, dict):
                # Dictionary metadata
                title = metadata.get("title", f"Result {i+1}")
                source = metadata.get("source", "")
                chunk_type = metadata.get("chunk_type", "unknown")
            else:
                # Handle newer ChromaDB format where metadata seems to be a complex object
                # Don't print debug info in production runs - only log it
                logger = logging.getLogger("result_formatter")
                logger.debug(f"Non-dictionary metadata type: {type(metadata)}. Content: {str(metadata)[:100]}")
                
                # From the debug output, it seems metadata is returned as a list containing a dictionary
                if isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
                    meta_dict = metadata[0]
                    title = meta_dict.get("title", f"Result {i+1}")
                    source = meta_dict.get("source", "")
                    chunk_type = meta_dict.get("chunk_type", "unknown")
                else:
                    # Default fallback if we can't extract metadata
                    title = f"Result {i+1}"
                    source = ""
                    chunk_type = "unknown"
                    
                    # Try to extract metadata if possible
                    if hasattr(metadata, "__getitem__"):
                        try:
                            # Common field positions in some ChromaDB versions
                            if len(metadata) > 0:
                                title = metadata[0] if metadata[0] else title
                            if len(metadata) > 1:
                                source = metadata[1] if metadata[1] else source
                            if len(metadata) > 2:
                                chunk_type = metadata[2] if metadata[2] else chunk_type
                        except:
                            pass
                
            # Add result with rich metadata
            formatted_results.append({
                "title": title,
                "snippet": snippet,
                "source": source,
                "chunk_type": chunk_type,
                "score": float(similarity)
            })
        
        # Return the formatted response
        return {
            "results": formatted_results,
            "query": query,
            "count": len(formatted_results)
        }
    
    def rank_results(self, results: Dict[str, Any], is_code_query: bool, 
                       intent_confidence: float = 0.75) -> Dict[str, Any]:
        """Rank and filter results based on query type and confidence.
        
        Args:
            results: The formatted search results
            is_code_query: Whether the query is looking for code
            intent_confidence: Confidence score for query intent classification (0.0-1.0)
        
        Returns:
            Re-ranked results
        """
        if "results" not in results:
            return results
        
        # Get the results
        formatted_results = results["results"]
        
        # Adjust boost factor based on confidence
        # Lower confidence = smaller boost, higher confidence = larger boost
        confidence_factor = (intent_confidence - 0.5) * 2.0 if intent_confidence > 0.5 else 0.0
        
        # Determine boost values based on confidence
        # Max boost of 30% for high confidence, min boost of 0% for low confidence
        code_boost = 1.0 + (0.3 * confidence_factor)
        text_boost = 1.0 + (0.2 * confidence_factor)
        
        # Special handling for ambiguous/mixed intent queries (confidence near 0.5)
        is_ambiguous = 0.4 < intent_confidence < 0.6
        
        # Handle mixed queries by combining both result types
        if is_ambiguous:
            # For ambiguous queries, do a balanced ranking
            # Group results by type
            code_results = []
            text_results = []
            
            for result in formatted_results:
                if result.get("chunk_type") == "code":
                    code_results.append(result)
                else:
                    text_results.append(result)
            
            # Sort each group
            code_results.sort(key=lambda x: x["score"], reverse=True)
            text_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Interleave results with a slight preference for the detected type
            # This ensures diversity while maintaining the primary intent
            merged_results = []
            code_idx, text_idx = 0, 0
            
            # Determine preference ratio (close to 50:50 for ambiguous queries)
            code_preference = intent_confidence if is_code_query else (1 - intent_confidence)
            
            # Interleave with preference
            while code_idx < len(code_results) or text_idx < len(text_results):
                # Determine which type to take next
                if code_idx >= len(code_results):
                    merged_results.append(text_results[text_idx])
                    text_idx += 1
                elif text_idx >= len(text_results):
                    merged_results.append(code_results[code_idx])
                    code_idx += 1
                # Use preference for interleaving
                elif ((code_idx / (code_idx + 1)) <= code_preference and 
                      code_idx < len(code_results)):
                    merged_results.append(code_results[code_idx])
                    code_idx += 1
                else:
                    merged_results.append(text_results[text_idx])
                    text_idx += 1
            
            # Update results with merged list
            formatted_results = merged_results
        else:
            # Normal boosting for clear intent queries
            for result in formatted_results:
                base_score = result["score"]
                
                # Boost code results for code queries
                if is_code_query and result.get("chunk_type") == "code":
                    result["score"] = min(1.0, base_score * code_boost)
                
                # Boost text results for concept queries
                elif not is_code_query and result.get("chunk_type") == "text":
                    result["score"] = min(1.0, base_score * text_boost)
            
            # Re-sort by score
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Special handling for comparative queries
        # Look for comparison terms in the query recorded in results
        query = results.get("query", "").lower()
        comparative_patterns = ["vs", "versus", "compared to", "or", "better than", 
                               "difference between", "advantages", "disadvantages"]
        
        is_comparative = any(pattern in query for pattern in comparative_patterns)
        
        if is_comparative:
            # For comparative queries, ensure diversity of content
            # This helps when comparing two concepts or implementations
            seen_sources = set()
            filtered_results = []
            
            for result in formatted_results:
                source = result.get("source", "")
                # Get base source without specific parts
                base_source = source.split("/")[-1] if "/" in source else source
                
                # Add result if we haven't seen this source yet
                if base_source not in seen_sources:
                    seen_sources.add(base_source)
                    filtered_results.append(result)
            
            # Update results with filtered list
            formatted_results = filtered_results
        
        # Update results
        results["results"] = formatted_results
        
        # Add confidence metadata to results
        results["intent_confidence"] = intent_confidence
        results["is_ambiguous"] = is_ambiguous
        results["is_comparative"] = is_comparative if 'is_comparative' not in results else results["is_comparative"]
        
        return results