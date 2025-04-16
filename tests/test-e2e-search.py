#!/usr/bin/env python3

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("search_test")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the search functionality
from scripts.search.query_processor import QueryProcessor
from scripts.search.result_formatter import ResultFormatter
from scripts.database.chroma_manager import ChromaManager
from scripts.config import MAX_RESULTS

def run_search_test(query: str, filter_type: str = None, num_results: int = MAX_RESULTS) -> Dict[str, Any]:
    """
    Run an end-to-end search test with a given query.
    
    Args:
        query: The search query text
        filter_type: Optional filter for 'code' or 'text' results
        num_results: Number of results to return
        
    Returns:
        Dictionary containing the search results
    """
    logger.info(f"Running search test for query: '{query}'")
    print(f"\nTesting query: '{query}'")
    
    try:
        # Initialize components
        query_processor = QueryProcessor()
        result_formatter = ResultFormatter()
        db_manager = ChromaManager()
        
        # Process the query
        start_time = time.time()
        query_data = query_processor.process_query(query)
        query_time = time.time() - start_time
        
        # Prepare filters
        filters = {}
        if filter_type:
            filters["chunk_type"] = filter_type
            
        # Search the database
        start_time = time.time()
        results = db_manager.query(
            query_data["embedding"],
            n_results=num_results,
            filters=filters if filters else None
        )
        search_time = time.time() - start_time
        
        # Format the results
        start_time = time.time()
        formatted_results = result_formatter.format_results(results, query)
        format_time = time.time() - start_time
        
        # Rank results based on query type
        ranked_results = result_formatter.rank_results(
            formatted_results,
            query_data["is_code_query"]
        )
        
        # Calculate total time
        total_time = query_time + search_time + format_time
        
        # Add timing information
        ranked_results["timing"] = {
            "query_processing": query_time,
            "database_search": search_time,
            "result_formatting": format_time,
            "total": total_time
        }
        
        # Print results summary
        print(f"Query classified as: {'code query' if query_data['is_code_query'] else 'concept query'}")
        print(f"Found {len(ranked_results['results'])} results in {total_time:.4f} seconds")
        
        if ranked_results['results']:
            print("\nTop results:")
            for i, res in enumerate(ranked_results["results"][:3]):  # Show top 3
                print(f"\n--- Result {i+1} ({res['chunk_type']}) ---")
                print(f"Title: {res['title']}")
                print(f"Source: {res['source']}")
                print(f"Score: {res['score']:.4f}")
                print(f"Snippet: {res['snippet'][:100]}...")
                
        # Log result counts by type
        code_results = sum(1 for r in ranked_results["results"] if r["chunk_type"] == "code")
        text_results = sum(1 for r in ranked_results["results"] if r["chunk_type"] == "text")
        logger.info(f"Results breakdown: {code_results} code, {text_results} text")
        print(f"\nResults breakdown: {code_results} code, {text_results} text")
        
        return ranked_results
        
    except Exception as e:
        logger.error(f"Error during search test: {str(e)}")
        print(f"Error: {str(e)}")
        # Add detailed debug information
        import traceback
        logger.error(f"Error traceback: {traceback.format_exc()}")
        print(f"Results type: {type(results)}")
        print(f"Results structure: {str(results)[:200]}")
        return {
            "error": str(e),
            "query": query,
            "results": []
        }

def run_test_suite():
    """Run a suite of test queries to evaluate search functionality."""
    print("=== PyTorch Documentation Search - End-to-End Test Suite ===")
    
    # Define test queries
    test_queries = [
        # Code queries
        {"query": "How to implement a transformer block in PyTorch", "filter": "code"},
        {"query": "Custom autograd function example", "filter": "code"},
        {"query": "Implementing batch normalization in PyTorch", "filter": None},
        
        # Concept queries
        {"query": "What is autograd in PyTorch?", "filter": "text"},
        {"query": "Explain backpropagation in PyTorch", "filter": None},
        {"query": "Difference between nn.Module and nn.functional", "filter": None}
    ]
    
    # Run each test query
    results = {}
    for i, test in enumerate(test_queries):
        print(f"\n=== Test {i+1}/{len(test_queries)} ===")
        result = run_search_test(test["query"], test["filter"])
        results[test["query"]] = {
            "filter": test["filter"],
            "result_count": len(result.get("results", [])),
            "timing": result.get("timing", {})
        }
    
    # Print summary
    print("\n=== Test Suite Summary ===")
    print(f"Ran {len(test_queries)} test queries")
    
    total_time = sum(res["timing"].get("total", 0) for res in results.values())
    avg_time = total_time / len(results) if results else 0
    print(f"Average query time: {avg_time:.4f} seconds")
    
    # Save detailed results to file
    with open("search_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to search_test_results.json")
    
if __name__ == "__main__":
    run_test_suite()
