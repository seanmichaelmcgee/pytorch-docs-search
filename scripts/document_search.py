#!/usr/bin/env python3

import sys
import os
import json
import argparse
from typing import Dict, Any, Optional

from scripts.search.query_processor import QueryProcessor
from scripts.search.result_formatter import ResultFormatter
from scripts.database.chroma_manager import ChromaManager
from scripts.config import MAX_RESULTS

class DocumentSearch:
    def __init__(self, db_dir: Optional[str] = None, collection_name: Optional[str] = None):
        """Initialize the document search system."""
        # Initialize components
        self.query_processor = QueryProcessor()
        self.result_formatter = ResultFormatter()
        self.db_manager = ChromaManager(db_dir, collection_name)
    
    def search(self, query: str, num_results: int = MAX_RESULTS, filter_type: Optional[str] = None) -> Dict[str, Any]:
        """Search for documents matching the query."""
        try:
            # Process the query
            query_data = self.query_processor.process_query(query)
            
            # Prepare filters
            filters = {}
            if filter_type:
                filters["chunk_type"] = filter_type
            
            # Search the database
            results = self.db_manager.query(
                query_data["embedding"],
                n_results=num_results,
                filters=filters if filters else None
            )
            
            # Format the results
            formatted_results = self.result_formatter.format_results(results, query)
            
            # Rank results based on query type
            ranked_results = self.result_formatter.rank_results(
                formatted_results,
                query_data["is_code_query"]
            )
            
            return ranked_results
            
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "results": []
            }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Search PyTorch documentation')
    parser.add_argument('query', nargs='?', help='The search query')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--results', '-n', type=int, default=MAX_RESULTS, help='Number of results to return')
    parser.add_argument('--filter', '-f', choices=['code', 'text'], help='Filter results by type')
    args = parser.parse_args()
    
    # Initialize search
    search = DocumentSearch()
    
    if args.interactive:
        # Interactive mode
        print("PyTorch Documentation Search (type 'exit' to quit)")
        while True:
            query = input("\nEnter search query: ")
            if query.lower() in ('exit', 'quit'):
                break
            
            results = search.search(query, args.results, args.filter)
            
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print(f"\nFound {len(results['results'])} results for '{query}':")
                
                for i, res in enumerate(results["results"]):
                    print(f"\n--- Result {i+1} ({res['chunk_type']}) ---")
                    print(f"Title: {res['title']}")
                    print(f"Source: {res['source']}")
                    print(f"Score: {res['score']:.4f}")
                    print(f"Snippet: {res['snippet']}")
    
    elif args.query:
        # Direct query mode
        results = search.search(args.query, args.results, args.filter)
        print(json.dumps(results, indent=2))
    
    else:
        # Read from stdin (for Claude Code tool integration)
        query = sys.stdin.read().strip()
        if query:
            results = search.search(query, args.results)
            print(json.dumps(results))
        else:
            print(json.dumps({"error": "No query provided", "results": []}))

if __name__ == "__main__":
    main()