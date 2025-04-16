#!/usr/bin/env python3

import sys
import json
import os
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='claude_code_tool.log'
)
logger = logging.getLogger("claude_code_tool")

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

# Import search functionality
from scripts.search.query_processor import QueryProcessor
from scripts.search.result_formatter import ResultFormatter
from scripts.database.chroma_manager import ChromaManager
from scripts.config import MAX_RESULTS

def search_pytorch_docs(query: str, num_results: int = MAX_RESULTS, filter_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Search PyTorch documentation using the vector database.
    
    Args:
        query: The search query about PyTorch
        num_results: Number of results to return
        filter_type: Optional filter for 'code' or 'text' results
        
    Returns:
        Dictionary containing search results
    """
    try:
        logger.info(f"Received search query: '{query}'")
        
        # Initialize components
        query_processor = QueryProcessor()
        result_formatter = ResultFormatter()
        db_manager = ChromaManager()
        
        # Process the query
        logger.info("Processing query...")
        query_data = query_processor.process_query(query)
        
        # Prepare filters
        filters = {}
        if filter_type:
            filters["chunk_type"] = filter_type
            logger.info(f"Using filter: chunk_type={filter_type}")
        
        # Search the database
        logger.info("Searching database...")
        results = db_manager.query(
            query_data["embedding"],
            n_results=num_results,
            filters=filters if filters else None
        )
        
        # Format the results
        logger.info("Formatting results...")
        formatted_results = result_formatter.format_results(results, query)
        
        # Rank results based on query type
        ranked_results = result_formatter.rank_results(
            formatted_results,
            query_data["is_code_query"]
        )
        
        # Log result summary
        result_count = len(ranked_results.get("results", []))
        logger.info(f"Found {result_count} results for query")
        
        # Add additional context for Claude
        ranked_results["claude_context"] = {
            "is_code_query": query_data["is_code_query"],
            "query_description": "code-related" if query_data["is_code_query"] else "conceptual"
        }
        
        return ranked_results
        
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return {
            "error": str(e),
            "query": query,
            "results": [],
            "count": 0
        }

def process_mcp_input() -> Dict[str, Any]:
    """
    Process input according to the Model-Context Protocol (MCP).
    
    Returns:
        Processed tool response
    """
    try:
        # Read input from stdin (MCP sends a JSON object)
        logger.info("Reading input from stdin...")
        stdin_data = sys.stdin.read()
        logger.info(f"Received stdin data: {stdin_data[:100]}")
        
        try:
            input_data = json.loads(stdin_data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}, input: {stdin_data[:100]}")
            return {"error": f"Invalid JSON input: {str(e)}", "results": [], "count": 0}
        
        # Extract query and parameters
        query = input_data.get("query", "")
        if not query:
            logger.warning("No query provided in input")
            return {"error": "No query provided", "results": [], "count": 0}
        
        # Get optional parameters
        num_results = input_data.get("num_results", MAX_RESULTS)
        filter_type = input_data.get("filter", None)
        
        # Log received parameters
        logger.info(f"Query: '{query}'")
        logger.info(f"Number of results: {num_results}")
        if filter_type:
            logger.info(f"Filter type: {filter_type}")
        
        # Perform search
        return search_pytorch_docs(query, num_results, filter_type)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {str(e)}")
        return {"error": "Invalid JSON input", "results": [], "count": 0}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e), "results": [], "count": 0}

def main():
    """Main entry point for the Claude Code tool."""
    try:
        # Process the input
        result = process_mcp_input()
        
        # Output results as JSON to stdout
        json_result = json.dumps(result)
        print(json_result)
        
        # Log completion
        logger.info("Successfully completed search request")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        error_response = json.dumps({"error": f"Fatal error: {str(e)}", "results": [], "count": 0})
        print(error_response)

if __name__ == "__main__":
    main()
