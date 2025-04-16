#!/usr/bin/env python3

import sys
import json
import os
import logging
import signal
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='claude_code_tool.log'
)
logger = logging.getLogger("claude_code_tool")

# Timeout handler for operations
class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

# Import search functionality
try:
    from scripts.search.query_processor import QueryProcessor
    from scripts.search.result_formatter import ResultFormatter
    from scripts.database.chroma_manager import ChromaManager
    from scripts.config import MAX_RESULTS, OPENAI_API_KEY
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    # Try alternative import paths (for when running directly)
    try:
        sys.path.append(script_dir)
        from search.query_processor import QueryProcessor
        from search.result_formatter import ResultFormatter
        from database.chroma_manager import ChromaManager
        from config import MAX_RESULTS, OPENAI_API_KEY
    except ImportError as e2:
        logger.error(f"Alternative import path also failed: {e2}")
        # Fallback defaults
        MAX_RESULTS = 5
        OPENAI_API_KEY = None

# Check for API key availability
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment, checking user config")
    try:
        home_dir = os.path.expanduser("~")
        config_path = os.path.join(home_dir, ".pytorch_docs_config")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                OPENAI_API_KEY = config.get("openai_api_key")
                if OPENAI_API_KEY:
                    logger.info("Found API key in user config")
                    # Set environment variable for modules that need it
                    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                else:
                    logger.error("No API key found in user config")
    except Exception as e:
        logger.error(f"Error loading fallback configuration: {e}")

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
        
        # Check if running as Claude Code tool and limit results if needed
        is_claude_tool = 'CLAUDE_MCP_TOOL' in os.environ
        if is_claude_tool:
            # Restrict results when running in Claude to optimize context window usage
            num_results = min(num_results, 3)
            logger.info(f"Running as Claude tool, limiting results to {num_results}")
        
        # Initialize components
        query_processor = QueryProcessor()
        result_formatter = ResultFormatter()
        db_manager = ChromaManager()
        
        # Set up timeout for operations (10 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        
        try:
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
            
            # Cancel alarm as operations completed successfully
            signal.alarm(0)
            
            # Log result summary
            result_count = len(ranked_results.get("results", []))
            logger.info(f"Found {result_count} results for query")
            
            # Add additional context for Claude
            ranked_results["claude_context"] = {
                "is_code_query": query_data["is_code_query"],
                "query_description": "code-related" if query_data["is_code_query"] else "conceptual"
            }
            
            return ranked_results
            
        except TimeoutError:
            logger.error("Search operation timed out")
            signal.alarm(0)  # Cancel the alarm
            return {
                "error": "Search operation timed out after 10 seconds",
                "query": query,
                "results": [],
                "count": 0,
                "timeout": True
            }
            
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
