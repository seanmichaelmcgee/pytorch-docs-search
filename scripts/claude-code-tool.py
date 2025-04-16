#!/usr/bin/env python3

import sys
import json
import os
import logging
import signal
import time
from typing import Dict, Any, Optional, List

# Global state for custom timeout stages
_current_stages = None

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
    Search PyTorch documentation using the vector database with progressive timeout.
    
    Args:
        query: The search query about PyTorch
        num_results: Number of results to return
        filter_type: Optional filter for 'code' or 'text' results
        
    Returns:
        Dictionary containing search results, potentially partial if timeouts occur
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
        
        # Initialize result tracking for progressive timeout
        partial_results = {
            "query": query,
            "results": [],
            "count": 0,
            "status": {
                "complete": False,
                "stages_completed": [],
                "stages_timed_out": [],
                "is_partial": False
            }
        }
        
        # Define the stages and their timeouts (use custom timeouts if available)
        global _current_stages
        if _current_stages:
            stages = _current_stages
            logger.info(f"Using custom timeouts: {stages}")
        else:
            stages = [
                {"name": "query_processing", "timeout": 2},  # 2 seconds for query processing
                {"name": "database_search", "timeout": 5},   # 5 seconds for database search
                {"name": "result_formatting", "timeout": 1}  # 1 second for result formatting
            ]
        
        query_data = None
        raw_results = None
        formatted_results = None
        
        # Setup signal handler for timeouts
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # Stage 1: Process the query
        try:
            logger.info("Stage 1: Processing query...")
            signal.alarm(stages[0]["timeout"])
            
            query_data = query_processor.process_query(query)
            
            # Track successful completion
            signal.alarm(0)
            partial_results["status"]["stages_completed"].append("query_processing")
            
            # Store essential query metadata for partial results
            partial_results["intent"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75)
            }
            
        except TimeoutError:
            logger.warning("Query processing timed out")
            signal.alarm(0)
            partial_results["status"]["stages_timed_out"].append("query_processing")
            partial_results["status"]["is_partial"] = True
            partial_results["error"] = "Query processing timed out after 2 seconds"
            return partial_results
        
        # Cannot proceed without query data
        if not query_data:
            partial_results["error"] = "Failed to process query"
            return partial_results
        
        # Stage 2: Search the database
        try:
            logger.info("Stage 2: Searching database...")
            signal.alarm(stages[1]["timeout"])
            
            # Prepare filters
            filters = {}
            if filter_type:
                filters["chunk_type"] = filter_type
                logger.info(f"Using filter: chunk_type={filter_type}")
            
            # Search the database
            raw_results = db_manager.query(
                query_data["embedding"],
                n_results=num_results,
                filters=filters if filters else None
            )
            
            # Track successful completion
            signal.alarm(0)
            partial_results["status"]["stages_completed"].append("database_search")
            
        except TimeoutError:
            logger.warning("Database search timed out")
            signal.alarm(0)
            partial_results["status"]["stages_timed_out"].append("database_search")
            partial_results["status"]["is_partial"] = True
            partial_results["error"] = "Database search timed out after 5 seconds"
            
            # If we got query processing data, add it to the response
            partial_results["claude_context"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75),
                "query_description": "code-related" if query_data["is_code_query"] else "conceptual"
            }
            
            return partial_results
        
        # Stage 3: Format and rank results
        try:
            logger.info("Stage 3: Formatting results...")
            signal.alarm(stages[2]["timeout"])
            
            # Format the results
            formatted_results = result_formatter.format_results(raw_results, query)
            
            # Rank results based on query type and confidence
            ranked_results = result_formatter.rank_results(
                formatted_results,
                query_data["is_code_query"],
                query_data.get("intent_confidence", 0.75)  # Use confidence if available
            )
            
            # Track successful completion
            signal.alarm(0)
            partial_results["status"]["stages_completed"].append("result_formatting")
            partial_results["status"]["complete"] = True
            
            # Update with complete results
            result_count = len(ranked_results.get("results", []))
            logger.info(f"Found {result_count} results for query")
            
            # Add additional context for Claude
            ranked_results["claude_context"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75),
                "query_description": "code-related" if query_data["is_code_query"] else "conceptual"
            }
            
            # Mark the process as complete
            ranked_results["status"] = {
                "complete": True,
                "stages_completed": partial_results["status"]["stages_completed"],
                "is_partial": False
            }
            
            return ranked_results
            
        except TimeoutError:
            logger.warning("Result formatting timed out")
            signal.alarm(0)
            partial_results["status"]["stages_timed_out"].append("result_formatting")
            partial_results["status"]["is_partial"] = True
            partial_results["error"] = "Result formatting timed out after 1 second"
            
            # Return partial results with raw database results if available
            if raw_results:
                # Create minimal formatted results
                basic_results = []
                for i, (doc_id, score) in enumerate(zip(raw_results['ids'][0], raw_results['distances'][0])):
                    # Convert distance to score (1 - distance for cosine similarity)
                    similarity_score = 1.0 - score
                    
                    # Extract basic metadata
                    metadata = raw_results['metadatas'][0][i]
                    
                    basic_results.append({
                        "id": doc_id,
                        "score": similarity_score,
                        "chunk_type": metadata.get("chunk_type", "unknown"),
                        "title": metadata.get("title", "Unknown Title"),
                        "source": metadata.get("source", "Unknown Source"),
                        "snippet": raw_results['documents'][0][i][:200] + "..." if len(raw_results['documents'][0][i]) > 200 else raw_results['documents'][0][i]
                    })
                
                partial_results["results"] = basic_results
                partial_results["count"] = len(basic_results)
            
            # Add query intent information
            partial_results["claude_context"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75),
                "query_description": "code-related" if query_data["is_code_query"] else "conceptual"
            }
            
            return partial_results
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return {
            "error": str(e),
            "query": query,
            "results": [],
            "count": 0,
            "status": {
                "complete": False,
                "error_type": "exception",
                "is_partial": True
            }
        }

def process_mcp_input() -> Dict[str, Any]:
    """
    Process input according to the Model-Context Protocol (MCP).
    
    Returns:
        Processed tool response with progressive timeout support
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
            return {
                "error": f"Invalid JSON input: {str(e)}", 
                "results": [], 
                "count": 0,
                "status": {"complete": False, "error_type": "json_parse", "is_partial": True}
            }
        
        # Extract query and parameters
        query = input_data.get("query", "")
        if not query:
            logger.warning("No query provided in input")
            return {
                "error": "No query provided", 
                "results": [], 
                "count": 0,
                "status": {"complete": False, "error_type": "missing_query", "is_partial": True}
            }
        
        # Get optional parameters
        num_results = input_data.get("num_results", MAX_RESULTS)
        filter_type = input_data.get("filter", None)
        
        # Get custom timeouts if provided
        custom_timeouts = input_data.get("timeouts", {})
        query_timeout = custom_timeouts.get("query_processing", 2)  # Default 2s
        search_timeout = custom_timeouts.get("database_search", 5)   # Default 5s
        format_timeout = custom_timeouts.get("result_formatting", 1) # Default 1s
        
        # Log received parameters
        logger.info(f"Query: '{query}'")
        logger.info(f"Number of results: {num_results}")
        if filter_type:
            logger.info(f"Filter type: {filter_type}")
        if custom_timeouts:
            logger.info(f"Custom timeouts: {custom_timeouts}")
        
        # Set up custom stages with provided timeouts
        stages = [
            {"name": "query_processing", "timeout": query_timeout},
            {"name": "database_search", "timeout": search_timeout},
            {"name": "result_formatting", "timeout": format_timeout}
        ]
        
        # Store stages in global state for the search function to use
        global _current_stages
        _current_stages = stages
        
        # Perform search with progressive timeout
        result = search_pytorch_docs(query, num_results, filter_type)
        
        # Add timestamp and query metadata
        result["timestamp"] = time.time()
        result["count"] = len(result.get("results", []))
        
        # Add partial result notification if needed
        if result.get("status", {}).get("is_partial", False):
            logger.warning("Returning partial results due to timeout or error")
            result["warning"] = "This response contains partial results due to operation timeout."
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {str(e)}")
        return {
            "error": "Invalid JSON input", 
            "results": [], 
            "count": 0,
            "status": {"complete": False, "error_type": "json_parse", "is_partial": True}
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "error": str(e), 
            "results": [], 
            "count": 0,
            "status": {"complete": False, "error_type": "exception", "is_partial": True}
        }

def main():
    """Main entry point for the Claude Code tool with progressive timeout support."""
    try:
        # Process the input
        result = process_mcp_input()
        
        # Output results as JSON to stdout
        json_result = json.dumps(result)
        print(json_result)
        
        # Log completion with status
        if result.get("status", {}).get("is_partial", False):
            logger.info("Completed search request with partial results")
            logger.info(f"Stages completed: {result.get('status', {}).get('stages_completed', [])}")
            logger.info(f"Stages timed out: {result.get('status', {}).get('stages_timed_out', [])}")
        else:
            logger.info("Successfully completed search request with full results")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        error_response = json.dumps({
            "error": f"Fatal error: {str(e)}", 
            "results": [], 
            "count": 0,
            "status": {
                "complete": False,
                "error_type": "fatal_exception",
                "is_partial": True
            }
        })
        print(error_response)

if __name__ == "__main__":
    main()
