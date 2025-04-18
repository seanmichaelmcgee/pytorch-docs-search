#!/usr/bin/env python
"""
Flask API for PyTorch Documentation Search Tool.
MCP-compliant endpoint for Claude Code CLI integration.
"""

from flask import Flask, request, jsonify
import os
import sys
import json
import logging
import time
import traceback
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='flask_api.log'
)
logger = logging.getLogger("flask_api")

# Add project root to Python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import our search functionality
try:
    from scripts.search.query_processor import QueryProcessor
    from scripts.search.result_formatter import ResultFormatter
    from scripts.database.chroma_manager import ChromaManager
    from scripts.config import MAX_RESULTS, OPENAI_API_KEY
except ImportError as e:
    logger.error(f"Error importing search modules: {str(e)}")
    raise

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    """
    Process search requests from Claude Code CLI via MCP.
    
    Expected JSON input:
    {
        "query": "string",  // Required: Search query text
        "num_results": 5,   // Optional: Number of results to return
        "filter": "code"    // Optional: Filter for 'code' or 'text' results
    }
    
    Returns:
        JSON response with search results or error message
    """
    # Log request receipt
    logger.info(f"Received search request: {request.data.decode()[:100]}...")
    request_start_time = time.time()
    
    try:
        # Parse request JSON
        data = request.get_json()
        
        if not data or not isinstance(data, dict):
            return jsonify({
                "error": "Invalid request format: Expected JSON object",
                "results": [],
                "count": 0,
                "status": {
                    "complete": False,
                    "error_type": "invalid_request",
                    "is_partial": True
                }
            }), 400
            
        # Extract query (required)
        query = data.get('query')
        if not query or not isinstance(query, str):
            return jsonify({
                "error": "Missing or invalid 'query' parameter",
                "results": [],
                "count": 0,
                "status": {
                    "complete": False,
                    "error_type": "missing_query",
                    "is_partial": True
                }
            }), 400
            
        # Extract optional parameters
        num_results = int(data.get('num_results', MAX_RESULTS))
        filter_type = data.get('filter')
        
        # Log query details
        logger.info(f"Processing query: '{query}', num_results: {num_results}, filter: {filter_type}")
        
        # Set environment flag to indicate running from MCP
        os.environ['CLAUDE_MCP_TOOL'] = 'true'
        
        # Reuse the search functionality with progressive stages
        results = search_pytorch_docs(query, num_results, filter_type)
        
        # Add timing information for full request
        request_time = time.time() - request_start_time
        if "timing" not in results:
            results["timing"] = {}
        results["timing"]["total_request"] = request_time
        
        # Log completion
        result_count = len(results.get('results', []))
        logger.info(f"Query processed in {request_time:.2f}s with {result_count} results")
        
        # Return results
        return jsonify(results)
        
    except Exception as e:
        error_time = time.time() - request_start_time
        logger.error(f"Error processing search request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "results": [],
            "count": 0,
            "status": {
                "complete": False,
                "error_type": "server_exception",
                "is_partial": True
            },
            "timing": {"error_time": error_time}
        }), 500

def search_pytorch_docs(query: str, num_results: int = MAX_RESULTS, filter_type: str = None) -> Dict[str, Any]:
    """
    Search PyTorch documentation using the vector database with progressive stages.
    
    Args:
        query: The search query about PyTorch
        num_results: Number of results to return
        filter_type: Optional filter for 'code' or 'text' results
        
    Returns:
        Dictionary containing search results, potentially partial if errors occur
    """
    # Track timing for each stage
    timing = {}
    overall_start = time.time()
    
    try:
        logger.info(f"Searching for: '{query}'")
        
        # Initialize components with robust error handling
        try:
            init_start = time.time()
            query_processor = QueryProcessor()
            result_formatter = ResultFormatter()
            db_manager = ChromaManager()
            timing["initialization"] = time.time() - init_start
        except Exception as init_error:
            logger.error(f"Failed to initialize search components: {str(init_error)}")
            return {
                "error": f"Search initialization failed: {str(init_error)}",
                "query": query,
                "results": [],
                "count": 0,
                "status": {
                    "complete": False,
                    "error_type": "initialization_error",
                    "is_partial": True
                },
                "timing": {"initialization_error": time.time() - overall_start}
            }
        
        # Initialize result tracking for progressive stages
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
        
        # STAGE 1: Process the query
        stage1_start = time.time()
        try:
            logger.info("Stage 1: Processing query...")
            query_data = query_processor.process_query(query)
            
            # Track timing
            timing["query_processing"] = time.time() - stage1_start
            
            # Track successful completion
            partial_results["status"]["stages_completed"].append("query_processing")
            
            # Store essential query metadata for partial results
            partial_results["intent"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75)
            }
            
            logger.info(f"Query processed as {'code' if query_data['is_code_query'] else 'concept'} query " +
                      f"with confidence {query_data.get('intent_confidence', 0.75):.2f}")
            
        except Exception as e:
            timing["query_processing_error"] = time.time() - stage1_start
            logger.error(f"Query processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            partial_results["status"]["stages_timed_out"].append("query_processing")
            partial_results["status"]["is_partial"] = True
            partial_results["error"] = f"Query processing failed: {str(e)}"
            partial_results["timing"] = timing
            return partial_results
        
        # Cannot proceed without query data
        if not query_data:
            partial_results["error"] = "Failed to process query"
            partial_results["timing"] = timing
            return partial_results
        
        # STAGE 2: Search the database
        stage2_start = time.time()
        try:
            logger.info("Stage 2: Searching database...")
            
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
            
            # Track timing
            timing["database_search"] = time.time() - stage2_start
            
            # Track successful completion
            partial_results["status"]["stages_completed"].append("database_search")
            
        except Exception as e:
            timing["database_search_error"] = time.time() - stage2_start
            logger.error(f"Database search failed: {str(e)}")
            logger.error(traceback.format_exc())
            partial_results["status"]["stages_timed_out"].append("database_search")
            partial_results["status"]["is_partial"] = True
            partial_results["error"] = f"Database search failed: {str(e)}"
            
            # If we got query processing data, add it to the response
            partial_results["claude_context"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75),
                "query_description": "code-related" if query_data["is_code_query"] else "conceptual"
            }
            
            partial_results["timing"] = timing
            return partial_results
        
        # STAGE 3: Format and rank results
        stage3_start = time.time()
        try:
            logger.info("Stage 3: Formatting results...")
            
            # Format the results
            formatted_results = result_formatter.format_results(raw_results, query)
            
            # Rank results based on query type and confidence
            ranked_results = result_formatter.rank_results(
                formatted_results,
                query_data["is_code_query"],
                query_data.get("intent_confidence", 0.75)
            )
            
            # Track timing
            timing["result_formatting"] = time.time() - stage3_start
            timing["total_processing"] = time.time() - overall_start
            
            # Track successful completion
            partial_results["status"]["stages_completed"].append("result_formatting")
            partial_results["status"]["complete"] = True
            
            # Update with complete results
            result_count = len(ranked_results.get("results", []))
            logger.info(f"Found {result_count} results for query")
            
            # Add additional context for Claude
            ranked_results["claude_context"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75),
                "query_description": "code-related" if query_data["is_code_query"] else "conceptual",
                "is_comparative": ranked_results.get("is_comparative", False),
                "is_ambiguous": ranked_results.get("is_ambiguous", False)
            }
            
            # Mark the process as complete
            ranked_results["status"] = {
                "complete": True,
                "stages_completed": partial_results["status"]["stages_completed"],
                "is_partial": False
            }
            
            # Add timing information
            ranked_results["timing"] = timing
            
            return ranked_results
            
        except Exception as e:
            timing["result_formatting_error"] = time.time() - stage3_start
            logger.error(f"Result formatting failed: {str(e)}")
            logger.error(traceback.format_exc())
            partial_results["status"]["stages_timed_out"].append("result_formatting")
            partial_results["status"]["is_partial"] = True
            partial_results["error"] = f"Result formatting failed: {str(e)}"
            
            # Return partial results with raw database results if available
            if raw_results:
                # Create minimal formatted results
                basic_results = []
                
                try:
                    # Handle different result formats - support both old and new ChromaDB response format
                    if 'documents' in raw_results and raw_results['documents']:
                        documents = raw_results['documents'][0] if isinstance(raw_results['documents'], list) and isinstance(raw_results['documents'][0], list) else raw_results['documents']
                        metadatas = raw_results['metadatas'][0] if isinstance(raw_results['metadatas'], list) and isinstance(raw_results['metadatas'][0], list) else raw_results['metadatas']
                        distances = raw_results['distances'][0] if isinstance(raw_results['distances'], list) and isinstance(raw_results['distances'][0], list) else raw_results['distances']
                        
                        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                            # Convert distance to similarity score
                            similarity = 1.0 - distance if isinstance(distance, (int, float)) else 0.5
                            
                            # Handle metadata format differences
                            if isinstance(metadata, dict):
                                title = metadata.get("title", f"Result {i+1}")
                                source = metadata.get("source", "Unknown")
                                chunk_type = metadata.get("chunk_type", "unknown")
                            else:
                                # Use sensible defaults
                                title = f"Result {i+1}"
                                source = "Unknown"
                                chunk_type = "unknown"
                            
                            basic_results.append({
                                "title": title,
                                "snippet": doc[:200] + "..." if len(doc) > 200 else doc,
                                "source": source,
                                "chunk_type": chunk_type,
                                "score": similarity
                            })
                except Exception as nested_e:
                    logger.error(f"Error creating simplified results: {str(nested_e)}")
                    logger.error(traceback.format_exc())
                
                partial_results["results"] = basic_results
                partial_results["count"] = len(basic_results)
            
            # Add query intent information
            partial_results["claude_context"] = {
                "is_code_query": query_data["is_code_query"],
                "intent_confidence": query_data.get("intent_confidence", 0.75),
                "query_description": "code-related" if query_data["is_code_query"] else "conceptual"
            }
            
            partial_results["timing"] = timing
            return partial_results
            
    except Exception as e:
        total_time = time.time() - overall_start
        logger.error(f"Unhandled error during search: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "query": query,
            "results": [],
            "count": 0,
            "status": {
                "complete": False,
                "error_type": "exception", 
                "is_partial": True
            },
            "timing": {
                "error": total_time
            }
        }

if __name__ == '__main__':
    # Print status message
    print("\n=== PyTorch Documentation Search API ===")
    print("Server running at: http://localhost:5000/search")
    print("Register with Claude Code CLI using:")
    print("claude mcp add mcp__pytorch_docs__semantic_search http://localhost:5000/search --transport sse")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Log startup
    logger.info("Starting PyTorch Documentation Search API")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)