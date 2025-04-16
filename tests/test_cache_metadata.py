#!/usr/bin/env python3

"""
This is a test script for examining caching and metadata handling in the PyTorch Documentation Search Tool.
It provides detailed output on cache performance, metadata handling, and ChromaDB integration.
"""

import os
import sys
import json
import time
import logging
import unittest
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cache_metadata_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_cache_metadata")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

# Import components to test
from scripts.embedding.cache import EmbeddingCache
from scripts.embedding.generator import EmbeddingGenerator
from scripts.search.result_formatter import ResultFormatter
from scripts.database.chroma_manager import ChromaManager
from scripts.search.query_processor import QueryProcessor
from scripts.config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, DB_DIR

class TestCacheMetadata(unittest.TestCase):
    """Test suite for caching and metadata handling."""
    
    def setUp(self):
        """Setup for tests."""
        self.test_cache_dir = os.path.join(script_dir, "test_cache")
        os.makedirs(self.test_cache_dir, exist_ok=True)
        self.cleanup_files = []
        
        # Test data
        self.test_text = "Testing the PyTorch embedding cache functionality with a unique string " + str(time.time())
        self.test_query = "How to implement batch normalization in PyTorch"
        self.test_model = EMBEDDING_MODEL
        self.test_result = {
            "documents": [["This is a test document with batch normalization example."]],
            "metadatas": [[{"title": "Test Document", "source": "test.md", "chunk_type": "code"}]],
            "distances": [[0.2]]
        }
        self.test_result_flat = {
            "documents": ["This is a test document with batch normalization example."],
            "metadatas": [{"title": "Test Document", "source": "test.md", "chunk_type": "code"}],
            "distances": [0.2]
        }
        self.test_result_list_metadata = {
            "documents": ["This is a test document with batch normalization example."],
            "metadatas": [[{"title": "Test Document", "source": "test.md", "chunk_type": "code"}]],
            "distances": [0.2]
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files
        for file_path in self.cleanup_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove test cache directory
        import shutil
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_embedding_cache(self):
        """Test the EmbeddingCache component with detailed output."""
        logger.info("=== Testing EmbeddingCache Component ===")
        
        # Create a cache instance
        cache = EmbeddingCache(self.test_cache_dir, max_size_gb=0.01)  # Small size for testing
        logger.info(f"Created cache in {self.test_cache_dir}")
        
        # Generate a fake embedding
        test_embedding = [0.1] * EMBEDDING_DIMENSIONS
        logger.info(f"Created test embedding with {len(test_embedding)} dimensions")
        
        # Save to cache
        cache.set(self.test_text, self.test_model, test_embedding)
        logger.info("Saved embedding to cache")
        
        # Get cache stats
        stats = cache.get_stats()
        logger.info(f"Cache stats after save: {stats}")
        
        # Verify file was created
        text_hash = cache._get_hash(self.test_text, self.test_model)
        cache_file = os.path.join(self.test_cache_dir, text_hash + ".json")
        self.cleanup_files.append(cache_file)
        
        self.assertTrue(os.path.exists(cache_file), f"Cache file was not created at {cache_file}")
        logger.info(f"Verified cache file exists at {cache_file}")
        
        # Check file content
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        logger.info(f"Cache file content keys: {list(cache_data.keys())}")
        self.assertEqual(cache_data["model"], self.test_model)
        self.assertEqual(len(cache_data["embedding"]), EMBEDDING_DIMENSIONS)
        
        # Get from cache
        cached_embedding = cache.get(self.test_text, self.test_model)
        self.assertIsNotNone(cached_embedding)
        logger.info("Successfully retrieved embedding from cache")
        
        # Verify hit/miss counts
        stats = cache.get_stats()
        logger.info(f"Cache stats after retrieval: {stats}")
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)
        
        # Try with non-existent entry
        missing_embedding = cache.get("This text is not in cache", self.test_model)
        self.assertIsNone(missing_embedding)
        logger.info("Correctly returned None for missing cache entry")
        
        # Verify miss count
        stats = cache.get_stats()
        logger.info(f"Cache stats after miss: {stats}")
        self.assertEqual(stats["misses"], 1)
        
        # Test cache pruning
        for i in range(10):
            # Add multiple entries to trigger pruning
            cache.set(f"Test text {i}", self.test_model, test_embedding)
            
        # Check stats after multiple additions
        stats = cache.get_stats()
        logger.info(f"Cache stats after multiple additions (with potential pruning): {stats}")
        
        # Load a new cache instance from the same directory to test persistence
        new_cache = EmbeddingCache(self.test_cache_dir)
        new_stats = new_cache.get_stats()
        logger.info(f"Stats after reloading cache from disk: {new_stats}")
        
        # Test getting an entry from the reloaded cache
        reloaded_embedding = new_cache.get(f"Test text 9", self.test_model)
        if reloaded_embedding:
            logger.info("Successfully retrieved embedding from reloaded cache")
        else:
            logger.info("Entry not found in reloaded cache (may have been pruned)")
        
        logger.info("=== EmbeddingCache Test Complete ===")
    
    def test_result_formatter_metadata_handling(self):
        """Test the ResultFormatter's handling of different metadata formats."""
        logger.info("=== Testing ResultFormatter Metadata Handling ===")
        
        formatter = ResultFormatter()
        logger.info("Created ResultFormatter instance")
        
        # Test with nested list format (old ChromaDB format)
        logger.info("Testing with nested list format (old ChromaDB format)")
        formatted_results = formatter.format_results(self.test_result, self.test_query)
        logger.info(f"Formatted results keys: {list(formatted_results.keys())}")
        logger.info(f"Result count: {formatted_results['count']}")
        
        self.assertEqual(formatted_results["count"], 1)
        self.assertEqual(formatted_results["results"][0]["title"], "Test Document")
        self.assertEqual(formatted_results["results"][0]["chunk_type"], "code")
        
        # Test with flat list format (newer ChromaDB format)
        logger.info("Testing with flat list format (newer ChromaDB format)")
        formatted_results = formatter.format_results(self.test_result_flat, self.test_query)
        logger.info(f"Formatted results with flat format: {formatted_results}")
        
        self.assertEqual(formatted_results["count"], 1)
        self.assertEqual(formatted_results["results"][0]["title"], "Test Document")
        
        # Test with list-based metadata
        logger.info("Testing with list-based metadata format")
        formatted_results = formatter.format_results(self.test_result_list_metadata, self.test_query)
        logger.info(f"Formatted results with list metadata: {formatted_results}")
        
        self.assertEqual(formatted_results["count"], 1)
        self.assertEqual(formatted_results["results"][0]["title"], "Test Document")
        
        # Test with malformed metadata
        logger.info("Testing with malformed metadata")
        malformed_result = {
            "documents": ["Test document content"],
            "metadatas": [None],  # Missing metadata
            "distances": [0.2]
        }
        
        formatted_results = formatter.format_results(malformed_result, self.test_query)
        logger.info(f"Formatted results with malformed metadata: {formatted_results}")
        
        self.assertEqual(formatted_results["count"], 1)
        self.assertEqual(formatted_results["results"][0]["title"], "Result 1")  # Should use default
        
        # Test with no results
        logger.info("Testing with no results")
        empty_result = {
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        formatted_results = formatter.format_results(empty_result, self.test_query)
        logger.info(f"Formatted results with no results: {formatted_results}")
        
        self.assertEqual(formatted_results["count"], 0)
        
        logger.info("=== ResultFormatter Metadata Test Complete ===")
    
    def test_query_processor_embedding_integration(self):
        """Test the integration of QueryProcessor with embedding generation."""
        logger.info("=== Testing QueryProcessor with Embedding Integration ===")
        
        try:
            query_processor = QueryProcessor()
            logger.info("Created QueryProcessor instance")
            
            # Process a query
            logger.info(f"Processing query: '{self.test_query}'")
            query_data = query_processor.process_query(self.test_query)
            
            # Check the results
            logger.info(f"Query data keys: {list(query_data.keys())}")
            logger.info(f"Is code query: {query_data['is_code_query']}")
            
            self.assertTrue(query_data["is_code_query"])  # This should be detected as a code query
            self.assertIn("embedding", query_data)
            self.assertEqual(len(query_data["embedding"]), EMBEDDING_DIMENSIONS)
            
            logger.info(f"Embedding dimensions: {len(query_data['embedding'])}")
            
            # Test with a concept query
            concept_query = "What is batch normalization in deep learning?"
            logger.info(f"Processing concept query: '{concept_query}'")
            
            concept_data = query_processor.process_query(concept_query)
            logger.info(f"Is code query: {concept_data['is_code_query']}")
            
            self.assertFalse(concept_data["is_code_query"])  # This should be detected as a concept query
            
            # Test embedding cache integration
            logger.info("Testing embedding cache integration")
            # Process the same query again
            start_time = time.time()
            query_data_repeat = query_processor.process_query(self.test_query)
            end_time = time.time()
            
            logger.info(f"Query processing time (likely cache hit): {end_time - start_time:.6f} seconds")
            
            # Embedding should be identical
            self.assertEqual(query_data["embedding"], query_data_repeat["embedding"])
            
        except Exception as e:
            logger.error(f"Error in query processor test: {str(e)}")
            self.fail(f"QueryProcessor test failed with error: {str(e)}")
        
        logger.info("=== QueryProcessor Test Complete ===")
    
    def test_chroma_manager_metadata_handling(self):
        """Test ChromaManager's handling of metadata."""
        logger.info("=== Testing ChromaManager Metadata Handling ===")
        
        # Use a separate test directory for ChromaDB
        test_db_dir = os.path.join(script_dir, "test_chroma_db")
        os.makedirs(test_db_dir, exist_ok=True)
        self.cleanup_files.append(test_db_dir)
        
        try:
            # Create a ChromaManager instance
            chroma_manager = ChromaManager(persist_directory=test_db_dir, collection_name="test_collection")
            logger.info(f"Created ChromaManager with test DB at {test_db_dir}")
            
            # Reset the collection to ensure a clean state
            chroma_manager.reset_collection()
            logger.info("Reset test collection")
            
            # Create test chunks
            test_chunks = [
                {
                    "id": "chunk1",
                    "text": "This is a test chunk for PyTorch documentation search.",
                    "embedding": [0.1] * EMBEDDING_DIMENSIONS,
                    "metadata": {
                        "title": "Test Chunk 1",
                        "source": "test1.md",
                        "chunk_type": "text",
                        "chunk": 1
                    }
                },
                {
                    "id": "chunk2",
                    "text": "def batch_norm_example(x):\n    return nn.BatchNorm2d(x)",
                    "embedding": [0.2] * EMBEDDING_DIMENSIONS,
                    "metadata": {
                        "title": "Test Chunk 2",
                        "source": "test2.md",
                        "chunk_type": "code",
                        "chunk": 1
                    }
                }
            ]
            
            # Add chunks to the collection
            logger.info(f"Adding {len(test_chunks)} test chunks to ChromaDB")
            chroma_manager.add_chunks(test_chunks)
            
            # Query the collection
            logger.info("Querying ChromaDB with test embedding")
            query_embedding = [0.2] * EMBEDDING_DIMENSIONS  # Should be more similar to the second chunk
            results = chroma_manager.query(query_embedding, n_results=2)
            
            # Log the results structure
            logger.info(f"Query result keys: {list(results.keys())}")
            for key in results:
                if isinstance(results[key], list):
                    logger.info(f"{key} is a list of length {len(results[key])}")
                    if results[key] and isinstance(results[key][0], list):
                        logger.info(f"{key}[0] is a list of length {len(results[key][0])}")
            
            # Examine metadata structure
            if 'metadatas' in results:
                metadata_list = results['metadatas']
                logger.info(f"Metadata list type: {type(metadata_list)}")
                
                if isinstance(metadata_list, list) and metadata_list:
                    logger.info(f"First metadata entry type: {type(metadata_list[0])}")
                    logger.info(f"First metadata entry content: {metadata_list[0]}")
            
            # Format the results to see how metadata is handled
            formatter = ResultFormatter()
            formatted_results = formatter.format_results(results, "test query")
            
            logger.info(f"Formatted result count: {formatted_results['count']}")
            if formatted_results["count"] > 0:
                for i, result in enumerate(formatted_results["results"]):
                    logger.info(f"Result {i+1} - Title: {result['title']}, Type: {result['chunk_type']}")
            
            # Test with filters
            logger.info("Testing ChromaDB queries with filters")
            
            # Query for code chunks only
            code_results = chroma_manager.query(
                query_embedding, 
                n_results=2,
                filters={"chunk_type": "code"}
            )
            
            formatter = ResultFormatter()
            code_formatted = formatter.format_results(code_results, "test query")
            
            logger.info(f"Code filtered results count: {code_formatted['count']}")
            if code_formatted["count"] > 0:
                for i, result in enumerate(code_formatted["results"]):
                    logger.info(f"Code result {i+1} - Title: {result['title']}, Type: {result['chunk_type']}")
                    # Verify filter worked correctly
                    self.assertEqual(result["chunk_type"], "code")
            
        except Exception as e:
            logger.error(f"Error in ChromaManager test: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"ChromaManager test failed with error: {str(e)}")
        finally:
            # Clean up the test DB directory
            import shutil
            if os.path.exists(test_db_dir):
                shutil.rmtree(test_db_dir)
            logger.info("Cleaned up test ChromaDB directory")
        
        logger.info("=== ChromaManager Test Complete ===")
    
    def test_claude_code_tool_input_parsing(self):
        """Test the input parsing of the Claude Code tool."""
        logger.info("=== Testing Claude Code Tool Input Parsing ===")
        
        try:
            # Import the tool functions directly
            from scripts.claude_code_tool import process_mcp_input
            logger.info("Imported Claude Code tool functions")
        except ImportError:
            # Try with hyphen instead of underscore
            try:
                sys.path.append(os.path.join(script_dir, "../scripts"))
                from claude_code_tool import process_mcp_input
                logger.info("Imported Claude Code tool functions")
            except ImportError:
                logger.warning("Could not import claude_code_tool, trying with direct import")
                try:
                    from claude_code_tool import process_mcp_input
                    logger.info("Imported Claude Code tool with direct import")
                except ImportError:
                    logger.error("Could not import Claude Code tool functions")
                    self.skipTest("Claude Code tool module not found")
        
        # Test with various input formats
        test_inputs = [
            # Valid input
            json.dumps({"query": "How to implement batch normalization in PyTorch"}),
            # Input with additional parameters
            json.dumps({"query": "PyTorch DataLoader examples", "num_results": 3, "filter": "code"}),
            # Invalid input - missing query
            json.dumps({"filter": "code"}),
            # Invalid input - malformed JSON
            "{query: 'This is not valid JSON'}",
            # Empty input
            ""
        ]
        
        for i, test_input in enumerate(test_inputs):
            logger.info(f"\nTesting input format {i+1}: {test_input[:50]}...")
            
            # Set up stdin to read from string
            original_stdin = sys.stdin
            sys.stdin = type('StringIOWrapper', (), {
                'read': lambda self: test_input,
                'close': lambda self: None
            })()
            
            try:
                # Process the input
                result = process_mcp_input()
                logger.info(f"Result keys: {list(result.keys())}")
                
                # Check for error
                if "error" in result:
                    logger.warning(f"Error in result: {result['error']}")
                else:
                    logger.info(f"Success: Found {result.get('count', 0)} results")
                
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")
            finally:
                # Restore stdin
                sys.stdin = original_stdin
        
        logger.info("=== Claude Code Tool Input Test Complete ===")

def run_tests():
    """Run the test suite."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

def run_tests_with_detailed_output():
    """Run tests with detailed output about caching and metadata."""
    logger.info("================================")
    logger.info("Beginning Cache and Metadata Test Suite")
    logger.info("================================")
    
    # Test embedding cache
    cache_test = TestCacheMetadata('test_embedding_cache')
    cache_test.setUp()
    cache_test.test_embedding_cache()
    cache_test.tearDown()
    
    # Test result formatter
    formatter_test = TestCacheMetadata('test_result_formatter_metadata_handling')
    formatter_test.setUp()
    formatter_test.test_result_formatter_metadata_handling()
    formatter_test.tearDown()
    
    # Test query processor
    query_test = TestCacheMetadata('test_query_processor_embedding_integration')
    query_test.setUp()
    query_test.test_query_processor_embedding_integration()
    query_test.tearDown()
    
    # Test ChromaManager
    chroma_test = TestCacheMetadata('test_chroma_manager_metadata_handling')
    chroma_test.setUp()
    chroma_test.test_chroma_manager_metadata_handling()
    chroma_test.tearDown()
    
    # Test Claude Code tool
    claude_test = TestCacheMetadata('test_claude_code_tool_input_parsing')
    claude_test.setUp()
    claude_test.test_claude_code_tool_input_parsing()
    claude_test.tearDown()
    
    logger.info("================================")
    logger.info("Cache and Metadata Test Suite Complete")
    logger.info("================================")

if __name__ == "__main__":
    # Run the detailed test suite
    run_tests_with_detailed_output()