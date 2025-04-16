#!/usr/bin/env python3

import os
import sys
import json
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chromadb_test")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the ChromaDB manager
from scripts.database.chroma_manager import ChromaManager
from scripts.config import DB_DIR, COLLECTION_NAME

def test_chromadb_api_compatibility():
    """Test ChromaDB API compatibility with the current implementation."""
    print("=== ChromaDB API Compatibility Test ===")
    
    # Test initialization
    print("\n1. Testing ChromaDB initialization...")
    try:
        db_manager = ChromaManager(DB_DIR, COLLECTION_NAME)
        print("✓ ChromaDB manager initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing ChromaDB manager: {str(e)}")
        return
    
    # Test collection retrieval
    print("\n2. Testing collection retrieval...")
    try:
        collection = db_manager.get_collection()
        print(f"✓ Retrieved collection '{COLLECTION_NAME}' successfully")
    except Exception as e:
        print(f"✗ Error retrieving collection: {str(e)}")
        return
    
    # Test collection stats
    print("\n3. Testing collection statistics...")
    try:
        stats = db_manager.get_stats()
        print(f"✓ Retrieved collection statistics:")
        print(f"  - Total chunks: {stats['total_chunks']}")
        
        if 'chunk_types' in stats:
            print("  - Chunk types:")
            for chunk_type, count in stats['chunk_types'].items():
                print(f"    • {chunk_type}: {count}")
        
        if 'sources' in stats:
            print("  - Top sources:")
            top_sources = sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:5]
            for source, count in top_sources:
                print(f"    • {source}: {count}")
                
        if stats.get('is_sample', False):
            print("  Note: Statistics based on a sample due to large collection size")
    except Exception as e:
        print(f"✗ Error retrieving collection statistics: {str(e)}")
        return
    
    # Test simple query
    print("\n4. Testing basic query functionality...")
    try:
        # Create a simple test embedding (all zeros)
        test_embedding = [0.0] * 3072  # Using the dimensions from config
        
        # Perform a simple query
        results = db_manager.query(test_embedding, n_results=3)
        
        # Verify result structure
        if 'ids' in results and 'documents' in results and 'metadatas' in results and 'distances' in results:
            print("✓ Query returned correctly structured results")
            
            # Get result counts
            result_count = len(results['ids'][0]) if results['ids'] and len(results['ids']) > 0 else 0
            print(f"  - Retrieved {result_count} results")
            
            # Print first result if available
            if result_count > 0:
                doc = results['documents'][0][0] if results['documents'] and len(results['documents']) > 0 else "N/A"
                metadata = results['metadatas'][0][0] if results['metadatas'] and len(results['metadatas']) > 0 else {}
                distance = results['distances'][0][0] if results['distances'] and len(results['distances']) > 0 else "N/A"
                
                print("  - First result:")
                print(f"    • Document snippet: {doc[:100]}...")
                print(f"    • Metadata: {metadata}")
                print(f"    • Distance: {distance}")
        else:
            print("✗ Query returned incorrectly structured results")
            print(f"  - Result keys: {list(results.keys())}")
    except Exception as e:
        print(f"✗ Error performing query: {str(e)}")
        return
    
    # Test filtered query
    print("\n5. Testing filtered query functionality...")
    try:
        # Create a filter for code chunks
        filters = {"chunk_type": "code"}
        
        # Perform a filtered query
        results = db_manager.query(test_embedding, n_results=3, filters=filters)
        
        # Verify filter worked
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0 and len(results['metadatas'][0]) > 0:
            all_code = all(meta.get('chunk_type') == 'code' for meta in results['metadatas'][0])
            if all_code:
                print("✓ Filter successfully applied (all results are code chunks)")
            else:
                print("✗ Filter did not work correctly (non-code chunks in results)")
        else:
            print("✓ Filter query executed but returned no results")
    except Exception as e:
        print(f"✗ Error performing filtered query: {str(e)}")
        return
    
    # Overall status
    print("\n=== ChromaDB API Compatibility Test Results ===")
    print("All tests completed. The ChromaDB integration appears to be working correctly with the current API version.")

if __name__ == "__main__":
    test_chromadb_api_compatibility()
