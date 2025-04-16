#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from typing import Dict, Any, List

def test_tool_direct(queries: List[str]):
    """Test the Claude Code tool directly without going through Claude."""
    
    # Get the path to the tool script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tool_path = os.path.join(script_dir, "..", "scripts", "claude-code-tool.py")
    
    print(f"Testing tool at: {tool_path}")
    
    if not os.path.exists(tool_path):
        print(f"Error: Tool script not found at {tool_path}")
        return
    
    # Make sure the script is executable
    try:
        os.chmod(tool_path, 0o755)
    except Exception as e:
        print(f"Warning: Could not make tool executable: {e}")
    
    # Run tests for each query
    for i, query in enumerate(queries):
        print(f"\n=== Test {i+1}/{len(queries)} ===")
        print(f"Query: '{query}'")
        
        # Create input data
        input_data = {
            "query": query,
            "num_results": 3  # Limit to 3 results for cleaner output
        }
        
        # Convert to JSON
        input_json = json.dumps(input_data)
        
        # Call the tool
        try:
            result = subprocess.run(
                [tool_path],
                input=input_json,
                text=True,
                capture_output=True
            )
            
            if result.returncode == 0:
                # Parse the output
                try:
                    # Try to extract JSON from the output - there might be debug lines before the JSON
                    json_start = result.stdout.find('{')
                    if json_start >= 0:
                        json_str = result.stdout[json_start:]
                        output = json.loads(json_str)
                        
                        # Print results summary
                        print(f"\nFound {len(output.get('results', []))} results:")
                        
                        # Print each result
                        for j, res in enumerate(output.get('results', [])[:3]):  # Show up to 3 results
                            print(f"\n--- Result {j+1} ({res['chunk_type']}) ---")
                            print(f"Title: {res['title']}")
                            print(f"Source: {res['source']}")
                            print(f"Score: {res['score']:.4f}")
                            print(f"Snippet: {res['snippet'][:150]}...")
                        
                        # Print any additional info
                        if "claude_context" in output:
                            print(f"\nDetected as: {output['claude_context'].get('query_description', 'unknown')} query")
                    else:
                        raise json.JSONDecodeError("No JSON found in output", result.stdout, 0)
                    
                except json.JSONDecodeError:
                    print("Error: Could not parse tool output as JSON")
                    print(f"Raw output: {result.stdout[:200]}...")
            else:
                print(f"Error: Tool returned non-zero exit code: {result.returncode}")
                print(f"Error output: {result.stderr}")
        
        except Exception as e:
            print(f"Error running tool: {e}")

if __name__ == "__main__":
    # List of test queries
    test_queries = [
        # Code queries
        "How to implement custom nn.Module in PyTorch",
        "Example of PyTorch DataLoader with custom Dataset",
        "Implementing batch normalization in PyTorch",
        
        # Concept queries
        "What is autograd in PyTorch?",
        "Explain backpropagation in PyTorch",
        "Difference between nn.Module and nn.functional",
        
        # Mixed query
        "How does PyTorch handle GPU memory management?"
    ]
    
    # Allow command line arguments to override the default queries
    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]
    
    test_tool_direct(test_queries)
