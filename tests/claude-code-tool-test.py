#!/usr/bin/env python3

import os
import sys
import json
import tempfile
import subprocess
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tool_test")

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

def simulate_claude_code_call(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Simulate a call from Claude Code to our tool wrapper.
    
    Args:
        query: The search query text
        num_results: Number of results to return
        
    Returns:
        Dictionary containing the tool response
    """
    print(f"\nSimulating Claude Code call with query: '{query}'")
    
    # Find the tool wrapper script
    tool_script = os.path.join(script_dir, "..", "scripts", "claude-code-tool.py")
    
    # Check if the script exists
    if not os.path.exists(tool_script):
        print(f"✗ Tool script not found at: {tool_script}")
        return {"error": "Tool script not found"}
    
    # Make it executable
    try:
        os.chmod(tool_script, 0o755)
    except Exception as e:
        print(f"Warning: Could not make tool script executable: {str(e)}")
    
    # Create input data in MCP format
    input_data = {
        "query": query,
        "num_results": num_results
    }
    
    # Create a temporary file for the input
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_in:
        json.dump(input_data, temp_in)
        temp_in_path = temp_in.name
    
    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_out:
        temp_out_path = temp_out.name
    
    try:
        # Run the tool script with the input
        print(f"Executing tool: {tool_script}")
        result = subprocess.run(
            [tool_script],
            input=json.dumps(input_data),
            text=True,
            capture_output=True
        )
        
        # Check if the script executed successfully
        if result.returncode == 0:
            print(f"✓ Tool executed successfully")
            
            # Try to parse the output - extract JSON from potentially mixed output
            try:
                # Find the start of the JSON object
                json_start = result.stdout.find('{')
                if json_start >= 0:
                    json_str = result.stdout[json_start:]
                    output_data = json.loads(json_str)
                    return output_data
                else:
                    raise json.JSONDecodeError("No JSON found in output", result.stdout, 0)
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing tool output: {str(e)}")
                print(f"Raw output: {result.stdout[:500]}")
                return {"error": f"Error parsing tool output: {str(e)}"}
        else:
            print(f"✗ Tool execution failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return {"error": f"Tool execution failed: {result.stderr}"}
        
    except Exception as e:
        print(f"✗ Error running tool: {str(e)}")
        return {"error": f"Error running tool: {str(e)}"}
    finally:
        # Clean up temporary files
        for path in [temp_in_path, temp_out_path]:
            try:
                os.unlink(path)
            except:
                pass

def verify_tool_response(response: Dict[str, Any]) -> bool:
    """
    Verify that the tool response is correctly formatted.
    
    Args:
        response: The response from the tool
        
    Returns:
        True if the response is valid, False otherwise
    """
    print("\nVerifying tool response format...")
    
    # Check for error
    if "error" in response:
        print(f"✗ Response contains an error: {response['error']}")
        return False
    
    # Check required fields
    required_fields = ["results", "query", "count"]
    missing_fields = [field for field in required_fields if field not in response]
    
    if missing_fields:
        print(f"✗ Response is missing required fields: {', '.join(missing_fields)}")
        return False
    
    print("✓ Response contains all required fields")
    
    # Check results structure
    if not isinstance(response["results"], list):
        print("✗ 'results' field is not a list")
        return False
    
    if not response["results"]:
        print("✓ Results list is empty (this might be expected for some queries)")
        return True
    
    # Check result item structure
    result_item = response["results"][0]
    required_result_fields = ["title", "snippet", "source", "chunk_type", "score"]
    missing_result_fields = [field for field in required_result_fields if field not in result_item]
    
    if missing_result_fields:
        print(f"✗ Result item is missing required fields: {', '.join(missing_result_fields)}")
        return False
    
    print("✓ Result items have the correct structure")
    
    # Check score is a float between 0 and 1
    if not isinstance(result_item["score"], float) or result_item["score"] < 0 or result_item["score"] > 1:
        print(f"✗ Score is not a float between 0 and 1: {result_item['score']}")
        return False
    
    print("✓ Score values are valid")
    
    # All checks passed
    return True

def test_claude_code_integration():
    """Test the Claude Code integration by simulating tool calls."""
    print("=== Claude Code Tool Integration Test ===")
    
    # Define test queries
    test_queries = [
        # Code query
        "How to implement a PyTorch Dataset class",
        
        # Concept query
        "What is backpropagation in PyTorch?",
        
        # Edge case: very short query
        "tensor",
        
        # Edge case: very long query
        "I need to understand how to implement a custom autograd function in PyTorch that can handle both forward and backward passes efficiently while preserving gradient information for a complex transformation involving multiple tensor operations including matrix multiplication, convolution, and non-linear activations"
    ]
    
    # Track test results
    test_results = {}
    
    # Run tests
    for i, query in enumerate(test_queries):
        print(f"\n=== Test {i+1}/{len(test_queries)} ===")
        
        # Simulate the call
        response = simulate_claude_code_call(query)
        
        # Verify the response
        is_valid = verify_tool_response(response)
        
        # Store the result
        test_results[query] = {
            "valid_format": is_valid,
            "result_count": len(response.get("results", [])),
            "error": response.get("error")
        }
        
        # Print summary
        print(f"\nTest {i+1} result: {'PASS' if is_valid else 'FAIL'}")
        if is_valid:
            print(f"Found {len(response.get('results', []))} results for query")
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Overall summary
    print("\n=== Claude Code Tool Integration Test Summary ===")
    passed = sum(1 for result in test_results.values() if result["valid_format"])
    print(f"Tests passed: {passed}/{len(test_queries)}")
    
    if passed == len(test_queries):
        print("✓ All tests passed - Claude Code integration appears to be working correctly")
    else:
        print("✗ Some tests failed - check the detailed results for more information")
        
        # Show failed tests
        print("\nFailed tests:")
        for query, result in test_results.items():
            if not result["valid_format"]:
                print(f"- '{query[:50]}...' - Error: {result.get('error', 'Unknown error')}")
    
    # Save detailed results to file
    with open("claude_code_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nDetailed results saved to claude_code_test_results.json")

if __name__ == "__main__":
    test_claude_code_integration()
