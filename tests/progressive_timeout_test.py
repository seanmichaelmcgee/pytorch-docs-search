#!/usr/bin/env python3

import os
import sys
import json
import tempfile
import subprocess
import logging
import time
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("progressive_timeout_test")

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

def simulate_claude_code_call_with_timeouts(query: str, timeouts: Dict[str, int]) -> Dict[str, Any]:
    """
    Simulate a call from Claude Code to our tool wrapper with custom timeouts.
    
    Args:
        query: The search query text
        timeouts: Dictionary of timeout values for each stage
        
    Returns:
        Dictionary containing the tool response
    """
    print(f"\nSimulating Claude Code call with query: '{query}' and timeouts: {timeouts}")
    
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
        "num_results": 3,
        "timeouts": timeouts
    }
    
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

def verify_progressive_timeout_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that the progressive timeout response is correctly formatted.
    
    Args:
        response: The response from the tool
        
    Returns:
        Dict with verification results
    """
    print("\nVerifying progressive timeout response...")
    
    verification = {
        "valid": True,
        "has_status": False,
        "is_partial": False,
        "stages_completed": [],
        "stages_timed_out": [],
        "notes": []
    }
    
    # Check for status field
    if "status" not in response:
        verification["notes"].append("Response does not contain status field")
        verification["valid"] = False
        return verification
    
    verification["has_status"] = True
    
    # Check for partial results
    if response["status"].get("is_partial", False):
        verification["is_partial"] = True
        verification["notes"].append(f"Received partial results: {response.get('error', 'No error message')}")
    
    # Check for stages completed/timed out
    verification["stages_completed"] = response["status"].get("stages_completed", [])
    verification["stages_timed_out"] = response["status"].get("stages_timed_out", [])
    
    # Check if we have results even with timeout
    if verification["is_partial"] and response.get("results", []):
        verification["notes"].append(f"Successfully received {len(response['results'])} partial results despite timeout")
    
    # All checks passed
    return verification

def test_progressive_timeout():
    """Test the progressive timeout feature with various timeout configurations."""
    print("=== Progressive Timeout Feature Test ===")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Normal operation (generous timeouts)",
            "query": "How to create a CNN in PyTorch",
            "timeouts": {
                "query_processing": 5,
                "database_search": 10,
                "result_formatting": 5
            },
            "expected": {
                "is_partial": False,
                "stages_completed": ["query_processing", "database_search", "result_formatting"]
            }
        },
        {
            "name": "Query processing timeout",
            "query": "Explain how transformers work in PyTorch with detailed examples",
            "timeouts": {
                "query_processing": 0.001,  # Very short timeout to trigger failure
                "database_search": 5,
                "result_formatting": 5
            },
            "expected": {
                "is_partial": True,
                "stages_timed_out": ["query_processing"]
            }
        },
        {
            "name": "Database search timeout",
            "query": "Show me how to implement attention mechanism in PyTorch",
            "timeouts": {
                "query_processing": 5,
                "database_search": 0.001,  # Very short timeout to trigger failure
                "result_formatting": 5
            },
            "expected": {
                "is_partial": True,
                "stages_completed": ["query_processing"],
                "stages_timed_out": ["database_search"]
            }
        },
        {
            "name": "Result formatting timeout",
            "query": "How to use distributed training in PyTorch",
            "timeouts": {
                "query_processing": 5,
                "database_search": 5,
                "result_formatting": 0.001  # Very short timeout to trigger failure
            },
            "expected": {
                "is_partial": True,
                "stages_completed": ["query_processing", "database_search"],
                "stages_timed_out": ["result_formatting"]
            }
        }
    ]
    
    # Track test results
    test_results = {}
    
    # Run tests
    for i, scenario in enumerate(test_scenarios):
        print(f"\n=== Test {i+1}/{len(test_scenarios)}: {scenario['name']} ===")
        
        # Simulate the call
        start_time = time.time()
        response = simulate_claude_code_call_with_timeouts(scenario["query"], scenario["timeouts"])
        elapsed_time = time.time() - start_time
        
        # Verify the response
        verification = verify_progressive_timeout_response(response)
        
        # Check against expected behavior
        expected = scenario["expected"]
        matched_expectations = True
        
        if expected.get("is_partial") != verification["is_partial"]:
            verification["notes"].append(f"Expected is_partial={expected.get('is_partial')}, got {verification['is_partial']}")
            matched_expectations = False
        
        if "stages_completed" in expected:
            for stage in expected["stages_completed"]:
                if stage not in verification["stages_completed"]:
                    verification["notes"].append(f"Expected completion of stage '{stage}' but it was not completed")
                    matched_expectations = False
        
        if "stages_timed_out" in expected:
            for stage in expected["stages_timed_out"]:
                if stage not in verification["stages_timed_out"]:
                    verification["notes"].append(f"Expected timeout of stage '{stage}' but it did not time out")
                    matched_expectations = False
        
        # Store the result
        test_results[scenario["name"]] = {
            "valid_format": verification["valid"],
            "matched_expectations": matched_expectations,
            "is_partial": verification["is_partial"],
            "stages_completed": verification["stages_completed"],
            "stages_timed_out": verification["stages_timed_out"],
            "result_count": len(response.get("results", [])),
            "elapsed_time": elapsed_time,
            "notes": verification["notes"]
        }
        
        # Print summary
        print(f"\nTest {i+1} result: {'PASS' if matched_expectations else 'FAIL'}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        if verification["is_partial"]:
            print(f"Received partial results with {len(response.get('results', []))} items")
        else:
            print(f"Received complete results with {len(response.get('results', []))} items")
        
        if verification["notes"]:
            print("Notes:")
            for note in verification["notes"]:
                print(f"- {note}")
    
    # Overall summary
    print("\n=== Progressive Timeout Feature Test Summary ===")
    passed = sum(1 for result in test_results.values() if result["matched_expectations"])
    print(f"Tests passed: {passed}/{len(test_scenarios)}")
    
    if passed == len(test_scenarios):
        print("✓ All tests passed - Progressive timeout feature appears to be working correctly")
    else:
        print("✗ Some tests failed - check the detailed results for more information")
        
        # Show failed tests
        print("\nFailed tests:")
        for name, result in test_results.items():
            if not result["matched_expectations"]:
                print(f"- '{name}' - Notes: {', '.join(result['notes'])}")
    
    # Save detailed results to file
    with open("progressive_timeout_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nDetailed results saved to progressive_timeout_test_results.json")

if __name__ == "__main__":
    test_progressive_timeout()