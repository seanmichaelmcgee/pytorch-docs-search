#!/bin/bash

# Get the absolute path to the tool script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TOOL_PATH="$SCRIPT_DIR/claude-code-tool.py"

# Validate that the tool path exists
if [ ! -f "$TOOL_PATH" ]; then
  echo "Error: Tool script not found at $TOOL_PATH"
  exit 1
fi

# Check if the tool is executable
if [ ! -x "$TOOL_PATH" ]; then
  echo "Making tool executable..."
  chmod +x "$TOOL_PATH"
  
  # Verify the chmod was successful
  if [ ! -x "$TOOL_PATH" ]; then
    echo "Error: Failed to make tool executable. Check file permissions."
    exit 1
  fi
fi

echo "Using tool script: $TOOL_PATH"

# Define the tool description
TOOL_NAME="pytorch_search"
TOOL_DESC="Search PyTorch documentation and find examples using advanced code-aware semantic search. This tool understands both natural language and code patterns, making it especially effective for finding API usage examples, implementation details, and code snippets. Results include both explanatory text and code examples with appropriate context."

# JSON schema for tool input
INPUT_SCHEMA='{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The search query about PyTorch - can include natural language or Python code patterns"
    },
    "num_results": {
      "type": "integer",
      "description": "Number of results to return (default: 5)",
      "default": 5
    },
    "filter": {
      "type": "string",
      "enum": ["code", "text", null],
      "description": "Optional filter to return only code examples or only text explanations",
      "default": null
    }
  },
  "required": ["query"]
}'

# Register the tool with Claude Code CLI
echo "Registering PyTorch Documentation Search Tool with Claude Code..."

# Check if claude mcp command exists
if command -v claude mcp &> /dev/null; then
    # Register the tool
    claude mcp add "$TOOL_NAME" --description "$TOOL_DESC" --command "$TOOL_PATH" --input-schema "$INPUT_SCHEMA"
    
    echo "Tool registration successful!"
    echo "You can now use the pytorch_search tool with Claude Code."
    echo "Example: Ask Claude 'Find me examples of implementing custom loss functions in PyTorch'"
else
    echo "Error: Claude Code CLI not found. Please install it first."
    echo "Visit https://docs.anthropic.com/claude/docs/claude-code-cli for installation instructions."
    exit 1
fi

# Verify registration
echo ""
echo "Verifying tool registration..."
if claude mcp list | grep -q "$TOOL_NAME"; then
    echo "✓ Tool '$TOOL_NAME' is registered successfully."
else
    echo "✗ Tool verification failed. Please check the registration manually."
fi

# Instructions for testing
echo ""
echo "To test the tool, you can use:"
echo "echo '{\"query\": \"How to implement a custom nn.Module in PyTorch\"}' | $TOOL_PATH"
echo ""
echo "For more comprehensive testing, use the test scripts:"
echo "python tests/test-tool-direct.py"
echo "python tests/claude-code-tool-test.py"
