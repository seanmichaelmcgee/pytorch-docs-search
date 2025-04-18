# PyTorch Documentation Search - MCP Integration with Claude Code CLI

This guide explains how to set up and use the MCP integration for the PyTorch Documentation Search tool with Claude Code CLI.

## Overview

The PyTorch Documentation Search tool is now integrated with Claude Code CLI through the Model-Context Protocol (MCP), allowing Claude to directly access our semantic search capabilities.

Key features of this integration:
- Progressive search with fallback behavior
- MCP-compliant API endpoint
- Detailed timing and diagnostics
- Compatibility with both code and concept queries
- Structured JSON responses

## Setup Instructions

### 1. Activate the Conda Environment

```bash
# Activate the environment
conda activate pytorch_docs_search

# OR with Mamba
mamba activate pytorch_docs_search
```

### 2. Start the Flask Server

```bash
# Navigate to the project root
cd /path/to/pytorch-docs-search

# Run the Flask server
python app.py
```

You should see the following output:
```
=== PyTorch Documentation Search API ===
Server running at: http://localhost:5000/search
Register with Claude Code CLI using:
claude mcp add mcp__pytorch_docs__semantic_search http://localhost:5000/search --transport sse

Press Ctrl+C to stop the server
```

### 3. Register the Tool with Claude Code CLI

In a new terminal window, run:

```bash
claude mcp add mcp__pytorch_docs__semantic_search "http://localhost:5000/search" \
  --description "Search PyTorch documentation and examples using code-aware semantic search" \
  --transport sse \
  --input-schema '{
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query about PyTorch"
      },
      "num_results": {
        "type": "integer",
        "description": "Number of results to return (default: 5)"
      },
      "filter": {
        "type": "string",
        "enum": ["code", "text", null],
        "description": "Filter results by type"
      }
    },
    "required": ["query"]
  }'
```

### 4. Verify Registration

Check that the tool is registered correctly:

```bash
claude mcp list
```

You should see `mcp__pytorch_docs__semantic_search` in the list of available tools.

## Usage

### Testing with CLI

To test the tool directly from the command line:

```bash
claude run tool mcp__pytorch_docs__semantic_search --input '{"query": "freeze layers in PyTorch"}'
```

For filtering results:

```bash
claude run tool mcp__pytorch_docs__semantic_search --input '{"query": "batch normalization", "filter": "code"}'
```

To retrieve more results:

```bash
claude run tool mcp__pytorch_docs__semantic_search --input '{"query": "autograd example", "num_results": 10}'
```

### Using with Claude CLI

When using Claude CLI, you can integrate the tool into your conversations:

```bash
claude run
```

Then within your conversation with Claude, you can ask about PyTorch topics and Claude will automatically use the tool to search the documentation.

## Monitoring and Logging

All API requests and responses are logged to `flask_api.log` in the project root directory. This file contains detailed information about:

- Request timestamps and content
- Query processing stages
- Search timing information
- Any errors encountered
- Result counts and metadata

To monitor the log in real-time:

```bash
tail -f flask_api.log
```

## Troubleshooting

### Common Issues

1. **Tool Registration Fails**
   - Ensure the Flask server is running
   - Check that you have the correct URL (http://localhost:5000/search)
   - Verify you have the latest Claude CLI installed

2. **Server Won't Start**
   - Verify the port 5000 is available
   - Ensure all dependencies are installed in your environment
   - Check for any import errors in the console output

3. **No Results Returned**
   - Verify that the ChromaDB database has been populated
   - Check that the OpenAI API key is set correctly in your environment
   - Look for error messages in the flask_api.log

4. **Partial Results**
   - Check the `is_partial` flag in the response
   - Look for `stages_timed_out` to identify which stage failed
   - The system will return as much information as available even if some stages fail

### Getting Help

If you encounter issues not covered here, check:
1. The main Flask API log: `flask_api.log`
2. The Python error output in the terminal running the Flask server
3. The Claude CLI error messages when attempting to use the tool

## Architecture

The MCP integration follows a three-stage pipeline:

1. **Query Processing**: Analyzes the query and generates embeddings
2. **Database Search**: Searches ChromaDB for relevant matches 
3. **Result Formatting**: Structures and ranks results based on query intent

Each stage is designed to fail gracefully, providing as much information as possible even if later stages encounter errors.

## Security Notes

- The server binds to all interfaces (0.0.0.0) by default; in production, consider restricting this
- The API doesn't implement authentication; if exposed publicly, add API key validation
- OpenAI API keys are loaded from environment variables; ensure they're properly secured

## Next Steps

- Add authentication to the API endpoint
- Implement caching for frequent queries
- Add support for more filter types
- Create a dashboard for monitoring API usage and performance