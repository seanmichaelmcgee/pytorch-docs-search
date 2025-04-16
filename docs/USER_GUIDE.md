# PyTorch Documentation Search Tool User Guide

This guide explains how to effectively use the PyTorch Documentation Search tool with Claude Code CLI for finding relevant PyTorch documentation and code examples.

## Overview

The PyTorch Documentation Search Tool provides semantic search capabilities for PyTorch documentation using vector embeddings. It understands both natural language and code contexts, making it especially useful for finding API usage examples, implementation details, and code snippets.

## Getting Started

### Prerequisites

- Claude Code CLI installed
- PyTorch Documentation Search Tool registered with Claude
- PyTorch documentation indexed in the database

### Basic Usage

The tool can be used in two ways:

1. **Directly through Claude Code CLI**:
   ```
   Ask Claude: "Find me examples of implementing custom loss functions in PyTorch"
   ```

2. **Using the command line tool**:
   ```bash
   echo '{"query": "How to implement a custom nn.Module in PyTorch"}' | ./scripts/claude-code-tool.py
   ```

## Asking Effective Questions

The tool performs best when you ask specific questions. Here are some tips for getting the best results:

### Good Examples

- "How do I implement a custom DataLoader in PyTorch?"
- "Show me examples of using nn.Conv2d with stride and padding"
- "What's the difference between torch.no_grad() and torch.set_grad_enabled(False)?"
- "How to save and load model checkpoints in PyTorch?"
- "Explain PyTorch's autograd mechanism"

### Less Effective Examples

- "PyTorch" (too vague)
- "How do I train a model?" (too general)
- "Code for neural networks" (too broad)

## Understanding Results

Search results include:

- **Title**: The title of the document section
- **Snippet**: A relevant excerpt from the documentation
- **Source**: The source file or URL
- **Type**: Whether the result is code or explanatory text
- **Score**: A relevance score (higher is better)

The tool automatically balances code examples and textual explanations based on your query.

## Advanced Usage

### Filtering Results

You can filter results to show only code examples or only textual explanations:

```
Ask Claude: "Find examples of implementing custom loss functions in PyTorch, but only show me code examples"
```

This will set the `filter` parameter to "code".

### Adjusting Result Count

By default, the tool returns 3-5 results. You can request more or fewer:

```
Ask Claude: "Show me just the top 2 most relevant examples of PyTorch DataLoader usage"
```

### Combining with Claude's Capabilities

The tool works best when combined with Claude's capabilities:

- Ask Claude to explain the code examples returned by the search
- Request summaries of the documentation
- Ask for comparisons between different approaches found in the results

## Troubleshooting

### No Results Found

If no results are found:

1. Try rephrasing your query to be more specific
2. Use PyTorch terminology (e.g., "tensor" instead of "array")
3. Check if your query is for a very recent feature not yet in the documentation

### Timeout Errors

For complex queries, the tool may time out. Try:

1. Simplifying your query
2. Breaking it down into smaller, more specific questions
3. Using fewer special characters or code snippets in the query

### Inaccurate Results

If results don't seem relevant:

1. Be more specific in your query
2. Include key PyTorch terms or class names
3. Specify the context (e.g., "in the context of computer vision")

## Examples in Context

### Example 1: Finding API Documentation

**Query**: "How to use torch.nn.Transformer?"

**Expected Results**:
- Documentation for the Transformer class
- Code examples showing parameter configuration
- Usage examples in sequence modeling

### Example 2: Debugging Common Errors

**Query**: "CUDA out of memory error in PyTorch"

**Expected Results**:
- Documentation about GPU memory management
- Techniques for reducing memory usage
- Code examples showing how to move tensors between devices

## Additional Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch GitHub Repository](https://github.com/pytorch/pytorch)
- [PyTorch Forums](https://discuss.pytorch.org/)