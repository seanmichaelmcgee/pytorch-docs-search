# OpenAI API Migration Guide: Updating to v1.0+ Format

## Overview

Your current codebase uses the OpenAI Python library's older interface (pre-1.0), but you have v1.0+ installed. This guide will help you methodically update your code to use the newer API format without needing to downgrade.

## Core API Changes

The OpenAI Python SDK v1.0+ introduces three key changes:

1. **Client-based approach**: Initialize a client object instead of using module-level functions
2. **Method naming changes**: Use `client.embeddings.create()` instead of `openai.Embedding.create()`
3. **Response structure**: Response objects now have attributes instead of dictionary keys

## Migration Process

Follow these steps in sequence to ensure a smooth transition:

### Step 1: Update the Embedding Generator

First, modify `scripts/embedding/generator.py`:

```python
# Change from:
import openai
# ...
openai.api_key = OPENAI_API_KEY

# To:
from openai import OpenAI
# ...
# Remove the global API key assignment, as it will be handled in the class
```

Then update the `EmbeddingGenerator` class:

```python
class EmbeddingGenerator:
    def __init__(self, model: str = EMBEDDING_MODEL, use_cache: bool = True):
        """Initialize the embedding generator."""
        self.model = model
        
        # Add OpenAI client initialization
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Cache initialization remains unchanged
        if use_cache:
            from scripts.config import EMBEDDING_CACHE_DIR, EMBEDDING_CACHE_MAX_SIZE_GB
            self.cache = EmbeddingCache(
                EMBEDDING_CACHE_DIR, 
                max_size_gb=EMBEDDING_CACHE_MAX_SIZE_GB
            )
            logger.info(f"Embedding cache initialized at {EMBEDDING_CACHE_DIR}")
        else:
            self.cache = None
            logger.info("Embedding cache disabled")
```

Update the embedding generation logic:

```python
# In _get_embedding_with_cache or similar method:

# Change from:
response = openai.Embedding.create(
    input=text,
    model=self.model
)
embedding = response["data"][0]["embedding"]

# To:
response = self.client.embeddings.create(
    input=text,
    model=self.model
)
embedding = response.data[0].embedding
```

Similarly, in the batch embedding method:

```python
# Change from:
response = openai.Embedding.create(
    input=uncached_texts,
    model=self.model
)
api_embeddings = [item["embedding"] for item in response["data"]]

# To:
response = self.client.embeddings.create(
    input=uncached_texts,
    model=self.model
)
api_embeddings = [item.embedding for item in response.data]
```

### Step 2: Update the Query Processor

Modify `scripts/search/query_processor.py`:

```python
# Change from:
import openai
# ...
openai.api_key = OPENAI_API_KEY

# To:
from openai import OpenAI
```

Update the `QueryProcessor` class:

```python
class QueryProcessor:
    def __init__(self, model: str = EMBEDDING_MODEL, use_cache: bool = True):
        """Initialize the query processor."""
        self.model = model
        
        # Add OpenAI client initialization
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Existing cache initialization remains unchanged
        if use_cache:
            from scripts.config import EMBEDDING_CACHE_DIR, EMBEDDING_CACHE_MAX_SIZE_GB
            self.cache = EmbeddingCache(
                EMBEDDING_CACHE_DIR + '/queries',
                max_size_gb=EMBEDDING_CACHE_MAX_SIZE_GB / 10  # Use smaller cache for queries
            )
            logger.info(f"Query embedding cache initialized")
        else:
            self.cache = None
```

Update the embedding generation logic:

```python
# In generate_embedding method:

# Change from:
response = openai.Embedding.create(
    input=query,
    model=self.model
)
embedding = response["data"][0]["embedding"]

# To:
response = self.client.embeddings.create(
    input=query,
    model=self.model
)
embedding = response.data[0].embedding
```

### Step 3: Update Any Additional Files

If there are any other modules using OpenAI embeddings (like benchmark scripts), update them following the same pattern.

### Step 4: Test Your Changes

1. Run the embedding generation script:
```bash
python -m scripts.generate_embeddings --input-file ./data/indexed_chunks.json
```

2. Test a search query:
```bash
python -m scripts.document_search "how to implement custom autograd function"
```

## Common Issues and Troubleshooting

### Response Structure Mismatch

**Problem**: Error accessing embedding data from response
**Solution**: The response structure has changed. Make sure to use attribute access (`.`) instead of dictionary access (`[]`):

```python
# Old: embedding = response["data"][0]["embedding"]
# New: embedding = response.data[0].embedding
```

### API Error Handling

**Problem**: Error handling code no longer catches exceptions properly
**Solution**: The exception classes have changed. Update your try/except blocks:

```python
# Old:
try:
    response = openai.Embedding.create(...)
except openai.error.RateLimitError:
    # handle rate limit

# New:
from openai import RateLimitError
try:
    response = client.embeddings.create(...)
except RateLimitError:
    # handle rate limit
```

### Multiple Parameter Changes

**Problem**: API calls failing with parameter errors
**Solution**: Some parameter names might have changed. Check the [OpenAI API reference](https://platform.openai.com/docs/api-reference/embeddings) for current parameters.

## Final Steps

After making these changes:
1. Ensure all tests pass
2. Run a full embedding generation and loading cycle
3. Verify search results match expected quality

By following this structured approach, you should be able to successfully migrate your codebase to use the OpenAI API v1.0+ format while maintaining all functionality.
