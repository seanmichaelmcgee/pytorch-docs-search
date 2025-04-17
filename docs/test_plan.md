# Conda Environment Migration - Test Plan

This document outlines the testing strategy for validating the Conda environment migration.

## Test Environment Setup

1. Create a fresh Conda environment using the updated configuration:
   ```bash
   # Remove existing environment if present
   conda env remove -n pytorch_docs_search

   # Create new environment
   ./setup_conda_env.sh
   # OR
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate pytorch_docs_search
   ```

3. Validate environment setup:
   ```bash
   python test_conda_env.py
   ```

## Component Tests

Run individual component tests to validate core functionality:

1. Test ChromaDB Integration:
   ```bash
   pytest -v tests/chromadb-api-test.py
   ```

2. Test Embedding Cache:
   ```bash
   pytest -v tests/embedding-cache-test.py
   ```

3. Test Metadata Cache:
   ```bash
   pytest -v tests/test_cache_metadata.py
   ```

4. Test Chunking Algorithm:
   ```bash
   pytest -v tests/test_robust_chunking.py
   ```

5. Test Progressive Timeout:
   ```bash
   pytest -v tests/progressive_timeout_test.py
   ```

## Integration Tests

Run end-to-end tests to validate system functionality:

1. Test Search Functionality:
   ```bash
   pytest -v tests/test-e2e-search.py
   ```

2. Test Claude Code Tool:
   ```bash
   pytest -v tests/claude-code-tool-test.py
   ```

3. Test Tool API Interface:
   ```bash
   pytest -v tests/test-tool-direct.py
   ```

## Full Test Suite

Run the complete test suite:

```bash
pytest -v tests/
```

## Regression Testing

Verify that key operations still work correctly:

1. Test Document Processing:
   ```bash
   python scripts/index_documents.py --docs-dir tests/data --output-file tests/data/test_chunks.json
   ```

2. Test Embedding Generation:
   ```bash
   python scripts/generate_embeddings.py --input-file tests/data/test_chunks.json --output-file tests/data/test_chunks_with_embeddings.json
   ```

3. Test Database Loading:
   ```bash
   python scripts/load_to_database.py --input-file tests/data/test_chunks_with_embeddings.json --db-dir tests/data/test_db
   ```

4. Test Search:
   ```bash
   python scripts/document_search.py "PyTorch tensor operations" --filter text
   ```

## Validation Criteria

The migration will be considered successful if:

1. All component tests pass
2. All integration tests pass
3. Regression tests show expected behavior
4. No references to the old virtual environment remain
5. No hard-coded paths need updating

## Issue Tracking

Document any issues encountered during testing in this format:

### Issue: [Brief Description]
- **Component**: [Affected component]
- **Test**: [Test that revealed the issue]
- **Description**: [Detailed description]
- **Resolution**: [How the issue was fixed]