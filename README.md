# PyTorch Documentation Search Tool

A specialized semantic search system for PyTorch documentation that understands both code and text, providing relevant results for technical queries.

## 📋 Overview

This tool enables developers to efficiently search PyTorch documentation using natural language or code-based queries. It preserves the integrity of code examples, understands programming patterns, and intelligently ranks results based on query intent.

**Key Features:**
- Code-aware document processing that preserves code block integrity
- Semantic search powered by OpenAI embeddings
- Query intent detection (code vs concept)
- Local operation with minimal dependencies
- Claude Code CLI integration
- Incremental document updates

## 🚀 Getting Started

### Prerequisites

- Ubuntu 24.04 LTS (or similar)
- Python 3.10+
- OpenAI API key
- 8GB RAM recommended

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pytorch-docs-search.git
   cd pytorch-docs-search
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file to add your OpenAI API key
   ```

## 🛠️ Usage

### Initial Setup

1. Index your PyTorch documentation:
   ```bash
   python scripts/index_documents.py --docs-dir /path/to/pytorch/docs
   ```

2. Generate embeddings:
   ```bash
   python scripts/generate_embeddings.py
   ```

3. Load embeddings to the database:
   ```bash
   python scripts/load_to_database.py
   ```

### Searching Documentation

#### Command Line Interface

Run the search tool directly:

```bash
python scripts/document_search.py "How to implement a custom autograd function"
```

Or use interactive mode:

```bash
python scripts/document_search.py --interactive
```

#### Filter Results

Limit results to specific content types:

```bash
python scripts/document_search.py "custom loss function" --filter code
```

### Updating Documentation

When PyTorch documentation is updated, you can incrementally process and add new or changed documents:

1. Process only new/changed files:
   ```bash
   python scripts/index_documents.py --docs-dir /path/to/updated/docs --output-file ./data/updated_chunks.json
   ```

2. Generate embeddings for these chunks:
   ```bash
   python scripts/generate_embeddings.py --input-file ./data/updated_chunks.json --output-file ./data/updated_chunks_with_embeddings.json
   ```

3. Add to the database without resetting:
   ```bash
   python scripts/load_to_database.py --input-file ./data/updated_chunks_with_embeddings.json --no-reset
   ```

The embedding cache ensures efficient processing by avoiding regenerating embeddings for unchanged content.

### Claude Code Integration

To integrate with Claude Code CLI:

1. Register the search tool:
   ```bash
   python scripts/register_tool.sh
   ```

2. The tool will be available to Claude through the MCP protocol

## 📦 Project Structure

```
pytorch-docs-search/
├── data/               # Storage for processed docs and database
│   ├── chroma_db/      # Vector database
│   ├── embedding_cache/ # Cached embeddings
│   └── indexed_chunks.json
├── scripts/            # Core scripts
│   ├── config/         # Configuration module
│   ├── database/       # ChromaDB integration
│   ├── document_processing/ # Document parsing and chunking
│   ├── embedding/      # Embedding generation
│   ├── search/         # Search interface
│   ├── document_search.py  # Main search script
│   ├── generate_embeddings.py # Embedding generation script
│   ├── index_documents.py # Document processing script
│   ├── load_to_database.py # Database loading script
│   ├── migrate_embeddings.py # Model migration script
│   └── register_tool.sh # Claude Code integration
├── docs/               # Documentation
├── tests/              # Unit tests
├── .env                # Environment variables
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## 🔧 Configuration

Edit `.env` file to configure:

- `OPENAI_API_KEY`: Your OpenAI API key
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `OVERLAP_SIZE`: Overlap between chunks (default: 200)
- `MAX_RESULTS`: Default number of search results (default: 5)
- `DB_DIR`: ChromaDB storage location (default: ./data/chroma_db)
- `COLLECTION_NAME`: Name of the ChromaDB collection (default: pytorch_docs)

Advanced settings can be modified in `scripts/config/__init__.py`.

## 🧠 How It Works

### Document Processing Pipeline

1. **Parsing**: Uses Tree-sitter to parse markdown and Python files, preserving structure.
2. **Chunking**: Intelligently divides documents into chunks, respecting code boundaries:
   - Keeps code blocks intact where possible
   - Uses semantic boundaries (functions, classes) for large code blocks
   - Uses paragraphs and sentences for text
3. **Metadata**: Enriches chunks with source, title, content type, and language information

### Search Process

1. **Query Analysis**: Determines if the query is code-focused or concept-focused
2. **Embedding Generation**: Creates a vector representation of the query
3. **Vector Search**: Finds semantically similar chunks in the database
4. **Result Ranking**: Boosts code examples for code queries, explanations for concept queries
5. **Formatting**: Returns structured results with relevant snippets and metadata

## 🚧 Maintenance

### Monitoring

Check system status:

```bash
python scripts/monitor_system.py
```

### Backups

Create a database backup:

```bash
scripts/maintenance.sh
```

### Upgrading Embedding Models

If you need to migrate to a newer embedding model:

```bash
python scripts/migrate_embeddings.py --input-file ./data/indexed_chunks.json --output-file ./data/migrated_chunks.json
python scripts/load_to_database.py --input-file ./data/migrated_chunks.json
```

## 📊 Benchmarking

To evaluate embedding performance:

```bash
python scripts/benchmark_embeddings.py
```

## 🔍 Troubleshooting

### API Key Issues

If you encounter API key errors:
- Check that your `.env` file contains a valid `OPENAI_API_KEY`
- Verify the key has access to the embedding models

### Memory Issues

If you encounter out-of-memory errors:
- Reduce batch sizes in `scripts/load_to_database.py` (--batch-size 50)
- Process documents in smaller batches
- Increase system swap space if necessary

### Database Issues

If ChromaDB fails to load or query:
- Check database directory permissions
- Verify ChromaDB installation
- Try resetting the collection: `python scripts/load_to_database.py --input-file ./data/indexed_chunks.json`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenAI API](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB](https://docs.trychroma.com/)
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [Claude Code CLI](https://www.anthropic.com/claude)
