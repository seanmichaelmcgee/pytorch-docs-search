# PyTorch Documentation Search Tool - Implementation Plan

This document provides a comprehensive, step-by-step implementation plan for creating a specialized documentation search tool for PyTorch. The system uses structure-aware document processing, semantic embeddings, and vector search to provide high-quality results, with a particular focus on preserving the integrity of code examples.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Document Processing Pipeline](#document-processing-pipeline)
3. [Embedding Generation](#embedding-generation)
4. [Vector Database Integration](#vector-database-integration)
5. [Search Interface Implementation](#search-interface-implementation)
6. [Claude Code Integration](#claude-code-integration)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [System Monitoring and Maintenance](#system-monitoring-and-maintenance)
9. [Appendix: Troubleshooting Guide](#appendix-troubleshooting-guide)

## Environment Setup

### Prerequisites

Before beginning the implementation, ensure you have:

- Ubuntu 24.04 LTS installed
- Python 3.10+ available
- Administrative access for package installation
- OpenAI API key

### Step 1: System Package Dependencies

Ubuntu 24.04 requires some system packages to support our Python dependencies:

```bash
# Update package listings
sudo apt update

# Install required system packages
sudo apt install -y python3-pip python3-venv git build-essential cmake
sudo apt install -y libssl-dev zlib1g-dev libbz2-dev libreadline-dev
sudo apt install -y libsqlite3-dev libffi-dev liblzma-dev
```

**Success criteria**: All system packages install without errors.

**Potential pitfalls**: 
- Package repository issues - ensure your sources.list is properly configured
- Permission issues - make sure you have sudo access

### Step 2: Project Directory Structure

Create a well-organized directory structure:

```bash
# Create project directory
mkdir -p ~/pytorch-docs-search
cd ~/pytorch-docs-search

# Create subdirectories
mkdir -p data
mkdir -p pytorch_docs
mkdir -p scripts
mkdir -p logs
mkdir -p tests
```

**Success criteria**: Directory structure created correctly.

### Step 3: Python Virtual Environment

Set up an isolated Python environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Success criteria**: Virtual environment created and activated successfully.

**Potential pitfalls**:
- Python version issues - ensure you're using Python 3.10+
- Virtual environment activation problems - check for error messages

### Step 4: Install Dependencies

Install required Python packages:

```bash
# Install core dependencies
pip install openai==1.2.4 tqdm==4.66.1 chromadb==0.4.18 uuid==1.30 python-dotenv==1.0.0
pip install tree-sitter==0.20.1 tree-sitter-languages==1.7.0

# Additional utilities
pip install pytest==7.4.3 black==23.11.0

# Save dependencies
pip freeze > requirements.txt
```

**Success criteria**: All packages install without errors.

**Debugging strategy**: If certain packages fail to install, check for system dependency issues or package conflicts.

### Step 5: Environment Configuration

Create configuration files:

```bash
# Create .env file for secrets
cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key_here
CHUNK_SIZE=1000
OVERLAP_SIZE=200
MAX_RESULTS=5
DB_DIR=./data/chroma_db
COLLECTION_NAME=pytorch_docs
EOL

# Create a configuration module
mkdir -p scripts/config
cat > scripts/config/__init__.py << EOL
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "200"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
DB_DIR = os.getenv("DB_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pytorch_docs")
INDEXED_FILE = "./data/indexed_chunks.json"
EOL
```

**Success criteria**: Configuration files created with appropriate values.

**Potential pitfalls**:
- API key security - ensure your .env file is not committed to version control
- Path configuration - verify all paths are correct for Ubuntu 24.04

## Document Processing Pipeline

### Step 1: Create Tree-sitter Parser

Create a script for code-aware document processing:

```bash
# Create document processing module
mkdir -p scripts/document_processing
cat > scripts/document_processing/__init__.py << EOL
# Document processing module initialization
EOL
```

Now, create the parser script:

```bash
cat > scripts/document_processing/parser.py << EOL
import os
from pathlib import Path
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser
import re
from typing import List, Dict, Any, Tuple

class DocumentParser:
    def __init__(self):
        """Initialize the document parser with Tree-sitter."""
        self.markdown_parser = get_parser('markdown')
        # Also initialize Python parser for code extraction
        self.python_parser = get_parser('python')
        
    def extract_title(self, content: str) -> str:
        """Extract title from markdown content."""
        # Look for the first heading
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Untitled Document"
    
    def extract_sections(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Extract sections from markdown, preserving code blocks."""
        # Parse the document
        tree = self.markdown_parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node
        
        # Extract basic document info
        filename = os.path.basename(filepath)
        title = self.extract_title(content)
        
        # Initialize sections list
        sections = []
        
        # Track current heading context
        current_heading = title
        current_heading_level = 1
        
        # Process each child node
        for child in root_node.children:
            # Check if it's a heading
            if child.type == 'atx_heading':
                heading_text_node = child.child_by_field_name('heading_content')
                if heading_text_node:
                    heading_text = content[heading_text_node.start_byte:heading_text_node.end_byte]
                    heading_level = len(content[child.start_byte:heading_text_node.start_byte].strip())
                    current_heading = heading_text
                    current_heading_level = heading_level
            
            # Check if it's a code block
            elif child.type == 'fenced_code_block':
                # Extract language info
                info_string = ''
                info_node = child.child_by_field_name('info_string')
                if info_node:
                    info_string = content[info_node.start_byte:info_node.end_byte]
                
                # Extract code content
                code_text = ''
                for code_node in child.children:
                    if code_node.type == 'code_fence_content':
                        code_text = content[code_node.start_byte:code_node.end_byte]
                
                # Add as a section
                if code_text.strip():
                    sections.append({
                        'text': code_text,
                        'metadata': {
                            'title': current_heading,
                            'source': filename,
                            'chunk_type': 'code',
                            'language': info_string
                        }
                    })
            
            # Check if it's a paragraph or other text content
            elif child.type in ('paragraph', 'block_quote', 'list'):
                text = content[child.start_byte:child.end_byte]
                if text.strip():
                    sections.append({
                        'text': text,
                        'metadata': {
                            'title': current_heading,
                            'source': filename,
                            'chunk_type': 'text'
                        }
                    })
        
        return sections
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse a file and return structured sections."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Only process markdown files
            if filepath.endswith(('.md', '.markdown')):
                return self.extract_sections(content, filepath)
            else:
                # For non-markdown files, treat entire file as a code block
                filename = os.path.basename(filepath)
                extension = Path(filepath).suffix.lstrip('.')
                
                return [{
                    'text': content,
                    'metadata': {
                        'title': filename,
                        'source': filename,
                        'chunk_type': 'code',
                        'language': extension
                    }
                }]
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
            return []
EOL
```

**Success criteria**: Parser script correctly extracts markdown sections and preserves code blocks.

**Testing strategy**:
- Create test markdown files with mixed content
- Verify that code blocks are preserved intact
- Check that heading contexts are correctly associated with content

### Step 2: Create Chunking Strategy

Implement intelligent chunking that respects document structure:

```bash
cat > scripts/document_processing/chunker.py << EOL
import uuid
from typing import List, Dict, Any
import re

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """Initialize the chunker with size parameters."""
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text sections while preserving semantic boundaries."""
        chunks = []
        
        # For code blocks, keep them intact if possible
        if metadata.get('chunk_type') == 'code':
            # If the code is small enough, keep it as one chunk
            if len(text) <= self.chunk_size * 1.5:  # Allow slightly larger chunks for code
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'text': text,
                    'metadata': {**metadata, 'chunk': 1}
                })
            else:
                # For larger code blocks, try to split at function/class boundaries
                chunk_points = self._find_code_chunk_points(text)
                if chunk_points:
                    # Use the identified chunk points
                    start_idx = 0
                    chunk_num = 1
                    
                    for point in chunk_points:
                        if point - start_idx >= self.chunk_size / 2:  # Ensure minimum chunk size
                            chunks.append({
                                'id': str(uuid.uuid4()),
                                'text': text[start_idx:point],
                                'metadata': {**metadata, 'chunk': chunk_num}
                            })
                            start_idx = max(0, point - self.overlap)
                            chunk_num += 1
                    
                    # Add the final chunk
                    if start_idx < len(text):
                        chunks.append({
                            'id': str(uuid.uuid4()),
                            'text': text[start_idx:],
                            'metadata': {**metadata, 'chunk': chunk_num}
                        })
                else:
                    # Fall back to character-based chunking
                    chunks.extend(self._character_chunk(text, metadata))
        else:
            # For regular text, use paragraph and sentence boundaries
            chunks.extend(self._semantic_chunk(text, metadata))
        
        return chunks
    
    def _find_code_chunk_points(self, code: str) -> List[int]:
        """Find good splitting points in code (class/function definitions)."""
        chunk_points = []
        
        # Look for function/class definitions
        patterns = [
            r'^\s*def\s+\w+\s*\(',  # Function definitions
            r'^\s*class\s+\w+\s*[:\(]',  # Class definitions
            r'^\s*@',  # Decorators (often before functions/methods)
            r'^\s*#\s*\w+',  # Section comments
            r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:',  # Main block
        ]
        
        line_start_positions = [0]
        for i, char in enumerate(code):
            if char == '\n':
                line_start_positions.append(i + 1)
        
        for line_start in line_start_positions:
            line_end = code.find('\n', line_start)
            if line_end == -1:
                line_end = len(code)
            
            line = code[line_start:line_end]
            
            # Check if line matches any pattern
            for pattern in patterns:
                if re.match(pattern, line):
                    chunk_points.append(line_start)
                    break
        
        return chunk_points
    
    def _semantic_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text using semantic boundaries like paragraphs and sentences."""
        chunks = []
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_num = 1
        
        for para in paragraphs:
            # If adding this paragraph exceeds the chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # If we have content to save
                if current_chunk:
                    chunks.append({
                        'id': str(uuid.uuid4()),
                        'text': current_chunk.strip(),
                        'metadata': {**metadata, 'chunk': chunk_num}
                    })
                    chunk_num += 1
                
                # Start a new chunk, potentially overlapping with previous content
                if len(para) > self.chunk_size:
                    # If paragraph itself is too large, fall back to sentence splitting
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > self.chunk_size:
                            if current_chunk:
                                chunks.append({
                                    'id': str(uuid.uuid4()),
                                    'text': current_chunk.strip(),
                                    'metadata': {**metadata, 'chunk': chunk_num}
                                })
                                chunk_num += 1
                                current_chunk = sentence + " "
                            else:
                                # If a single sentence is too long, we have to split it
                                chunks.extend(self._character_chunk(sentence, {**metadata, 'note': 'long_sentence'}))
                        else:
                            current_chunk += sentence + " "
                else:
                    current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'id': str(uuid.uuid4()),
                'text': current_chunk.strip(),
                'metadata': {**metadata, 'chunk': chunk_num}
            })
        
        return chunks
    
    def _character_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fall back to character-based chunking with overlap."""
        chunks = []
        
        start = 0
        chunk_num = 1
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # If we're not at the end and can look for a better split point
            if end < len(text):
                # Try to find sentence boundary
                sentence_boundary = text.rfind('.', start, end)
                if sentence_boundary > start + self.chunk_size / 2:
                    end = sentence_boundary + 1
                else:
                    # Try to find word boundary
                    space = text.rfind(' ', start, end)
                    if space > start + self.chunk_size / 2:
                        end = space
            
            chunks.append({
                'id': str(uuid.uuid4()),
                'text': text[start:end].strip(),
                'metadata': {**metadata, 'chunk': chunk_num}
            })
            
            # Move to next chunk with overlap
            start = end - self.overlap if end - self.overlap > start else end
            chunk_num += 1
            
            # Avoid infinite loop for very small texts
            if start >= len(text) or (end == len(text) and chunk_num > 1):
                break
        
        return chunks
    
    def process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all sections into appropriate chunks."""
        all_chunks = []
        
        for section in sections:
            chunks = self.chunk_text(section['text'], section['metadata'])
            all_chunks.extend(chunks)
        
        return all_chunks
EOL
```

**Success criteria**: Chunker correctly splits documents while preserving the semantic integrity of code and text.

**Testing strategy**:
- Test with documents of varying lengths
- Verify that code blocks are kept intact when possible
- Check that chunk boundaries respect semantic boundaries

### Step 3: Build the Document Processor

Create the main document processing script:

```bash
cat > scripts/index_documents.py << EOL
#!/usr/bin/env python3

import os
import json
import glob
from tqdm import tqdm
from typing import List, Dict, Any
import argparse

# Import our modules
from document_processing.parser import DocumentParser
from document_processing.chunker import DocumentChunker
from config import CHUNK_SIZE, OVERLAP_SIZE, INDEXED_FILE

def process_documents(docs_dir: str, output_file: str = INDEXED_FILE) -> List[Dict[str, Any]]:
    """Process all documents in the directory and save chunks."""
    # Initialize our components
    parser = DocumentParser()
    chunker = DocumentChunker(CHUNK_SIZE, OVERLAP_SIZE)
    
    # Find all markdown and Python files
    file_patterns = ['**/*.md', '**/*.markdown', '**/*.py']
    all_files = []
    
    for pattern in file_patterns:
        matched_files = glob.glob(os.path.join(docs_dir, pattern), recursive=True)
        all_files.extend(matched_files)
    
    print(f"Found {len(all_files)} documentation files to process")
    
    # Process each file
    all_chunks = []
    
    for filepath in tqdm(all_files, desc="Processing documents"):
        # Parse the file into sections
        sections = parser.parse_file(filepath)
        
        # Chunk the sections
        chunks = chunker.process_sections(sections)
        
        # Add to our collection
        all_chunks.extend(chunks)
    
    print(f"Generated {len(all_chunks)} chunks from {len(all_files)} files")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the chunks to a file (without embeddings for now)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Saved chunks to {output_file}")
    
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PyTorch documentation files into chunks")
    parser.add_argument("--docs-dir", type=str, default="./pytorch_docs",
                        help="Directory containing documentation files")
    parser.add_argument("--output-file", type=str, default=INDEXED_FILE,
                        help="Output JSON file to save chunks")
    args = parser.parse_args()
    
    chunks = process_documents(args.docs_dir, args.output_file)
    print(f"Successfully processed {len(chunks)} chunks")
EOL

# Make the script executable
chmod +x scripts/index_documents.py
```

**Success criteria**: Script successfully processes documentation files and generates structured chunks.

**Potential pitfalls**:
- File encoding issues - use explicit UTF-8 encoding
- Path issues on Ubuntu - use os.path.join for cross-platform compatibility

## Embedding Generation

### Step 1: Create Embedding Generator

Create a script to generate embeddings for document chunks:

```bash
cat > scripts/embedding/generator.py << EOL
import os
import json
import openai
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from config import OPENAI_API_KEY, EMBEDDING_MODEL

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

class EmbeddingGenerator:
    def __init__(self, model: str = EMBEDDING_MODEL):
        """Initialize the embedding generator."""
        self.model = model
    
    def _batch_embed(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Retry mechanism for API calls
            max_retries = 5
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = openai.Embedding.create(
                        input=batch_texts,
                        model=self.model
                    )
                    batch_embeddings = [item["embedding"] for item in response["data"]]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Respect API rate limits
                    if i + batch_size < len(texts):
                        time.sleep(0.5)  # Sleep to avoid hitting rate limits
                        
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error generating embeddings (attempt {attempt+1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        print(f"Failed to generate embeddings after {max_retries} attempts: {str(e)}")
                        # Return empty embeddings for this batch
                        all_embeddings.extend([[] for _ in batch_texts])
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 20) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of chunks."""
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings
        embeddings = self._batch_embed(texts, batch_size)
        
        # Add embeddings to chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding
        
        print(f"Successfully generated {len(embeddings)} embeddings")
        
        return chunks
    
    def process_chunks_file(self, input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a file containing chunks and add embeddings."""
        # Load chunks from file
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Generate embeddings
        chunks_with_embeddings = self.embed_chunks(chunks)
        
        # Save to file if output_file is provided
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_with_embeddings, f)
        
        return chunks_with_embeddings
EOL
```

**Success criteria**: Script successfully generates embeddings for document chunks.

**Potential pitfalls**:
- API rate limits - implement proper backoff and retry logic
- API key issues - verify the key is correctly loaded
- Large batch sizes - monitor memory usage with large batches

### Step 2: Create Embedding Script

Create the main embedding generation script:

```bash
cat > scripts/generate_embeddings.py << EOL
#!/usr/bin/env python3

import os
import argparse
from embedding.generator import EmbeddingGenerator
from config import INDEXED_FILE

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for document chunks")
    parser.add_argument("--input-file", type=str, default=INDEXED_FILE,
                        help="Input JSON file with document chunks")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output JSON file to save chunks with embeddings (defaults to overwriting input)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Batch size for embedding generation")
    args = parser.parse_args()
    
    # Default to overwriting the input file if no output file is specified
    output_file = args.output_file if args.output_file else args.input_file
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    chunks = generator.process_chunks_file(args.input_file, output_file)
    
    print(f"Successfully generated embeddings for {len(chunks)} chunks")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
EOL

# Make the script executable
chmod +x scripts/generate_embeddings.py
```

**Success criteria**: Script successfully generates and saves embeddings for document chunks.

**Testing strategy**:
- Test with a small number of chunks
- Verify embedding dimensionality
- Check for failed embedding generation

## Vector Database Integration

### Step 1: Create ChromaDB Manager

Create a script to manage the vector database:

```bash
mkdir -p scripts/database
cat > scripts/database/chroma_manager.py << EOL
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import json

from config import DB_DIR, COLLECTION_NAME

class ChromaManager:
    def __init__(self, persist_directory: str = DB_DIR, collection_name: str = COLLECTION_NAME):
        """Initialize the ChromaDB manager."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the client
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
    
    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            # Collection might not exist yet
            pass
        
        # Create a new collection
        self.collection = self.client.create_collection(self.collection_name)
        print(f"Reset collection '{self.collection_name}'")
    
    def get_collection(self):
        """Get or create the collection."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(self.collection_name)
        
        return self.collection
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Add chunks to the collection."""
        collection = self.get_collection()
        
        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add data to collection in batches
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch_num = i // batch_size + 1
            
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(f"Added batch {batch_num}/{total_batches} ({end_idx - i} chunks)")
    
    def load_from_file(self, filepath: str, reset: bool = True, batch_size: int = 100) -> None:
        """Load chunks from a file into ChromaDB."""
        print(f"Loading chunks from {filepath}...")
        
        # Load the chunks
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} chunks from file")
        
        # Reset collection if requested
        if reset:
            self.reset_collection()
        
        # Add chunks to collection
        self.add_chunks(chunks, batch_size)
        
        print(f"Successfully loaded {len(chunks)} chunks into ChromaDB")
    
    def query(self, query_embedding: List[float], 
              n_results: int = 5, 
              filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the collection."""
        collection = self.get_collection()
        
        # Prepare query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Add filters if provided
        if filters:
            query_params["where"] = filters
        
        # Execute query
        results = collection.query(**query_params)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        collection = self.get_collection()
        
        # Get all metadata
        results = collection.get()
        
        # Calculate statistics
        total_chunks = len(results["ids"]) if "ids" in results else 0
        
        # Count by chunk type
        chunk_types = {}
        if "metadatas" in results and results["metadatas"]:
            for metadata in results["metadatas"]:
                chunk_type = metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Count by source
        sources = {}
        if "metadatas" in results and results["metadatas"]:
            for metadata in results["metadatas"]:
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_chunks": total_chunks,
            "chunk_types": chunk_types,
            "sources": sources
        }
EOL
```

**Success criteria**: ChromaDB manager correctly stores and retrieves vector embeddings.

**Potential pitfalls**:
- Persistence issues - verify ChromaDB is correctly saving data
- Memory usage - monitor for large collections
- Embedding dimensionality mismatches

### Step 2: Create Database Loading Script

Create a script to load embeddings into the database:

```bash
cat > scripts/load_to_database.py << EOL
#!/usr/bin/env python3

import argparse
import os
from database.chroma_manager import ChromaManager
from config import INDEXED_FILE, DB_DIR, COLLECTION_NAME

def main():
    parser = argparse.ArgumentParser(description="Load document chunks into ChromaDB")
    parser.add_argument("--input-file", type=str, default=INDEXED_FILE,
                        help="Input JSON file with document chunks and embeddings")
    parser.add_argument("--db-dir", type=str, default=DB_DIR,
                        help="Directory for ChromaDB storage")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME,
                        help="Name of the ChromaDB collection")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for loading data")
    parser.add_argument("--no-reset", action="store_true",
                        help="Don't reset the collection before loading")
    args = parser.parse_args()
    
    # Initialize ChromaDB manager
    db_manager = ChromaManager(args.db_dir, args.collection)
    
    # Load chunks into database
    db_manager.load_from_file(
        args.input_file,
        reset=not args.no_reset,
        batch_size=args.batch_size
    )
    
    # Display collection statistics
    stats = db_manager.get_stats()
    print("\nCollection Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    
    print("  Chunk types:")
    for chunk_type, count in stats['chunk_types'].items():
        print(f"    - {chunk_type}: {count}")
    
    print("  Sources:")
    for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {source}: {count}")
    if len(stats['sources']) > 10:
        print(f"    - ... and {len(stats['sources']) - 10} more sources")

if __name__ == "__main__":
    main()
EOL

# Make the script executable
chmod +x scripts/load_to_database.py
```

**Success criteria**: Script successfully loads embeddings into ChromaDB.

**Testing strategy**:
- Test with small dataset
- Verify statistics match expected values
- Check for database consistency

## Search Interface Implementation

### Step 1: Create Query Processor

Create a script to process search queries:

```bash
mkdir -p scripts/search
cat > scripts/search/query_processor.py << EOL
import openai
from typing import List, Dict, Any

from config import OPENAI_API_KEY, EMBEDDING_MODEL

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

class QueryProcessor:
    def __init__(self, model: str = EMBEDDING_MODEL):
        """Initialize the query processor."""
        self.model = model
    
    def generate_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query using OpenAI's API."""
        try:
            response = openai.Embedding.create(
                input=query,
                model=self.model
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query to extract intent and generate embedding."""
        # Basic preprocessing
        query = query.strip()
        
        # Detect query type (code vs. concept)
        is_code_query = self._is_code_query(query)
        
        # Generate embedding
        embedding = self.generate_embedding(query)
        
        return {
            "query": query,
            "embedding": embedding,
            "is_code_query": is_code_query
        }
    
    def _is_code_query(self, query: str) -> bool:
        """Determine if a query is likely looking for code examples."""
        # Look for code-related keywords
        code_indicators = [
            "code", "example", "implementation", "function", "class",
            "method", "snippet", "syntax", "API", "parameter", "argument",
            "return", "import", "module", "library", "package",
            "how to", "how do i", "sample"
        ]
        
        query_lower = query.lower()
        
        # Check for code indicators
        for indicator in code_indicators:
            if indicator.lower() in query_lower:
                return True
        
        # Check for Python code patterns
        code_patterns = [
            "def ", "class ", "import ", "from ", "torch.", "nn.",
            "self.", "->", "=>", "==", "!=", "+=", "-=", "*=", "/=",
            "():", "@", "if __name__", "__init__", "super()", 
        ]
        
        for pattern in code_patterns:
            if pattern in query:
                return True
        
        return False
EOL
```

**Success criteria**: Query processor correctly analyzes queries and generates embeddings.

**Potential pitfalls**:
- API key issues - verify it's correctly loaded
- Error handling - implement robust exception handling
- Query type detection - test with diverse query patterns

### Step 2: Create Result Formatter

Create a script to format search results:

```bash
cat > scripts/search/result_formatter.py << EOL
from typing import List, Dict, Any

class ResultFormatter:
    def __init__(self):
        """Initialize the result formatter."""
        pass
    
    def format_results(self, results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format ChromaDB results into a structured response."""
        formatted_results = []
        
        # Extract results from ChromaDB response
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        # Format each result
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Create a snippet (first 200 chars)
            snippet = doc[:200] + "..." if len(doc) > 200 else doc
            
            # Convert distance to similarity score (1.0 is exact match)
            similarity = 1.0 - distance
            
            # Add result with rich metadata
            formatted_results.append({
                "title": metadata.get("title", f"Result {i+1}"),
                "snippet": snippet,
                "source": metadata.get("source", ""),
                "chunk_type": metadata.get("chunk_type", "unknown"),
                "score": float(similarity)
            })
        
        # Return the formatted response
        return {
            "results": formatted_results,
            "query": query,
            "count": len(formatted_results)
        }
    
    def rank_results(self, results: Dict[str, Any], is_code_query: bool) -> Dict[str, Any]:
        """Rank and filter results based on query type."""
        if "results" not in results:
            return results
        
        # Get the results
        formatted_results = results["results"]
        
        # Adjust scores based on query type
        for result in formatted_results:
            base_score = result["score"]
            
            # Boost code results for code queries
            if is_code_query and result.get("chunk_type") == "code":
                result["score"] = min(1.0, base_score * 1.2)  # 20% boost, max 1.0
            
            # Boost text results for concept queries
            elif not is_code_query and result.get("chunk_type") == "text":
                result["score"] = min(1.0, base_score * 1.1)  # 10% boost, max 1.0
        
        # Re-sort by score
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update results
        results["results"] = formatted_results
        
        return results
EOL
```

**Success criteria**: Result formatter correctly structures and enhances search results.

**Testing strategy**:
- Test with different query types
- Verify ranking adjustments
- Check formatting consistency

### Step 3: Create Search Interface

Create the main search script:

```bash
cat > scripts/document_search.py << EOL
#!/usr/bin/env python3

import sys
import os
import json
import argparse
from typing import Dict, Any, Optional

from search.query_processor import QueryProcessor
from search.result_formatter import ResultFormatter
from database.chroma_manager import ChromaManager
from config import MAX_RESULTS

class DocumentSearch:
    def __init__(self, db_dir: Optional[str] = None, collection_name: Optional[str] = None):
        """Initialize the document search system."""
        # Initialize components
        self.query_processor = QueryProcessor()
        self.result_formatter = ResultFormatter()
        self.db_manager = ChromaManager(db_dir, collection_name)
    
    def search(self, query: str, num_results: int = MAX_RESULTS, filter_type: Optional[str] = None) -> Dict[str, Any]:
        """Search for documents matching the query."""
        try:
            # Process the query
            query_data = self.query_processor.process_query(query)
            
            # Prepare filters
            filters = {}
            if filter_type:
                filters["chunk_type"] = filter_type
            
            # Search the database
            results = self.db_manager.query(
                query_data["embedding"],
                n_results=num_results,
                filters=filters if filters else None
            )
            
            # Format the results
            formatted_results = self.result_formatter.format_results(results, query)
            
            # Rank results based on query type
            ranked_results = self.result_formatter.rank_results(
                formatted_results,
                query_data["is_code_query"]
            )
            
            return ranked_results
            
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "results": []
            }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Search PyTorch documentation')
    parser.add_argument('query', nargs='?', help='The search query')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--results', '-n', type=int, default=MAX_RESULTS, help='Number of results to return')
    parser.add_argument('--filter', '-f', choices=['code', 'text'], help='Filter results by type')
    args = parser.parse_args()
    
    # Initialize search
    search = DocumentSearch()
    
    if args.interactive:
        # Interactive mode
        print("PyTorch Documentation Search (type 'exit' to quit)")
        while True:
            query = input("\nEnter search query: ")
            if query.lower() in ('exit', 'quit'):
                break
            
            results = search.search(query, args.results, args.filter)
            
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print(f"\nFound {len(results['results'])} results for '{query}':")
                
                for i, res in enumerate(results["results"]):
                    print(f"\n--- Result {i+1} ({res['chunk_type']}) ---")
                    print(f"Title: {res['title']}")
                    print(f"Source: {res['source']}")
                    print(f"Score: {res['score']:.4f}")
                    print(f"Snippet: {res['snippet']}")
    
    elif args.query:
        # Direct query mode
        results = search.search(args.query, args.results, args.filter)
        print(json.dumps(results, indent=2))
    
    else:
        # Read from stdin (for Claude Code tool integration)
        query = sys.stdin.read().strip()
        if query:
            results = search.search(query, args.results)
            print(json.dumps(results))
        else:
            print(json.dumps({"error": "No query provided", "results": []}))

if __name__ == "__main__":
    main()
EOL

# Make the script executable
chmod +x scripts/document_search.py
```

**Success criteria**: Search interface correctly processes queries and returns relevant results.

**Potential pitfalls**:
- Error handling - ensure robust handling of edge cases
- Performance - monitor search latency
- Filter handling - verify filters are correctly applied

## Claude Code Integration

### Step 1: Create Tool Wrapper

Create a wrapper script for Claude Code integration:

```bash
cat > scripts/claude_code_tool.py << EOL
#!/usr/bin/env python3

import sys
import json
import os
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from scripts.search.query_processor import QueryProcessor
from scripts.search.result_formatter import ResultFormatter
from scripts.database.chroma_manager import ChromaManager
from scripts.config import MAX_RESULTS

def search_pytorch_docs(query: str, num_results: int = MAX_RESULTS) -> Dict[str, Any]:
    """Search PyTorch documentation using the vector database."""
    try:
        # Initialize components
        query_processor = QueryProcessor()
        result_formatter = ResultFormatter()
        db_manager = ChromaManager()
        
        # Process the query
        query_data = query_processor.process_query(query)
        
        # Search the database
        results = db_manager.query(
            query_data["embedding"],
            n_results=num_results
        )
        
        # Format the results
        formatted_results = result_formatter.format_results(results, query)
        
        # Rank results based on query type
        ranked_results = result_formatter.rank_results(
            formatted_results,
            query_data["is_code_query"]
        )
        
        return ranked_results
        
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "results": []
        }

def main():
    # Read input from stdin (MCP protocol format)
    try:
        input_data = json.loads(sys.stdin.read())
        query = input_data.get("query", "")
        
        if not query:
            print(json.dumps({"error": "No query provided", "results": []}))
            return
        
        # Set number of results
        num_results = input_data.get("num_results", MAX_RESULTS)
        
        # Perform search
        results = search_pytorch_docs(query, num_results)
        
        # Output results
        print(json.dumps(results))
        
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input", "results": []}))
    except Exception as e:
        print(json.dumps({"error": str(e), "results": []}))

if __name__ == "__main__":
    main()
EOL

# Make the script executable
chmod +x scripts/claude_code_tool.py
```

**Success criteria**: Tool wrapper correctly processes input from Claude Code and returns properly formatted results.

**Potential pitfalls**:
- MCP protocol compatibility
- Python path issues
- Error handling

### Step 2: Tool Registration Script

Create a script to register the tool with Claude Code:

```bash
cat > scripts/register_tool.sh << EOL
#!/bin/bash

# Get the absolute path to the tool script
SCRIPT_DIR=\$(dirname "\$(readlink -f "\$0")")
TOOL_PATH="\$SCRIPT_DIR/claude_code_tool.py"

# Make sure the tool script is executable
chmod +x "\$TOOL_PATH"

# Define the tool description
TOOL_NAME="pytorch_search"
TOOL_DESC="Use this tool to search PyTorch documentation and project guides using advanced code-aware semantic search. The search engine understands both natural language and Python code patterns, making it especially effective for finding API usage examples, PyTorch implementation details, and code snippets. Results include both text explanations and code examples with appropriate context."

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
    }
  },
  "required": ["query"]
}'

# Register the tool with Claude Code CLI
echo "Registering PyTorch Documentation Search Tool with Claude Code..."

# Check if claude mcp command exists
if command -v claude mcp &> /dev/null; then
    # Register the tool
    claude mcp add "\$TOOL_NAME" --description "\$TOOL_DESC" --command "\$TOOL_PATH" --input-schema "\$INPUT_SCHEMA"
    echo "Tool registration successful!"
else
    echo "Error: Claude Code CLI not found. Please install it first."
    exit 1
fi
EOL

# Make the script executable
chmod +x scripts/register_tool.sh
```

**Success criteria**: Tool registration script successfully registers the tool with Claude Code.

**Potential pitfalls**:
- Claude Code CLI installation issues
- Permission problems
- Schema definition errors

## Testing and Evaluation

### Step 1: Create End-to-End Test Script

Create a script to test the entire pipeline:

```bash
cat > scripts/test_pipeline.sh << EOL
#!/bin/bash

set -e  # Exit on any error

# Get the absolute path to the script directory
SCRIPT_DIR=\$(dirname "\$(readlink -f "\$0")")
PROJECT_ROOT=\$(dirname "\$SCRIPT_DIR")

echo "====== PyTorch Documentation Search Tool - Pipeline Test ======"
echo "Project Root: \$PROJECT_ROOT"

# Create test directory
TEST_DIR="\$PROJECT_ROOT/test_data"
mkdir -p "\$TEST_DIR"

# Create sample PyTorch documentation
echo "Creating sample documentation..."
mkdir -p "\$TEST_DIR/pytorch_docs"

# Sample markdown file
cat > "\$TEST_DIR/pytorch_docs/custom_autograd.md" << EOF
# Custom Autograd Functions in PyTorch

This document explains how to create custom autograd functions in PyTorch.

## Overview

PyTorch's autograd system allows users to define custom differentiable operations. This is useful when:

- You need operations not available in PyTorch
- You want to optimize the backward pass of an operation
- You want to combine multiple operations into a single one for clarity

## Implementation

To create a custom autograd function, subclass \`torch.autograd.Function\` and implement the \`forward\` and \`backward\` static methods:

\`\`\`python
import torch

class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        # Save values for backward pass
        ctx.save_for_backward(input1, input2)
        
        # Perform the operation
        result = input1 * input2
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved values
        input1, input2 = ctx.saved_tensors
        
        # Calculate gradients
        grad_input1 = grad_output * input2
        grad_input2 = grad_output * input1
        
        return grad_input1, grad_input2

# Usage
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
z = CustomFunction.apply(x, y)  # Returns 6.0
z.backward()
print(x.grad)  # Should be 3.0
print(y.grad)  # Should be 2.0
\`\`\`

## Important Considerations

1. The \`forward\` method can take any number of arguments, but must return a tensor or tuple of tensors.
2. The \`backward\` method should return the same number of gradients as there are inputs to \`forward\`.
3. Use \`ctx.save_for_backward\` to save tensors needed for the backward pass.
4. If an input doesn't require gradients, the corresponding gradient can be \`None\`.

## Performance Optimization

When implementing custom autograd functions, consider:

- Memory usage: Only save what's needed for the backward pass
- Computational efficiency: Optimize the backward computation
- Numeric stability: Watch out for operations that might cause instability
EOF

# Sample Python file
cat > "\$TEST_DIR/pytorch_docs/custom_nn_module.py" << EOF
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputNet(nn.Module):
    """A neural network module that takes multiple inputs."""
    
    def __init__(self, input1_size, input2_size, hidden_size, output_size):
        super(MultiInputNet, self).__init__()
        
        # Layers for first input
        self.fc1_input1 = nn.Linear(input1_size, hidden_size)
        
        # Layers for second input
        self.fc1_input2 = nn.Linear(input2_size, hidden_size)
        
        # Combined layers
        self.fc_combined = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_output = nn.Linear(hidden_size, output_size)
        
        # Batch normalization and dropout for regularization
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input1, input2):
        # Process first input
        x1 = F.relu(self.fc1_input1(input1))
        
        # Process second input
        x2 = F.relu(self.fc1_input2(input2))
        
        # Combine the processed inputs
        combined = torch.cat((x1, x2), dim=1)
        combined = self.bn(combined)
        
        # Final processing
        x = F.relu(self.fc_combined(combined))
        x = self.dropout(x)
        x = self.fc_output(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Create sample inputs
    input1 = torch.randn(32, 10)  # Batch size 32, input size 10
    input2 = torch.randn(32, 15)  # Batch size 32, input size 15
    
    # Instantiate the model
    model = MultiInputNet(input1_size=10, input2_size=15, hidden_size=50, output_size=5)
    
    # Forward pass
    output = model(input1, input2)
    print(f"Output shape: {output.shape}")  # Should be [32, 5]
EOF

echo "Running pipeline..."

# 1. Process documents
echo -e "\n===== Step 1: Processing Documents ====="
python "\$SCRIPT_DIR/index_documents.py" --docs-dir "\$TEST_DIR/pytorch_docs" --output-file "\$TEST_DIR/indexed_chunks.json"

# 2. Generate embeddings
echo -e "\n===== Step 2: Generating Embeddings ====="
python "\$SCRIPT_DIR/generate_embeddings.py" --input-file "\$TEST_DIR/indexed_chunks.json" --output-file "\$TEST_DIR/chunks_with_embeddings.json"

# 3. Load to database
echo -e "\n===== Step 3: Loading to Database ====="
python "\$SCRIPT_DIR/load_to_database.py" --input-file "\$TEST_DIR/chunks_with_embeddings.json" --db-dir "\$TEST_DIR/chroma_db"

# 4. Test search
echo -e "\n===== Step 4: Testing Search ====="
echo "Test Query 1: How to create a custom autograd function"
python "\$SCRIPT_DIR/document_search.py" "How to create a custom autograd function" --filter code

echo -e "\nTest Query 2: MultiInputNet implementation"
python "\$SCRIPT_DIR/document_search.py" "MultiInputNet implementation" --filter code

echo -e "\nPipeline test completed successfully!"
EOL

# Make the script executable
chmod +x scripts/test_pipeline.sh
```

**Success criteria**: End-to-end test script successfully executes the entire pipeline with test data.

**Potential pitfalls**:
- Path issues - use absolute paths where possible
- Error propagation - ensure good error handling
- Test data quality - ensure test data is representative

### Step 2: Create Unit Tests

Create basic unit tests for key components:

```bash
mkdir -p tests
cat > tests/test_chunker.py << EOL
import sys
import os
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from scripts.document_processing.chunker import DocumentChunker

def test_chunk_text_code():
    """Test chunking of code blocks."""
    chunker = DocumentChunker(chunk_size=500, overlap=100)
    
    # Test code chunking
    code = """
def function1():
    print("Hello World")
    return 42

def function2():
    x = 10
    y = 20
    return x + y

class MyClass:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value
"""
    
    metadata = {"chunk_type": "code", "source": "test.py", "title": "Test"}
    chunks = chunker.chunk_text(code, metadata)
    
    # Assertions
    assert len(chunks) > 0, "No chunks were generated"
    assert all(c["metadata"]["chunk_type"] == "code" for c in chunks), "Chunk type is not preserved"
    
    # Check that function definitions aren't split
    found_function1 = False
    found_function2 = False
    found_class = False
    
    for chunk in chunks:
        text = chunk["text"]
        if "def function1()" in text and "return 42" in text:
            found_function1 = True
        if "def function2()" in text and "return x + y" in text:
            found_function2 = True
        if "class MyClass:" in text and "def increment(self):" in text:
            found_class = True
    
    assert found_function1, "function1 was split across chunks"
    assert found_function2, "function2 was split across chunks"
    assert found_class, "MyClass was split across chunks"

def test_chunk_text_markdown():
    """Test chunking of markdown text."""
    chunker = DocumentChunker(chunk_size=500, overlap=100)
    
    # Test markdown chunking
    markdown = """
# Title

## Section 1

This is the first paragraph of section 1.
It has multiple sentences for testing.
The chunker should try to keep sentences together.

This is the second paragraph of section 1.

## Section 2

This is a paragraph in section 2.
It also has multiple sentences.

* List item 1
* List item 2
* List item 3

## Section 3

Final section with some content.
"""
    
    metadata = {"chunk_type": "text", "source": "test.md", "title": "Test"}
    chunks = chunker.chunk_text(markdown, metadata)
    
    # Assertions
    assert len(chunks) > 0, "No chunks were generated"
    assert all(c["metadata"]["chunk_type"] == "text" for c in chunks), "Chunk type is not preserved"
    
    # Check that paragraphs aren't split mid-sentence
    for chunk in chunks:
        text = chunk["text"]
        sentences = text.split('.')
        for sentence in sentences[:-1]:  # Exclude last one which might be cut off
            if sentence and not sentence.endswith(('.', '!', '?')):
                # If it's not empty and doesn't end with punctuation, it might be cut
                # But we need to check it's not just a list item or heading
                if not (sentence.strip().startswith('#') or sentence.strip().startswith('*')):
                    assert False, f"Sentence was cut: {sentence}"
    
    # All good if we reach here
    assert True

def test_process_sections():
    """Test processing of multiple sections."""
    chunker = DocumentChunker(chunk_size=500, overlap=100)
    
    # Create test sections
    sections = [
        {
            "text": "# Title\n\nThis is a test paragraph.\nIt has multiple lines.",
            "metadata": {"chunk_type": "text", "source": "test.md", "title": "Test"}
        },
        {
            "text": "def test():\n    return 'Hello World'",
            "metadata": {"chunk_type": "code", "source": "test.py", "title": "Test"}
        }
    ]
    
    # Process sections
    chunks = chunker.process_sections(sections)
    
    # Assertions
    assert len(chunks) >= 2, "Not enough chunks generated"
    assert any(c["metadata"]["chunk_type"] == "text" for c in chunks), "Missing text chunks"
    assert any(c["metadata"]["chunk_type"] == "code" for c in chunks), "Missing code chunks"
    
    # Verify IDs are unique
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"
EOL

cat > tests/test_parser.py << EOL
import sys
import os
import pytest
import tempfile

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from scripts.document_processing.parser import DocumentParser

def test_extract_title():
    """Test title extraction from markdown."""
    parser = DocumentParser()
    
    # Test with a title
    content = "# This is a title\n\nSome content."
    assert parser.extract_title(content) == "This is a title"
    
    # Test with no title
    content = "Some content without a title."
    assert parser.extract_title(content) == "Untitled Document"
    
    # Test with multiple headings
    content = "# Main Title\n\n## Section 1\n\nContent"
    assert parser.extract_title(content) == "Main Title"

def test_extract_sections():
    """Test section extraction from markdown."""
    parser = DocumentParser()
    
    # Create a test file
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+', delete=False) as f:
        f.write("""# Test Document

This is a paragraph.

```python
def test_function():
    return "Hello World"
```

## Section 1

Another paragraph.

* List item 1
* List item 2

```
Plain code block
```
""")
        filepath = f.name
    
    try:
        # Read the content
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract sections
        sections = parser.extract_sections(content, filepath)
        
        # Assertions
        assert len(sections) >= 4, "Not enough sections extracted"
        
        # Check for code blocks
        code_sections = [s for s in sections if s["metadata"]["chunk_type"] == "code"]
        assert len(code_sections) >= 2, "Not enough code sections found"
        
        # Check for text blocks
        text_sections = [s for s in sections if s["metadata"]["chunk_type"] == "text"]
        assert len(text_sections) >= 2, "Not enough text sections found"
        
        # Check metadata
        for section in sections:
            assert "title" in section["metadata"], "Missing title in metadata"
            assert "source" in section["metadata"], "Missing source in metadata"
            assert "chunk_type" in section["metadata"], "Missing chunk_type in metadata"
        
        # Check Python code block
        python_blocks = [s for s in sections if s["metadata"].get("language") == "python"]
        assert len(python_blocks) >= 1, "Python code block not detected"
        assert "def test_function()" in python_blocks[0]["text"], "Function code not preserved"
    
    finally:
        # Clean up
        os.unlink(filepath)

def test_parse_file():
    """Test parsing of different file types."""
    parser = DocumentParser()
    
    # Test with markdown file
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+', delete=False) as f:
        f.write("# Markdown Test\n\nContent")
        md_path = f.name
    
    # Test with Python file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+', delete=False) as f:
        f.write("def python_test():\n    pass")
        py_path = f.name
    
    try:
        # Parse markdown file
        md_sections = parser.parse_file(md_path)
        assert len(md_sections) > 0, "No sections from markdown file"
        assert md_sections[0]["metadata"]["source"] == os.path.basename(md_path), "Wrong source for markdown"
        
        # Parse Python file
        py_sections = parser.parse_file(py_path)
        assert len(py_sections) > 0, "No sections from Python file"
        assert py_sections[0]["metadata"]["chunk_type"] == "code", "Python file not marked as code"
        assert py_sections[0]["metadata"]["language"] == "py", "Wrong language for Python file"
    
    finally:
        # Clean up
        os.unlink(md_path)
        os.unlink(py_path)
EOL

cat > tests/test_search.py << EOL
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from scripts.search.query_processor import QueryProcessor
from scripts.search.result_formatter import ResultFormatter

def test_is_code_query():
    """Test code query detection."""
    processor = QueryProcessor()
    
    # Code queries
    assert processor._is_code_query("How to implement a custom nn.Module"), "Should detect code query"
    assert processor._is_code_query("Example of torch.nn.Linear"), "Should detect API reference"
    assert processor._is_code_query("def forward() implementation"), "Should detect function reference"
    assert processor._is_code_query("if __name__ == '__main__' usage"), "Should detect code pattern"
    
    # Non-code queries
    assert not processor._is_code_query("What is PyTorch?"), "Should not mark as code query"
    assert not processor._is_code_query("Explain backpropagation"), "Should not mark as code query"

@patch('openai.Embedding.create')
def test_generate_embedding(mock_embedding):
    """Test embedding generation."""
    # Mock the OpenAI API response
    mock_embedding.return_value = {
        "data": [{
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }]
    }
    
    processor = QueryProcessor()
    embedding = processor.generate_embedding("Test query")
    
    # Check the embedding
    assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5], "Embedding doesn't match expected values"
    
    # Verify the API was called correctly
    mock_embedding.assert_called_once()
    args, kwargs = mock_embedding.call_args
    assert kwargs["input"] == "Test query", "Wrong query sent to API"
    assert kwargs["model"] == "text-embedding-ada-002", "Wrong model used"

def test_format_results():
    """Test result formatting."""
    formatter = ResultFormatter()
    
    # Create mock ChromaDB results
    mock_results = {
        "documents": [["Code sample 1", "Text explanation 1"]],
        "metadatas": [[
            {"title": "Code Example", "source": "example.py", "chunk_type": "code"},
            {"title": "Documentation", "source": "doc.md", "chunk_type": "text"}
        ]],
        "distances": [[0.1, 0.3]]
    }
    
    formatted = formatter.format_results(mock_results, "test query")
    
    # Check structure
    assert "results" in formatted, "Missing results key"
    assert "query" in formatted, "Missing query key"
    assert formatted["query"] == "test query", "Wrong query in results"
    assert len(formatted["results"]) == 2, "Wrong number of results"
    
    # Check result details
    assert formatted["results"][0]["title"] == "Code Example", "Wrong title"
    assert formatted["results"][0]["chunk_type"] == "code", "Wrong chunk type"
    assert formatted["results"][0]["score"] == 0.9, "Wrong score calculation"  # 1.0 - 0.1
    
    assert formatted["results"][1]["title"] == "Documentation", "Wrong title"
    assert formatted["results"][1]["chunk_type"] == "text", "Wrong chunk type"
    assert formatted["results"][1]["score"] == 0.7, "Wrong score calculation"  # 1.0 - 0.3

def test_rank_results():
    """Test result ranking."""
    formatter = ResultFormatter()
    
    # Create mock results
    results = {
        "results": [
            {"score": 0.8, "chunk_type": "text", "title": "Text 1"},
            {"score": 0.7, "chunk_type": "code", "title": "Code 1"},
            {"score": 0.6, "chunk_type": "text", "title": "Text 2"},
            {"score": 0.5, "chunk_type": "code", "title": "Code 2"}
        ],
        "query": "test query"
    }
    
    # Test with code query
    code_ranked = formatter.rank_results(results.copy(), True)
    assert code_ranked["results"][0]["chunk_type"] == "code", "Code should be boosted for code query"
    assert code_ranked["results"][0]["score"] > 0.8, "Code score should be boosted"
    
    # Test with concept query
    text_ranked = formatter.rank_results(results.copy(), False)
    assert text_ranked["results"][0]["chunk_type"] == "text", "Text should be boosted for concept query"
EOL

cat > tests/run_tests.sh << EOL
#!/bin/bash

# Change to the tests directory
cd "\$(dirname "\$0")"

# Run pytest
pytest -v
EOL

# Make the test script executable
chmod +x tests/run_tests.sh
```

**Success criteria**: Unit tests successfully verify the functionality of key components.

**Testing strategy**:
- Run tests with pytest
- Verify all tests pass
- Check code coverage

### Step 3: Create Evaluation Script

Create a script to evaluate search quality:

```bash
cat > scripts/evaluate_search.py << EOL
#!/usr/bin/env python3

import json
import os
import argparse
from typing import List, Dict, Any
import random
import time

from search.query_processor import QueryProcessor
from search.result_formatter import ResultFormatter
from database.chroma_manager import ChromaManager
from config import MAX_RESULTS

# Sample evaluation queries
EVALUATION_QUERIES = [
    # Code queries
    {"query": "How to create a custom autograd function", "type": "code"},
    {"query": "Implementation of nn.Module with multiple inputs", "type": "code"},
    {"query": "Implementing custom loss function PyTorch", "type": "code"},
    {"query": "How to save and load models in PyTorch", "type": "code"},
    {"query": "Batch normalization implementation", "type": "code"},
    
    # Concept queries
    {"query": "What is autograd in PyTorch", "type": "concept"},
    {"query": "Explain backpropagation in PyTorch", "type": "concept"},
    {"query": "PyTorch vs TensorFlow differences", "type": "concept"},
    {"query": "How does distributed training work", "type": "concept"},
    {"query": "Transfer learning in PyTorch", "type": "concept"}
]

def evaluate_search(db_dir: str = None, collection_name: str = None) -> Dict[str, Any]:
    """Evaluate search quality and performance."""
    # Initialize components
    query_processor = QueryProcessor()
    result_formatter = ResultFormatter()
    db_manager = ChromaManager(db_dir, collection_name)
    
    results = {
        "queries": [],
        "summary": {
            "total_queries": len(EVALUATION_QUERIES),
            "avg_query_time": 0,
            "avg_results_returned": 0,
            "code_query_metrics": {
                "avg_code_results": 0,
                "avg_text_results": 0
            },
            "concept_query_metrics": {
                "avg_code_results": 0,
                "avg_text_results": 0
            }
        }
    }
    
    total_query_time = 0
    total_results = 0
    
    code_queries = [q for q in EVALUATION_QUERIES if q["type"] == "code"]
    concept_queries = [q for q in EVALUATION_QUERIES if q["type"] == "concept"]
    
    code_code_results = 0
    code_text_results = 0
    concept_code_results = 0
    concept_text_results = 0
    
    # Run evaluation queries
    for query_info in EVALUATION_QUERIES:
        query = query_info["query"]
        query_type = query_info["type"]
        
        # Measure query time
        start_time = time.time()
        
        # Process the query
        query_data = query_processor.process_query(query)
        
        # Search the database
        search_results = db_manager.query(
            query_data["embedding"],
            n_results=MAX_RESULTS
        )
        
        # Format the results
        formatted_results = result_formatter.format_results(search_results, query)
        
        # Rank results based on query type
        ranked_results = result_formatter.rank_results(
            formatted_results,
            query_data["is_code_query"]
        )
        
        query_time = time.time() - start_time
        
        # Count result types
        code_results = sum(1 for r in ranked_results["results"] if r["chunk_type"] == "code")
        text_results = sum(1 for r in ranked_results["results"] if r["chunk_type"] == "text")
        
        # Update metrics
        if query_type == "code":
            code_code_results += code_results
            code_text_results += text_results
        else:
            concept_code_results += code_results
            concept_text_results += text_results
        
        # Add to results
        results["queries"].append({
            "query": query,
            "type": query_type,
            "is_code_query": query_data["is_code_query"],
            "query_time": query_time,
            "num_results": len(ranked_results["results"]),
            "code_results": code_results,
            "text_results": text_results,
            "top_result": ranked_results["results"][0] if ranked_results["results"] else None
        })
        
        total_query_time += query_time
        total_results += len(ranked_results["results"])
    
    # Calculate summary metrics
    num_queries = len(EVALUATION_QUERIES)
    results["summary"]["avg_query_time"] = total_query_time / num_queries
    results["summary"]["avg_results_returned"] = total_results / num_queries
    
    num_code_queries = len(code_queries)
    if num_code_queries > 0:
        results["summary"]["code_query_metrics"]["avg_code_results"] = code_code_results / num_code_queries
        results["summary"]["code_query_metrics"]["avg_text_results"] = code_text_results / num_code_queries
    
    num_concept_queries = len(concept_queries)
    if num_concept_queries > 0:
        results["summary"]["concept_query_metrics"]["avg_code_results"] = concept_code_results / num_concept_queries
        results["summary"]["concept_query_metrics"]["avg_text_results"] = concept_text_results / num_concept_queries
    
    # Calculate precision of query type detection
    correct_type_detection = sum(1 for q in results["queries"] 
                                if (q["type"] == "code" and q["is_code_query"]) or 
                                   (q["type"] == "concept" and not q["is_code_query"]))
    
    results["summary"]["query_type_detection_accuracy"] = correct_type_detection / num_queries
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate search quality and performance")
    parser.add_argument("--output-file", type=str, default="evaluation_results.json",
                        help="File to save evaluation results")
    args = parser.parse_args()
    
    print("Evaluating search quality and performance...")
    results = evaluate_search()
    
    # Save results to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total Queries: {results['summary']['total_queries']}")
    print(f"Average Query Time: {results['summary']['avg_query_time']:.4f} seconds")
    print(f"Average Results Returned: {results['summary']['avg_results_returned']:.2f}")
    print(f"Query Type Detection Accuracy: {results['summary']['query_type_detection_accuracy'] * 100:.1f}%")
    
    print("\nCode Query Metrics:")
    print(f"  Avg Code Results: {results['summary']['code_query_metrics']['avg_code_results']:.2f}")
    print(f"  Avg Text Results: {results['summary']['code_query_metrics']['avg_text_results']:.2f}")
    
    print("\nConcept Query Metrics:")
    print(f"  Avg Code Results: {results['summary']['concept_query_metrics']['avg_code_results']:.2f}")
    print(f"  Avg Text Results: {results['summary']['concept_query_metrics']['avg_text_results']:.2f}")
    
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
EOL

# Make the script executable
chmod +x scripts/evaluate_search.py
```

**Success criteria**: Evaluation script successfully measures search quality and performance.

**Testing strategy**:
- Run with a small test dataset
- Verify metrics are calculated correctly
- Check for reasonable search quality

## System Monitoring and Maintenance

### Step 1: Create Monitoring Script

Create a script to monitor system health:

```bash
cat > scripts/monitor_system.py << EOL
#!/usr/bin/env python3

import os
import json
import time
import psutil
import argparse
from typing import Dict, Any
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/system_monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('system_monitor')

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        }
    }
    
    return stats

def check_database_health(db_dir: str) -> Dict[str, Any]:
    """Check database health."""
    try:
        # Import here to avoid circular imports
        from database.chroma_manager import ChromaManager
        
        # Initialize ChromaDB manager
        db_manager = ChromaManager(db_dir)
        
        # Get collection statistics
        stats = db_manager.get_stats()
        
        return {
            "status": "healthy",
            "total_chunks": stats.get("total_chunks", 0),
            "chunk_types": stats.get("chunk_types", {}),
            "source_count": len(stats.get("sources", {}))
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def monitor_system(interval: int = 60, db_dir: str = None, output_file: str = None):
    """Monitor system health at regular intervals."""
    logger.info(f"Starting system monitoring (interval: {interval}s)")
    
    try:
        while True:
            # Get system stats
            stats = get_system_stats()
            
            # Check database health if db_dir provided
            if db_dir:
                stats["database"] = check_database_health(db_dir)
            
            # Log stats
            logger.info(f"CPU: {stats['cpu']['percent']}%, "
                        f"Memory: {stats['memory']['percent']}%, "
                        f"Disk: {stats['disk']['percent']}%")
            
            # Save to file if requested
            if output_file:
                try:
                    # Read existing data
                    data = []
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            data = json.load(f)
                    
                    # Append new stats
                    data.append(stats)
                    
                    # Trim to last 1000 entries
                    if len(data) > 1000:
                        data = data[-1000:]
                    
                    # Write back
                    with open(output_file, 'w') as f:
                        json.dump(data, f)
                except Exception as e:
                    logger.error(f"Error saving stats: {str(e)}")
            
            # Sleep
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Monitor system health")
    parser.add_argument("--interval", type=int, default=60,
                        help="Monitoring interval in seconds")
    parser.add_argument("--db-dir", type=str, default=None,
                        help="ChromaDB directory to monitor")
    parser.add_argument("--output-file", type=str, default="logs/system_stats.json",
                        help="File to save monitoring data")
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Start monitoring
    monitor_system(args.interval, args.db_dir, args.output_file)

if __name__ == "__main__":
    main()
EOL

# Make the script executable
chmod +x scripts/monitor_system.py
```

**Success criteria**: Monitoring script successfully collects and logs system information.

**Potential pitfalls**:
- Dependency on psutil - ensure it's installed
- Log file permissions
- Database connectivity issues

### Step 2: Create Maintenance Script

Create a script for system maintenance:

```bash
cat > scripts/maintenance.sh << EOL
#!/bin/bash

# PyTorch Documentation Search Tool - Maintenance Script
# This script performs routine maintenance tasks for the search system

# Get the script directory
SCRIPT_DIR=\$(dirname "\$(readlink -f "\$0")")
PROJECT_ROOT=\$(dirname "\$SCRIPT_DIR")

# Configuration
DB_DIR="\$PROJECT_ROOT/data/chroma_db"
LOG_DIR="\$PROJECT_ROOT/logs"
BACKUP_DIR="\$PROJECT_ROOT/backups"

# Create directories if they don't exist
mkdir -p "\$LOG_DIR"
mkdir -p "\$BACKUP_DIR"

# Timestamp for backup files
TIMESTAMP=\$(date +"%Y%m%d_%H%M%S")

# Log file
LOG_FILE="\$LOG_DIR/maintenance_\$TIMESTAMP.log"

# Start logging
echo "=== PyTorch Documentation Search Tool - Maintenance ===" | tee -a "\$LOG_FILE"
echo "Started at: \$(date)" | tee -a "\$LOG_FILE"

# 1. Backup database
echo -e "\n== Backing up database ==" | tee -a "\$LOG_FILE"
if [ -d "\$DB_DIR" ]; then
    BACKUP_FILE="\$BACKUP_DIR/chroma_db_\$TIMESTAMP.tar.gz"
    echo "Creating backup: \$BACKUP_FILE" | tee -a "\$LOG_FILE"
    tar -czf "\$BACKUP_FILE" -C "\$(dirname "\$DB_DIR")" "\$(basename "\$DB_DIR")" 2>> "\$LOG_FILE"
    
    if [ \$? -eq 0 ]; then
        echo "Backup created successfully" | tee -a "\$LOG_FILE"
    else
        echo "ERROR: Backup failed" | tee -a "\$LOG_FILE"
    fi
else
    echo "WARNING: Database directory not found at \$DB_DIR" | tee -a "\$LOG_FILE"
fi

# 2. Clean old logs
echo -e "\n== Cleaning old logs ==" | tee -a "\$LOG_FILE"
find "\$LOG_DIR" -name "*.log" -type f -mtime +30 -exec rm {} \; -print | tee -a "\$LOG_FILE"
echo "Removed old log files (older than 30 days)" | tee -a "\$LOG_FILE"

# 3. Clean old backups
echo -e "\n== Cleaning old backups ==" | tee -a "\$LOG_FILE"
find "\$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +60 -exec rm {} \; -print | tee -a "\$LOG_FILE"
echo "Removed old backup files (older than 60 days)" | tee -a "\$LOG_FILE"

# 4. Check system health
echo -e "\n== Checking system health ==" | tee -a "\$LOG_FILE"
python -m scripts.monitor_system --interval 0 --db-dir "\$DB_DIR" 2>&1 | tee -a "\$LOG_FILE"

# 5. Check for available updates
echo -e "\n== Checking for dependency updates ==" | tee -a "\$LOG_FILE"
pip list --outdated 2>&1 | tee -a "\$LOG_FILE"

echo -e "\n=== Maintenance completed at: \$(date) ===" | tee -a "\$LOG_FILE"
EOL

# Make the script executable
chmod +x scripts/maintenance.sh
```

**Success criteria**: Maintenance script successfully performs routine maintenance tasks.

**Potential pitfalls**:
- Backup permissions
- Disk space issues
- Path resolution problems

## Appendix: Troubleshooting Guide

```bash
cat > docs/troubleshooting.md << EOL
# PyTorch Documentation Search Tool - Troubleshooting Guide

This guide provides solutions for common issues that may arise during the implementation and operation of the PyTorch Documentation Search Tool.

## Installation Issues

### Missing System Dependencies

**Problem**: Error when installing Python packages, particularly those with C extensions.

**Solution**:
\`\`\`bash
# Install required system dependencies
sudo apt update
sudo apt install -y python3-dev build-essential cmake
\`\`\`

### Virtual Environment Issues

**Problem**: Errors activating or using the virtual environment.

**Solution**:
\`\`\`bash
# Remove and recreate the virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## Document Processing Issues

### Tree-sitter Parser Errors

**Problem**: Errors related to Tree-sitter parsing or language loading.

**Solution**:
\`\`\`bash
# Reinstall Tree-sitter and languages
pip uninstall -y tree-sitter tree-sitter-languages
pip install tree-sitter==0.20.1 tree-sitter-languages==1.7.0

# Check language installation
python -c "from tree_sitter_languages import get_language; print(get_language('python').__dict__)"
\`\`\`

### File Encoding Issues

**Problem**: UnicodeDecodeError when processing files.

**Solution**: Ensure all files are UTF-8 encoded:
\`\`\`bash
# Check file encoding
file -i your_file.md

# Convert to UTF-8 if needed
iconv -f ISO-8859-1 -t UTF-8 your_file.md > your_file_utf8.md
mv your_file_utf8.md your_file.md
\`\`\`

## Embedding Generation Issues

### OpenAI API Issues

**Problem**: Errors when connecting to the OpenAI API.

**Solution**:
1. Check your API key in the .env file
2. Verify internet connectivity
3. Implement more robust error handling:
\`\`\`python
try:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]
except openai.error.AuthenticationError:
    print("Authentication error: Check your API key")
except openai.error.RateLimitError:
    print("Rate limit exceeded: Waiting before retrying...")
    time.sleep(60)  # Wait 60 seconds before retrying
except openai.error.APIError as e:
    print(f"API error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
\`\`\`

## Database Issues

### ChromaDB Persistence Issues

**Problem**: ChromaDB not saving or loading data correctly.

**Solution**:
\`\`\`bash
# Check database directory
ls -la ./data/chroma_db

# If corrupted, restore from backup
rm -rf ./data/chroma_db
tar -xzf ./backups/chroma_db_YYYYMMDD_HHMMSS.tar.gz -C ./data/
\`\`\`

### Memory Issues

**Problem**: System running out of memory when loading large collections.

**Solution**:
1. Reduce batch sizes in scripts:
\`\`\`bash
python scripts/load_to_database.py --batch-size 50  # Smaller batch size
\`\`\`

2. Add swap space if needed:
\`\`\`bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
\`\`\`

## Search Issues

### Slow Queries

**Problem**: Search queries taking too long to process.

**Solution**:
1. Check system resources:
\`\`\`bash
python scripts/monitor_system.py --interval 0
\`\`\`

2. Optimize database:
\`\`\`bash
# Rebuild the collection with optimized parameters
python scripts/load_to_database.py --batch-size 200
\`\`\`

### Poor Search Results

**Problem**: Search results not returning relevant documents.

**Solution**:
1. Test different queries using the evaluation script:
\`\`\`bash
python scripts/evaluate_search.py
\`\`\`

2. Adjust chunking parameters for better document segmentation:
\`\`\`bash
# Edit .env file to change chunking parameters
CHUNK_SIZE=800  # Try smaller chunks
OVERLAP_SIZE=150  # Increase overlap
\`\`\`

3. Re-process your documents with the new parameters.

## Claude Code Integration Issues

### Tool Registration Failures

**Problem**: Claude Code CLI fails to register the tool.

**Solution**:
\`\`\`bash
# Check Claude Code CLI installation
claude --version

# Register manually
claude mcp add pytorch_search \
  --description "Use this tool to search PyTorch documentation and project guides using advanced code-aware semantic search." \
  --command "$(pwd)/scripts/claude_code_tool.py" \
  --input-schema '{"type":"object","properties":{"query":{"type":"string","description":"The search query about PyTorch"}},"required":["query"]}'
\`\`\`

### Tool Communication Issues

**Problem**: Claude can't communicate with the search tool.

**Solution**:
1. Check the script permissions:
\`\`\`bash
chmod +x scripts/claude_code_tool.py
\`\`\`

2. Test the tool directly:
\`\`\`bash
echo '{"query": "How to implement a custom autograd function"}' | python scripts/claude_code_tool.py
\`\`\`

3. Check for proper JSON handling in the tool.

## System Monitoring Issues

### Log Directory Permissions

**Problem**: Unable to write to log files.

**Solution**:
\`\`\`bash
# Create log directory with proper permissions
mkdir -p logs
chmod 755 logs
\`\`\`

### Disk Space Issues

**Problem**: System running out of disk space.

**Solution**:
\`\`\`bash
# Check disk usage
df -h

# Clean up old backups and logs
find ./backups -name "*.tar.gz" -type f -mtime +30 -delete
find ./logs -name "*.log" -type f -mtime +15 -delete
\`\`\`

## Performance Optimization

If you're experiencing general performance issues, consider these optimizations:

1. **Hardware Recommendations**:
   - At least 4 CPU cores
   - 8GB RAM minimum, 16GB recommended
   - SSD storage for database

2. **Software Optimizations**:
   - Use PyPy for faster Python execution where possible
   - Enable JIT compilation for ChromaDB
   - Use smaller, more targeted documentation sets

3. **Query Optimization**:
   - Implement query caching
   - Pre-compute embeddings for common queries
   - Use more efficient filtering strategies

## Getting Help

If you're still experiencing issues:

1. Check the logs for detailed error messages
2. Run the test suite to verify component functionality
3. Check for updates to dependencies
4. Consult the project documentation
EOL
```

This detailed implementation plan provides a comprehensive roadmap for developing the PyTorch documentation search tool on Ubuntu 24.04. The plan covers everything from environment setup to system maintenance, with careful attention to potential pitfalls and debugging strategies at each step.

The implementation is designed with a modular architecture, making it easy to extend and optimize individual components as needed. The code-aware chunking strategy ensures that code examples remain intact, while the search interface provides relevant results for both code-related and conceptual queries.

By following this plan, you'll create a powerful search tool that enhances the developer experience with PyTorch by providing fast, accurate access to documentation and code examples.
