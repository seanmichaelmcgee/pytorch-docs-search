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