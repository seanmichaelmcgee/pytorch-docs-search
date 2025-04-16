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