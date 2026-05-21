"""
Text Chunking Module for the RAG Q&A System.

Implements multiple chunking strategies for splitting documents into
smaller segments suitable for embedding and retrieval. Provides
comparative analysis utilities for evaluating different strategies.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY, logger


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""
    text: str
    chunk_id: int
    source_doc: str
    start_char: int
    end_char: int
    strategy: str
    chunk_size_config: int
    overlap_config: int
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_doc": self.source_doc,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "strategy": self.strategy,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }


def chunk_fixed_size(
    text: str,
    source_doc: str = "unknown",
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Split text into fixed-size chunks based on character count.
    
    This is the simplest chunking strategy - splits text at exact character
    boundaries regardless of word or sentence boundaries.
    
    Args:
        text: Input text to chunk
        source_doc: Name of the source document for metadata
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of Chunk objects
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        
        if chunk_text:  # Skip empty chunks
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_doc=source_doc,
                start_char=start,
                end_char=min(end, len(text)),
                strategy="fixed",
                chunk_size_config=chunk_size,
                overlap_config=chunk_overlap,
            ))
            chunk_id += 1
        
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks


def chunk_recursive(
    text: str,
    source_doc: str = "unknown",
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Split text recursively using a hierarchy of separators.
    
    Tries to split on paragraph breaks first, then sentences, then words,
    to maintain semantic coherence within chunks.
    
    Separator hierarchy: \\n\\n -> \\n -> . -> ' ' -> ''
    
    Args:
        text: Input text to chunk
        source_doc: Name of the source document for metadata
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of Chunk objects
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _split_text(text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the separator hierarchy."""
        final_chunks = []
        
        # Find the appropriate separator
        separator = separators[-1]
        for sep in separators:
            if sep in text:
                separator = sep
                break
        
        # Split with the found separator
        splits = text.split(separator) if separator else list(text)
        
        # Combine small splits and recurse on large ones
        current_chunk = ""
        for split in splits:
            piece = split + separator if separator else split
            
            if len(current_chunk) + len(piece) <= chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                
                # If the piece itself is too large, recurse with next separator
                if len(piece) > chunk_size and separators.index(separator) < len(separators) - 1:
                    remaining_seps = separators[separators.index(separator) + 1:]
                    final_chunks.extend(_split_text(piece, remaining_seps))
                else:
                    current_chunk = piece
                    continue
                current_chunk = ""
        
        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())
        
        return final_chunks
    
    # Get raw splits
    raw_chunks = _split_text(text, separators)
    
    # Apply overlap by including end of previous chunk at start of next
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if not chunk_text.strip():
            continue
        
        # Calculate approximate character position
        preceding_text = "".join(raw_chunks[:i])
        start_char = len(preceding_text)
        
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=i,
            source_doc=source_doc,
            start_char=start_char,
            end_char=start_char + len(chunk_text),
            strategy="recursive",
            chunk_size_config=chunk_size,
            overlap_config=chunk_overlap,
        ))
    
    return chunks


def chunk_sentence_based(
    text: str,
    source_doc: str = "unknown",
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Split text into chunks based on sentence boundaries.
    
    Groups complete sentences together until reaching the chunk size limit.
    Ensures no sentence is split across chunks for better semantic coherence.
    
    Args:
        text: Input text to chunk
        source_doc: Name of the source document for metadata
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of Chunk objects
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    # Split into sentences using regex
    # Handles: periods, question marks, exclamation marks followed by space/newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    chunk_id = 0
    char_position = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        if current_length + sentence_len > chunk_size and current_chunk_sentences:
            # Finalize current chunk
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_doc=source_doc,
                start_char=char_position - current_length,
                end_char=char_position,
                strategy="sentence",
                chunk_size_config=chunk_size,
                overlap_config=chunk_overlap,
            ))
            chunk_id += 1
            
            # Keep overlap sentences
            overlap_text = ""
            overlap_sentences = []
            for s in reversed(current_chunk_sentences):
                if len(overlap_text) + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_text = " ".join(overlap_sentences)
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_length = len(" ".join(current_chunk_sentences))
        
        current_chunk_sentences.append(sentence)
        current_length += sentence_len + 1  # +1 for space
        char_position += sentence_len + 1
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            source_doc=source_doc,
            start_char=char_position - current_length,
            end_char=char_position,
            strategy="sentence",
            chunk_size_config=chunk_size,
            overlap_config=chunk_overlap,
        ))
    
    return chunks


# =============================================================================
# Strategy Dispatcher
# =============================================================================

STRATEGY_MAP = {
    "fixed": chunk_fixed_size,
    "recursive": chunk_recursive,
    "sentence": chunk_sentence_based,
}


def chunk_document(
    text: str,
    source_doc: str = "unknown",
    strategy: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Chunk a document using the specified strategy.
    
    This is the main entry point for chunking. It dispatches to the
    appropriate chunking function based on the strategy parameter.
    
    Args:
        text: Input text to chunk
        source_doc: Name of the source document for metadata
        strategy: Chunking strategy ("fixed", "recursive", "sentence")
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of Chunk objects
    """
    strategy = strategy or CHUNKING_STRATEGY
    
    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available: {list(STRATEGY_MAP.keys())}"
        )
    
    chunk_fn = STRATEGY_MAP[strategy]
    chunks = chunk_fn(
        text=text,
        source_doc=source_doc,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    logger.info(
        f"Chunked '{source_doc}' with strategy='{strategy}': "
        f"{len(chunks)} chunks (size={chunk_size or CHUNK_SIZE}, "
        f"overlap={chunk_overlap or CHUNK_OVERLAP})"
    )
    
    return chunks


def chunk_all_documents(
    documents: List[Dict],
    strategy: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Chunk all documents in the corpus using the specified strategy.
    
    Args:
        documents: List of document dictionaries from data_loader
        strategy: Chunking strategy to use
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Combined list of Chunk objects from all documents
    """
    all_chunks = []
    
    for doc in documents:
        source_name = doc["metadata"].get("title", doc["metadata"]["source_file"])
        chunks = chunk_document(
            text=doc["content"],
            source_doc=source_name,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    
    logger.info(
        f"Total chunks from {len(documents)} documents: {len(all_chunks)}"
    )
    
    return all_chunks


def get_chunk_stats(chunks: List[Chunk]) -> Dict:
    """
    Compute statistics for a list of chunks.
    
    Useful for comparing different chunking strategies.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {"num_chunks": 0}
    
    char_counts = [c.char_count for c in chunks]
    word_counts = [c.word_count for c in chunks]
    
    return {
        "num_chunks": len(chunks),
        "strategy": chunks[0].strategy,
        "chunk_size_config": chunks[0].chunk_size_config,
        "overlap_config": chunks[0].overlap_config,
        "avg_chars": sum(char_counts) / len(char_counts),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "median_chars": sorted(char_counts)[len(char_counts) // 2],
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "total_chars": sum(char_counts),
        "source_docs": list(set(c.source_doc for c in chunks)),
    }
