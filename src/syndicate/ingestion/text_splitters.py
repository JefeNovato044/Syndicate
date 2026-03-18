"""Text splitting utilities for document chunking.

This module provides text splitter implementations for breaking down
large documents into smaller chunks suitable for embedding and storage.

IMPORTANT: This module ONLY contains text splitting logic. Document
loading is intentionally NOT included to maintain a clear separation
of concerns between ingestion (external) and retrieval (internal).
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import re


class BaseTextSplitter(ABC):
    """Abstract base class for text splitters.
    
    Text splitters break down large documents into smaller chunks
    based on various strategies (character count, tokens, semantic
    boundaries, etc.).
    """
    
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        length_function: Optional[callable] = None
    ):
        """
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            length_function: Function to calculate length of text.
                           Defaults to len() for character count.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        pass
    
    def create_chunks(self, texts: List[str]) -> List[str]:
        """Split multiple texts into chunks.
        
        Args:
            texts: List of input texts
            
        Returns:
            Flattened list of all chunks from all texts
        """
        all_chunks = []
        for text in texts:
            chunks = self.split_text(text)
            all_chunks.extend(chunks)
        return all_chunks


class CharacterTextSplitter(BaseTextSplitter):
    """Split text by character count with separator awareness.
    
    This splitter attempts to break text at natural boundaries
    (separators) while respecting chunk size limits.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            separators: List of separator strings to try, in order.
                       Defaults to ["\\n\\n", "\\n", " ", ""]
            **kwargs: Additional arguments for BaseTextSplitter
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text by characters, respecting separators.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        current_separator = self.separators[0]
        
        # Try each separator until we get splits or run out
        for separator in self.separators:
            if not separator:
                # Empty separator means split by exact character count
                chunks = self._split_by_character(text)
                break
            
            # Split by current separator
            splits = text.split(separator)
            
            # If split produced more than one part, use this separator
            if len(splits) > 1:
                chunks = self._merge_splits(splits, separator)
                break
        
        return chunks
    
    def _split_by_character(self, text: str) -> List[str]:
        """Split text by exact character count.
        
        Args:
            text: Input text
            
        Returns:
            List of chunks with exact character sizes
        """
        chunks = []
        start = 0
        text_length = self.length_function(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits that are smaller than chunk_size.
        
        Args:
            splits: List of text splits from separator
            separator: Separator string used for splitting
            
        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            separator_length = self.length_function(separator)
            
            # Check if adding this split would exceed chunk_size
            if current_length + split_length + separator_length > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    chunks.append(chunk_text)
                
                # Start new chunk
                # If current split is larger than chunk_size, split it by character
                if split_length > self.chunk_size:
                    # Handle oversized split
                    sub_chunks = self._split_by_character(split)
                    if sub_chunks:
                        current_chunk = [sub_chunks[0]]
                        current_length = self.length_function(sub_chunks[0])
                        # Add remaining sub-chunks
                        for sub_chunk in sub_chunks[1:]:
                            chunks.append(sub_chunk)
                else:
                    current_chunk = [split]
                    current_length = split_length
            else:
                # Add split to current chunk
                current_chunk.append(split)
                current_length += split_length + separator_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """Recursively split text by separators until chunks fit within size limit.
    
    This splitter tries to split by larger separators first (paragraphs),
    then falls back to smaller ones (sentences, words) if needed.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            separators: List of separator strings to try, in order of priority.
                       Defaults to ["\\n\\n", "\\n", "\\n ", ". ", " ", ""]
            **kwargs: Additional arguments for BaseTextSplitter
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separators = separators or ["\n\n", "\n", "\n ", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Recursively split text by separators.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        return self._split_text_recursively(text)
    
    def _split_text_recursively(self, text: str) -> List[str]:
        """Recursively split text using separators.
        
        Args:
            text: Input text
            
        Returns:
            List of chunks
        """
        # Find the next separator to use
        next_separator = self.separators[-1] if not self.separators else ""
        separator_index = 0
        
        for i, separator in enumerate(self.separators):
            if separator in text:
                next_separator = separator
                separator_index = i
                break
        
        # Split by the chosen separator
        splits = text.split(next_separator)
        
        # Check if all splits are within chunk_size
        good_splits = []
        for split in splits:
            if self.length_function(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # Recursively split oversized chunks
                # Use remaining separators (excluding current one)
                remaining_separators = self.separators[separator_index + 1:]
                if remaining_separators:
                    temp_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=remaining_separators,
                        length_function=self.length_function
                    )
                    good_splits.extend(temp_splitter._split_text_recursively(split))
                else:
                    # No more separators, split by character
                    good_splits.extend(self._split_by_character(split))
        
        # Merge splits with overlap
        return self._join_splits_with_overlap(good_splits, next_separator)
    
    def _split_by_character(self, text: str) -> List[str]:
        """Split text by exact character count.
        
        Args:
            text: Input text
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        text_length = self.length_function(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _join_splits_with_overlap(self, splits: List[str], separator: str) -> List[str]:
        """Join splits back together, respecting chunk_size and overlap.
        
        Args:
            splits: List of text splits
            separator: Separator to use when joining
            
        Returns:
            List of joined chunks with overlap
        """
        if not splits:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        separator_length = self.length_function(separator) if separator else 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # Check if adding this split would exceed chunk_size
            if current_length + split_length + separator_length > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if chunks:
                    # Get overlap from previous chunk
                    last_chunk = chunks[-1]
                    overlap_text = last_chunk[-self.chunk_overlap:] if self.chunk_overlap else ""
                    current_chunk = [overlap_text]
                    current_length = self.length_function(overlap_text)
                
                # Add current split if it fits
                if current_length + split_length <= self.chunk_size:
                    current_chunk.append(split)
                    current_length += split_length + separator_length
                else:
                    # Split is too big, add it alone
                    chunks.append(split)
                    current_chunk = []
                    current_length = 0
            else:
                # Add split to current chunk
                current_chunk.append(split)
                current_length += split_length + separator_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Split Markdown documents respecting structure.
    
    This splitter is optimized for Markdown documents, splitting
    at logical boundaries like headers, code blocks, and lists.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            **kwargs: Additional arguments for RecursiveCharacterTextSplitter
        """
        # Markdown-specific separators in order of priority
        separators = [
            "\n\n## ",  # Level 2 header
            "\n\n### ",  # Level 3 header
            "\n\n#### ",  # Level 4 header
            "\n\n---",  # Horizontal rule
            "\n\n---\n",
            "\n\n> ",  # Blockquote
            "\n\n```",  # Code block
            "\n\n- ",  # List item
            "\n\n* ",  # List item
            "\n\n\\d+\\. ",  # Numbered list
            "\n\n",  # Paragraph
            "\n",  # Line break
            " ",  # Word
            ""
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )


class CodeTextSplitter(RecursiveCharacterTextSplitter):
    """Split code documents respecting structure.
    
    This splitter is optimized for code, attempting to split at
    function definitions, class definitions, and other logical
    boundaries.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            **kwargs: Additional arguments for RecursiveCharacterTextSplitter
        """
        # Code-specific separators
        separators = [
            "\n\nclass ",  # Class definition
            "\n\ndef ",  # Function definition
            "\n\nasync def ",  # Async function
            "\n\nfunction ",  # JS function
            "\n\nconst ",  # JS const
            "\n\nlet ",  # JS let
            "\n\nvar ",  # JS var
            "\n\n",  # Blank line
            "\n",  # Newline
            " ",  # Space
            ""
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )