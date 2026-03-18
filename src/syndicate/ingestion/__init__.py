"""Ingestion module for document processing.

This module provides utilities for preparing documents for storage
in vector stores. It includes text splitters for chunking and
embedding models for vector generation.

IMPORTANT: This module ONLY handles document preparation (splitting
and embedding). It does NOT include document loaders - those should
be implemented externally as part of your data ingestion pipeline.

Example:
    ```python
    from syndicate.ingestion import (
        RecursiveCharacterTextSplitter,
        SentenceTransformerEmbedding
    )
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text("Your document text here...")
    
    # Generate embeddings
    embedding_model = SentenceTransformerEmbedding()
    embeddings = await embedding_model.embed_batch(chunks)
    ```
"""

from .text_splitters import (
    BaseTextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    CodeTextSplitter
)

from .embedding_models import (
    EmbeddingModel,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    GeminiEmbedding,
    CohereEmbedding
)

__all__ = [
    # Text Splitters
    "BaseTextSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "MarkdownTextSplitter",
    "CodeTextSplitter",
    
    # Embedding Models
    "EmbeddingModel",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "CohereEmbedding"
]