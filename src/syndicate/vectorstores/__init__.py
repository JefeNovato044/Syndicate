"""Vector stores module for semantic search.

This module provides vector store implementations for storing and
retrieving documents using semantic similarity search.

IMPORTANT: All vector stores require an EmbeddingModel instance in
their constructor to generate embeddings for query texts at runtime.

Example:
    ```python
    from syndicate.ingestion import SentenceTransformerEmbedding
    from syndicate.vectorstores import MongoVectorStore
    
    # Create embedding model
    embedding_model = SentenceTransformerEmbedding()
    
    # Create vector store
    vector_store = MongoVectorStore(
        connection_string="mongodb+srv://...",
        database="mydb",
        collection="vectors",
        embedding_model=embedding_model,
        dims=384
    )
    
    # Add documents
    await vector_store.add_texts(["Document text"])
    
    # Search
    results = await vector_store.search("Query text")
    ```
"""

from .base import (
    BaseVectorStore,
    reciprocal_rank_fusion
)

from .mongo import (
    MongoVectorStore
)

__all__ = [
    # Base classes
    "BaseVectorStore",
    
    # Implementations
    "MongoVectorStore",
    
    # Utilities
    "reciprocal_rank_fusion"
]