"""Base vector store interface.

This module defines the abstract interface for vector stores in the
Syndicate framework. All vector store implementations must inherit
from BaseVectorStore.

IMPORTANT: Vector stores require an EmbeddingModel instance in their
constructor to generate embeddings for query texts at runtime. This
ensures that the same embedding model is used for both indexing and
retrieval, which is critical for accurate similarity search.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Sequence
import uuid
import warnings

from ..ingestion.embedding_models import EmbeddingModel, EmbeddingMode


logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.
    
    Vector stores provide semantic search capabilities by storing
    text chunks along with their vector embeddings. This base class
    defines the interface that all vector store implementations must
    follow.
    
    IMPORTANT: The embedding_model is required in the constructor
    because it's needed at runtime to embed query texts during search
    operations. This ensures consistency between indexing and retrieval.
    
    Example:
        ```python
        from syndicate.ingestion import SentenceTransformerEmbedding
        from syndicate.vectorstores import SomeVectorStore
        
        # Create embedding model
        embedding_model = SentenceTransformerEmbedding()
        
        # Create vector store with embedding model
        vector_store = SomeVectorStore(
            embedding_model=embedding_model,
            # other parameters...
        )
        
        # Add documents (embeddings are auto-generated)
        doc_ids = await vector_store.add_texts([
            "First document text",
            "Second document text"
        ])
        
        # Search (query is auto-embedded)
        results = await vector_store.search("What is the capital of France?")
        ```
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Args:
            embedding_model: Embedding model to use for generating
                           vector representations. REQUIRED - cannot be None.
        """
        if embedding_model is None:
            raise ValueError("embedding_model is required")
        
        self.embedding_model = embedding_model
        self.effective_dimension: Optional[int] = None
        self.dimension_source: str = "unknown"
        self.embedding_space_id: str = "unknown"
        self.embedding_capabilities: Dict[str, Any] = {}
        self._refresh_embedding_contract()
        self._validate_required_modes()
    
    @abstractmethod
    async def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        The query text is automatically embedded using the embedding_model
        provided in the constructor.
        
        Args:
            query: Search query text (will be auto-embedded)
            k: Number of results to return
            filter: Optional metadata filter dict
                   Example: {"category": "technical", "author": "john"}
            use_hybrid: If True and supported, use hybrid search
                       (semantic + keyword). Defaults to True.
        
        Returns:
            List of native Python dictionaries with the following structure:
            [
                {
                    "id": "doc_id",
                    "text": "document text",
                    "metadata": {"key": "value"},
                    "score": 0.95
                },
                ...
            ]
            
            IMPORTANT: Returns native dict objects, NOT wrapper classes.
        """
        pass
    
    @abstractmethod
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        The vector store automatically generates embeddings using the
        embedding_model provided in the constructor. Developers do NOT
        need to pre-compute embeddings.
        
        Args:
            texts: List of text chunks to store
            metadatas: Optional list of metadata dicts (one per text)
                      Example: [{"source": "file1.pdf"}, {"source": "file2.pdf"}]
                      - Do NOT include 'embedding' key - it will be auto-generated
            ids: Optional list of document IDs (auto-generated if None)
        
        Returns:
            List of document IDs (either provided or auto-generated)
        
        Example:
            ```python
            # Add with auto-generated IDs
            doc_ids = await vector_store.add_texts([
                "First document",
                "Second document"
            ])
            
            # Add with custom IDs and metadata
            doc_ids = await vector_store.add_texts(
                texts=["Document text"],
                metadatas=[{"source": "file.pdf", "page": 1}],
                ids=["custom-doc-id"]
            )
            ```
        """
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Will replace add_texts in the future - accepts list of dicts with 'content',
        Add documents to the vector store using a list of dicts.
        
        Each document dict should have at least a 'content' key for the text,
        and can optionally include 'metadata' and 'id' keys. The vector store
        will automatically generate embeddings from the 'content' field.
        
        Args:
            documents: List of document dicts with the following structure:
                       [
                           {
                               "content": "document text",  # REQUIRED
                               "metadata": {"key": "value"},  # Optional
                               "id": "custom-doc-id"  # Optional
                           },
                           ...
            """
        pass         
    
    @abstractmethod
    async def delete(self, ids: Optional[List[str]] = None) -> int:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete.
                If None, deletes all documents.
        
        Returns:
            Number of documents deleted
        """
        pass
    
    @abstractmethod
    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve
        
        Returns:
            List of native Python dictionaries:
            [
                {
                    "id": "doc_id",
                    "text": "document text",
                    "metadata": {"key": "value"}
                },
                ...
            ]
        """
        pass
    
    async def clear(self) -> int:
        """
        Remove all documents from the vector store.
        
        Returns:
            Number of documents deleted
        """
        return await self.delete()
    
    def _generate_ids(self, count: int) -> List[str]:
        """Generate unique IDs for documents.
        
        Args:
            count: Number of IDs to generate
        
        Returns:
            List of unique UUID strings
        """
        return [str(uuid.uuid4()) for _ in range(count)]
    
    def _validate_inputs(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        ids: Optional[List[str]]
    ) -> None:
        """Validate input parameters for add_texts.
        
        Args:
            texts: List of texts to validate
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match "
                f"texts length ({len(texts)})"
            )
        
        if ids is not None and len(ids) != len(texts):
            raise ValueError(
                f"ids length ({len(ids)}) must match texts length ({len(texts)})"
            )
        
        if ids is not None and len(ids) != len(set(ids)):
            raise ValueError("ids must be unique")

    async def validate_backend_configuration(self) -> None:
        """Best-effort backend configuration validation hook.

        Concrete stores can override this to validate index configuration,
        dimensions, or provider-specific readiness checks.
        """

    def _refresh_embedding_contract(self) -> None:
        """Refresh cached embedding metadata and capabilities."""
        model_info = self.embedding_model.get_model_info()
        capabilities = self.embedding_model.get_capabilities()

        self.effective_dimension = int(model_info["effective_dimension"])
        self.dimension_source = str(model_info["dimension_source"])
        self.embedding_space_id = str(model_info["embedding_space_id"])
        self.embedding_capabilities = dict(capabilities)

    def _resolve_effective_dimension(self, dims: Optional[int]) -> int:
        """Resolve and enforce store effective dimension.

        If dims is provided and differs from model effective dimension,
        the embedding model is reconfigured only when override is supported.
        """
        self._refresh_embedding_contract()
        model_effective_dimension = self.effective_dimension

        if dims is None:
            return int(model_effective_dimension)

        if dims <= 0:
            raise ValueError("dims must be a positive integer")

        requested_dimension = int(dims)
        if requested_dimension == model_effective_dimension:
            return requested_dimension

        if not self.embedding_model.supports_dimension_override:
            raise ValueError(
                "Store dims override requested but model has fixed dimension "
                f"{model_effective_dimension}; cannot configure {requested_dimension}."
            )

        self.embedding_model.configure_dimension(requested_dimension, source="vector_store")
        self._refresh_embedding_contract()
        self._warn_dimension_override(
            model_dims=model_effective_dimension,
            store_dims=requested_dimension,
            effective_dims=self.effective_dimension,
        )
        return int(self.effective_dimension)

    def _validate_required_modes(self) -> None:
        """Ensure query and document modes are available for core operations."""
        required_modes = [EmbeddingMode.QUERY, EmbeddingMode.DOCUMENT]
        missing = [mode.value for mode in required_modes if not self.embedding_model.supports_mode(mode)]
        if missing:
            raise ValueError(
                f"Embedding model {self.embedding_model.model_name} is missing required modes: {missing}."
            )

    def _validate_query_embedding(self, vector: Sequence[float]) -> None:
        """Validate query vector length against resolved store dimension."""
        if self.effective_dimension is None:
            raise ValueError("effective_dimension is not initialized")

        actual = len(vector)
        if actual != self.effective_dimension:
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"expected {self.effective_dimension}, got {actual}."
            )

    def _validate_document_embeddings(self, vectors: Sequence[Sequence[float]]) -> None:
        """Validate all document embedding vector lengths."""
        for idx, vector in enumerate(vectors):
            try:
                self._validate_query_embedding(vector)
            except ValueError as exc:
                raise ValueError(f"Document embedding dimension mismatch at index {idx}: {exc}") from exc

    def _warn_dimension_override(
        self,
        model_dims: int,
        store_dims: int,
        effective_dims: int,
    ) -> None:
        """Emit warning and log when store configuration overrides model dims."""
        message = (
            "VectorStore dims override detected: store dims "
            f"{store_dims} overrides model dims {model_dims}. "
            f"Effective dims={effective_dims}."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    
    RRF is an algorithm for combining multiple ranked lists into a single
    ranked list. It's particularly useful for merging results from different
    search algorithms (e.g., vector search + keyword search) that may have
    different scoring scales.
    
    The RRF score for a document is calculated as:
        score = sum(1 / (k + rank_i) for each list i where document appears)
    
    Where:
        - k is a smoothing constant (typically 60)
        - rank_i is the 1-based rank of the document in list i
    
    Args:
        results_list: List of ranked result lists. Each inner list should
                     contain dicts with at least an 'id' key.
        k: Smoothing constant. Higher values give more weight to lower-ranked
           documents. Default is 60 (common choice).
    
    Returns:
        Merged and ranked list of result dictionaries with RRF scores.
        Results are sorted by RRF score in descending order (best first).
    
    Example:
        ```python
        vector_results = [
            {"id": "doc1", "score": 0.95},
            {"id": "doc2", "score": 0.87}
        ]
        keyword_results = [
            {"id": "doc2", "score": 10.5},
            {"id": "doc3", "score": 8.2}
        ]
        
        merged = reciprocal_rank_fusion([vector_results, keyword_results])
        # doc2 will have highest RRF score (appears in both lists)
        ```
    
    Reference:
        Cormack, G. V., et al. "Reciprocal Rank Fusion Outperforms
        Relevance Feedback in Ad Hoc Retrieval." ECIR 2009.
    """
    # Dictionary to accumulate RRF scores for each document
    rrf_scores: Dict[str, float] = {}
    
    # Track all documents and their best original data
    all_docs: Dict[str, Dict[str, Any]] = {}
    
    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id")
            if not doc_id:
                continue
            
            # Calculate RRF score contribution from this list
            rrf_score = 1.0 / (k + rank)
            
            # Accumulate score
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += rrf_score
            else:
                rrf_scores[doc_id] = rrf_score
                all_docs[doc_id] = doc.copy()
    
    # Sort by RRF score (descending) and create result list
    sorted_docs = sorted(
        all_docs.items(),
        key=lambda x: rrf_scores[x[0]],
        reverse=True
    )
    
    # Build final result with RRF scores
    result = []
    for doc_id, doc_data in sorted_docs:
        result_doc = doc_data.copy()
        result_doc["rrf_score"] = rrf_scores[doc_id]
        result.append(result_doc)
    
    return result