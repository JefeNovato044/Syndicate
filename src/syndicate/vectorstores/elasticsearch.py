"""Elasticsearch 9 vector store implementation.

Provides hybrid search (dense vector kNN + BM25 full-text) using native
Elasticsearch 9 Reciprocal Rank Fusion (RRF), fully compatible with the
free tier of Elasticsearch 9.

All features used are available without paid plugins or ML nodes:
    - ``dense_vector`` with ``index: true`` for approximate kNN (since ES 8.0)
    - ``match`` queries for BM25 full-text search (always free)
    - Native ``rank.rrf`` for hybrid rank fusion (since ES 8.8, all tiers)

Requirements:
    - elasticsearch[async]>=9.0.0
    - An Elasticsearch 9 cluster (self-hosted or Elastic Cloud free tier)

Example:
    ```python
    from elasticsearch import AsyncElasticsearch
    from syndicate.ingestion import SentenceTransformerEmbedding
    from syndicate.vectorstores import ElasticsearchVectorStore

    es_client = AsyncElasticsearch(["http://localhost:9200"])
    embedding_model = SentenceTransformerEmbedding()

    store = ElasticsearchVectorStore(
        es_client=es_client,
        index_name="my_vectors",
        embedding_model=embedding_model,
        auto_setup=True,
    )

    doc_ids = await store.add_texts(["Hello world"])
    results = await store.search("Hello")
    await store.close()
    ```
"""

import logging
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk

from .base import BaseVectorStore
from ..ingestion.embedding_models import EmbeddingModel, EmbeddingMode


logger = logging.getLogger(__name__)


class ElasticsearchVectorStore(BaseVectorStore):
    """Elasticsearch 9 vector store with hybrid search support.

    Uses ``dense_vector`` kNN + BM25 ``match`` queries merged with native ES 9
    RRF rank fusion. All features are available on the free/basic tier of
    Elasticsearch 9 — no ML node or paid licence required.

    Features:
        - Approximate kNN vector similarity search (cosine)
        - BM25 full-text search
        - Hybrid search via ES 9 native ``rank.rrf`` fusion
        - Metadata filtering via pre-filter on the kNN clause
        - Auto index creation with proper field mapping

    Index mapping (auto-created when ``auto_setup=True`` or via
    :meth:`ensure_index_ready`):

    .. code-block:: json

        {
            "mappings": {
                "properties": {
                    "text":      {"type": "text", "analyzer": "standard"},
                    "embedding": {"type": "dense_vector", "dims": N,
                                  "index": true, "similarity": "cosine"},
                    "metadata":  {"type": "object", "dynamic": true}
                }
            }
        }
    """

    def __init__(
        self,
        es_client: AsyncElasticsearch,
        index_name: str,
        embedding_model: EmbeddingModel,
        dims: Optional[int] = None,
        auto_setup: bool = False,
    ):
        """
        Args:
            es_client: Pre-configured async Elasticsearch client.
            index_name: Name of the Elasticsearch index to store documents in.
            embedding_model: Embedding model used to generate vectors at
                             both index and query time.
            dims: Optional vector dimension override. Defaults to the
                  effective dimension of ``embedding_model``.
            auto_setup: When ``True``, automatically creates the index (if
                        missing) before the first read or write operation.
                        Set to ``False`` when you manage the index lifecycle
                        externally or via :meth:`ensure_index_ready`.
        """
        super().__init__(embedding_model=embedding_model)
        self.es_client = es_client
        self.index_name = index_name
        self.auto_setup = auto_setup

        self._index_ready: bool = False
        self._backend_validated: bool = False

        self._resolve_effective_dimension(dims)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    async def ensure_index_ready(self) -> None:
        """Ensure the index exists with the correct mapping.

        Idempotent — safe to call multiple times. Creates the index when it
        does not exist and validates the stored dimension against the model.
        """
        if self._index_ready:
            return
        await self._create_index_if_missing()
        if not self._backend_validated:
            await self.validate_backend_configuration()
            self._backend_validated = True
        self._index_ready = True

    async def _create_index_if_missing(self) -> None:
        """Create the ES index with the vector + text mapping if absent."""
        exists = await self.es_client.indices.exists(index=self.index_name)
        if exists:
            logger.info("Elasticsearch index already exists: %s", self.index_name)
            return

        mapping = self._build_index_mapping()
        await self.es_client.indices.create(index=self.index_name, body=mapping)
        logger.info(
            "Created Elasticsearch index '%s' with %d-dimensional vectors",
            self.index_name,
            self.effective_dimension,
        )

    def _build_index_mapping(self) -> Dict[str, Any]:
        """Return the ES index mapping for vector + text + metadata storage."""
        return {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.effective_dimension,
                        "index": True,
                        "similarity": "cosine",
                        "element_type": "float",
                    },
                    "metadata": {
                        "type": "object",
                        "dynamic": True,
                    },
                }
            }
        }

    async def validate_backend_configuration(self) -> None:
        """Validate that the index mapping dimension matches the store contract.

        Logs a warning when the index is not found; raises :exc:`ValueError`
        on a dimension mismatch to prevent silent misaligned retrievals.
        """
        try:
            response = await self.es_client.indices.get_mapping(index=self.index_name)
            index_mapping = response.get(self.index_name, {})
            props = (
                index_mapping
                .get("mappings", {})
                .get("properties", {})
            )
            embedding_field = props.get("embedding", {})
            index_dims = embedding_field.get("dims")
        except NotFoundError:
            logger.info(
                "Elasticsearch index '%s' not found during validation; skipping.",
                self.index_name,
            )
            return
        except Exception as exc:
            logger.warning("Elasticsearch index validation failed: %s", exc)
            return

        if index_dims is None:
            logger.info(
                "Could not extract dims from index '%s' mapping during validation.",
                self.index_name,
            )
            return

        if int(index_dims) != int(self.effective_dimension):
            raise ValueError(
                "Elasticsearch index dimension mismatch: "
                f"index '{self.index_name}' has {index_dims} dims, "
                f"store effective dimension is {self.effective_dimension}."
            )

    def _ensure_embedding_dimension_alignment(self) -> None:
        """Guard against external drift of a shared embedding model instance."""
        if self.effective_dimension is None:
            self._refresh_embedding_contract()

        target = int(self.effective_dimension)
        current = int(self.embedding_model.embedding_dimension)

        if current == target:
            return

        if not self.embedding_model.supports_dimension_override:
            raise ValueError(
                "Embedding model dimension drift detected and cannot be corrected: "
                f"store expects {target}, model currently has {current}."
            )

        self.embedding_model.configure_dimension(target, source="vector_store")
        self._refresh_embedding_contract()
        logger.warning(
            "Re-aligned embedding model dimension for ElasticsearchVectorStore '%s': %d -> %d",
            self.index_name,
            current,
            target,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add text documents and generate their embeddings automatically.

        Args:
            texts: Text strings to index.
            metadatas: Optional metadata dicts (one per text). Stored under
                       the ``metadata`` field and filterable via
                       ``metadata.<key>`` terms.
            ids: Optional document IDs. Auto-generated UUIDs when omitted.

        Returns:
            List of document IDs that were indexed.
        """
        if self.auto_setup:
            await self.ensure_index_ready()

        self._ensure_embedding_dimension_alignment()
        self._validate_inputs(texts, metadatas, ids)

        if ids is None:
            ids = self._generate_ids(len(texts))

        embeddings = await self.embedding_model.embed_batch(
            texts,
            mode=EmbeddingMode.DOCUMENT.value,
        )
        self._validate_document_embeddings(embeddings)

        actions = [
            {
                "_index": self.index_name,
                "_id": ids[i],
                "_source": {
                    "text": texts[i],
                    "embedding": embeddings[i],
                    "metadata": (metadatas[i] if metadatas else {}) or {},
                },
            }
            for i in range(len(texts))
        ]

        await async_bulk(self.es_client, actions, refresh="wait_for")
        return ids

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents using the dict-based interface.

        Each document must have a ``content`` key. Optional ``metadata``
        (dict) and ``id`` (str) keys are also accepted.

        Args:
            documents: List of dicts, each with at minimum ``{"content": "…"}``.

        Returns:
            List of document IDs that were indexed.
        """
        if not documents:
            raise ValueError("documents cannot be empty")

        texts = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata") or {} for doc in documents]

        # Only use provided IDs when every document supplies one
        raw_ids = [doc.get("id") for doc in documents]
        ids: Optional[List[str]] = (
            [str(i) for i in raw_ids]  # type: ignore[misc]
            if all(raw_ids)
            else None
        )

        return await self.add_texts(texts, metadatas, ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Query text — auto-embedded before the search.
            k: Number of top results to return.
            filter: Optional flat metadata filter dict applied to
                    ``metadata.<key>`` fields.
                    Example: ``{"category": "tech", "year": 2024}``
            use_hybrid: When ``True`` (default), performs a hybrid kNN + BM25
                        search merged with native ES RRF. Falls back to
                        pure kNN when ``False``.

        Returns:
            List of result dicts sorted by relevance::

                [
                    {"id": "…", "text": "…", "metadata": {…}, "score": 0.95},
                    …
                ]
        """
        if self.auto_setup:
            await self.ensure_index_ready()

        self._ensure_embedding_dimension_alignment()

        query_embedding = await self.embedding_model.embed(
            query,
            mode=EmbeddingMode.QUERY.value,
        )
        self._validate_query_embedding(query_embedding)

        if use_hybrid:
            return await self._hybrid_search(query_embedding, query, k, filter)
        return await self._vector_search(query_embedding, k, filter)

    async def _vector_search(
        self,
        query_embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Approximate kNN search using the ``dense_vector`` field."""
        self._validate_query_embedding(query_embedding)

        knn_clause: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": max(k * 10, 100),
        }
        if filter:
            knn_clause["filter"] = self._build_metadata_filter(filter)

        response = await self.es_client.search(
            index=self.index_name,
            body={
                "knn": knn_clause,
                "size": k,
                "_source": ["text", "metadata"],
            },
        )
        return self._format_hits(response["hits"]["hits"])

    async def _hybrid_search(
        self,
        query_embedding: List[float],
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid kNN + BM25 search using native ES 9 RRF rank fusion.

        ``rank.rrf`` is available on all Elasticsearch tiers (free included)
        since version 8.8 and produces better results than either kNN or BM25
        alone by rewarding documents that rank highly in both result sets.
        """
        self._validate_query_embedding(query_embedding)

        knn_clause: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": max(k * 10, 100),
        }
        if filter:
            knn_clause["filter"] = self._build_metadata_filter(filter)

        # BM25 clause — wrap in bool+filter when metadata filter is present
        text_query: Dict[str, Any] = {"match": {"text": {"query": query}}}
        if filter:
            text_query = {
                "bool": {
                    "must": text_query,
                    "filter": self._build_metadata_filter(filter),
                }
            }

        body: Dict[str, Any] = {
            "knn": knn_clause,
            "query": text_query,
            "rank": {"rrf": {}},
            "size": k,
            "_source": ["text", "metadata"],
        }

        try:
            response = await self.es_client.search(index=self.index_name, body=body)
            return self._format_hits(response["hits"]["hits"])
        except Exception as exc:
            logger.warning(
                "Hybrid search failed for index '%s', falling back to kNN-only: %s",
                self.index_name,
                exc,
            )
            return await self._vector_search(query_embedding, k, filter)

    async def delete(self, ids: Optional[List[str]] = None) -> int:
        """Delete documents by IDs, or purge the entire index when ``ids`` is ``None``.

        Args:
            ids: Document IDs to delete. Deletes **all** documents when ``None``.

        Returns:
            Number of documents deleted.
        """
        if ids is None:
            response = await self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                refresh=True,
            )
            return int(response.get("deleted", 0))

        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
            for doc_id in ids
        ]
        success, _ = await async_bulk(
            self.es_client,
            actions,
            refresh=True,
            raise_on_error=False,
        )
        return int(success)

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve documents by their IDs using the ``mget`` API.

        Args:
            ids: Document IDs to fetch.

        Returns:
            List of dicts ``{"id", "text", "metadata"}`` for found documents.
            Missing IDs are silently omitted.
        """
        if not ids:
            return []

        response = await self.es_client.mget(
            index=self.index_name,
            body={"ids": ids},
        )
        results: List[Dict[str, Any]] = []
        for doc in response["docs"]:
            if not doc.get("found"):
                continue
            source = doc.get("_source", {})
            results.append({
                "id": doc["_id"],
                "text": source.get("text", ""),
                "metadata": source.get("metadata", {}) or {},
            })
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metadata_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a flat metadata dict into an ES query clause.

        Each key is matched as a term against ``metadata.<key>``.

        Examples::

            {"category": "tech"}
            → {"term": {"metadata.category": "tech"}}

            {"category": "tech", "year": 2024}
            → {"bool": {"filter": [
                {"term": {"metadata.category": "tech"}},
                {"term": {"metadata.year": 2024}}
              ]}}
        """
        if not filter:
            return {"match_all": {}}

        clauses = [
            {"term": {f"metadata.{key}": value}}
            for key, value in filter.items()
        ]
        if len(clauses) == 1:
            return clauses[0]
        return {"bool": {"filter": clauses}}

    def _format_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalise raw ES hits into the standard result format."""
        results: List[Dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {})
            doc: Dict[str, Any] = {
                "id": hit["_id"],
                "text": source.get("text", ""),
                "metadata": source.get("metadata", {}) or {},
            }
            score = hit.get("_score")
            if score is not None:
                doc["score"] = float(score)
            results.append(doc)
        return results

    async def close(self) -> None:
        """Close the Elasticsearch client and embedding model resources."""
        try:
            await self.es_client.close()
        except Exception as exc:
            logger.warning("Error closing Elasticsearch client: %s", exc)

        try:
            await self.embedding_model.close()
        except Exception as exc:
            logger.warning("Error closing embedding model: %s", exc)