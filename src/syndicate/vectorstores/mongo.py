"""MongoDB Atlas vector store implementation.

This module provides a vector store implementation using MongoDB Atlas
with support for hybrid search (vector + keyword) and Reciprocal Rank
Fusion (RRF) for result merging.

IMPORTANT: This implementation requires MongoDB Atlas (not self-hosted
MongoDB) because it uses the $vectorSearch and $search aggregation stages
which are Atlas-only features.

Requirements:
    - pymongo>=4.16.0 (for AsyncMongoClient and Atlas search index APIs)
    - MongoDB Atlas cluster with vector search enabled

Example:
    ```python
    from syndicate.ingestion import SentenceTransformerEmbedding
    from syndicate.vectorstores import MongoVectorStore
    
    # Create embedding model
    embedding_model = SentenceTransformerEmbedding()
    
    # Create vector store
    vector_store = MongoVectorStore(
        connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
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

from typing import Any, Dict, List, Optional
import logging

from pymongo import AsyncMongoClient
from pymongo.errors import CollectionInvalid, OperationFailure
from pymongo.operations import SearchIndexModel

from .base import BaseVectorStore
from ..ingestion.embedding_models import EmbeddingModel, EmbeddingMode

logger = logging.getLogger(__name__)


class MongoVectorStore(BaseVectorStore):
    """MongoDB Atlas vector store with hybrid search support.
    
    This implementation uses MongoDB Atlas's vector search capabilities
    combined with BM25 keyword search, merging results using Reciprocal
    Rank Fusion (RRF) for optimal retrieval quality.
    
    Features:
        - Vector similarity search using cosine similarity
        - Keyword search using BM25 algorithm
        - Hybrid search combining both methods with RRF
        - Metadata filtering
        - Async operations using native pymongo AsyncMongoClient
    
    Atlas Requirements:
        - MongoDB Atlas cluster (free tier works)
        - Vector search index configured
        - Atlas Search index configured (for hybrid search)
    
    Index Setup:
        Before using this vector store, you must create the required
        indexes in MongoDB Atlas:
        
        1. Vector Search Index (via Atlas UI or MongoDB CLI):
        ```json
        {
            "fields": [
                {
                    "numDimensions": 384,
                    "path": "embedding",
                    "similarity": "cosine",
                    "type": "vector"
                }
            ]
        }
        ```
        
        2. Atlas Search Index (for keyword search):
        ```json
        {
            "mappings": {
                "dynamic": false,
                "fields": {
                    "text": {
                        "type": "string"
                    }
                }
            }
        }
        ```
    """
    
    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        embedding_model: EmbeddingModel,
        dims: Optional[int] = None,
        index_name: str = "vector_index",
        search_index_name: str = "text_index",
        auto_setup: bool = False,
    ):
        """
        Args:
            connection_string: MongoDB Atlas connection string
                              Example: "mongodb+srv://user:pass@cluster.mongodb.net/"
            database: Database name
            collection: Collection name for storing vectors
            embedding_model: Embedding model for generating vectors
            dims: Optional vector dimension override. If None, uses the
                  embedding model effective dimension.
            index_name: Name of the vector search index in Atlas
            search_index_name: Name of the Atlas Search index for keyword search
            auto_setup: If True, automatically attempts backend provisioning
                        (collection + required search indexes) before reads/writes.
        """
        super().__init__(embedding_model=embedding_model)
        
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self.index_name = index_name
        self.search_index_name = search_index_name
        self.auto_setup = auto_setup
        self.requested_dimension = dims
        
        self._client: Optional[AsyncMongoClient] = None
        self._collection = None
        self._collection_ready = False
        self._search_indexes_ready = False
        self._backend_validated = False
        self.model_info: Dict[str, Any] = {}

        self._resolve_effective_dimension(dims)
        self.model_info = self.embedding_model.get_model_info()
    
    def _get_client(self) -> AsyncMongoClient:
        """Get or create MongoDB client.
        
        Returns:
            AsyncMongoClient instance
        """
        if self._client is None:
            self._client = AsyncMongoClient(
                self.connection_string,
                connectTimeoutMS=5000,
                serverSelectionTimeoutMS=5000
            )
        return self._client
    
    def _get_collection(self):
        """Get collection reference.
        
        Returns:
            MongoDB collection object
        """
        if self._collection is None:
            client = self._get_client()
            db = client[self.database]
            self._collection = db[self.collection]
        return self._collection

    async def ensure_backend_ready(self, create_indexes: bool = False) -> None:
        """Ensure backend resources exist.

        This method is idempotent and safe to call multiple times.
        """
        await self._ensure_collection_exists()
        if create_indexes:
            await self._ensure_required_search_indexes()

        if not self._backend_validated:
            await self.validate_backend_configuration()
            self._backend_validated = True

    async def _ensure_collection_exists(self) -> None:
        """Ensure target collection exists (best effort, idempotent)."""
        if self._collection_ready:
            return

        if self._collection is not None:
            self._collection_ready = True
            return

        client = self._get_client()
        database = client[self.database]
        try:
            existing_collections = await database.list_collection_names()
            if self.collection not in existing_collections:
                await database.create_collection(self.collection)
                logger.info(
                    "Created Mongo collection %s.%s",
                    self.database,
                    self.collection,
                )
            else:
                logger.info(
                    "Mongo collection already exists: %s.%s",
                    self.database,
                    self.collection,
                )
        except CollectionInvalid:
            logger.info(
                "Mongo collection already exists (race): %s.%s",
                self.database,
                self.collection,
            )

        self._collection = database[self.collection]
        self._collection_ready = True

    async def _ensure_required_search_indexes(self) -> None:
        """Ensure Atlas vector/text search indexes exist (idempotent)."""
        if self._search_indexes_ready:
            return

        collection = self._get_collection()
        try:
            existing_search_indexes = await self._list_existing_search_indexes(collection)
        except Exception as exc:
            raise RuntimeError(
                "Unable to inspect Atlas search indexes automatically. "
                "Create required indexes manually in Atlas UI/CLI."
            ) from exc

        missing_models: List[SearchIndexModel] = []
        if self.index_name not in existing_search_indexes:
            missing_models.append(self._build_vector_search_index_model())
        if self.search_index_name not in existing_search_indexes:
            missing_models.append(self._build_text_search_index_model())

        if not missing_models:
            logger.info(
                "Required Atlas search indexes already exist: %s, %s",
                self.index_name,
                self.search_index_name,
            )
            self._search_indexes_ready = True
            return

        create_many = getattr(collection, "create_search_indexes", None)
        create_one = getattr(collection, "create_search_index", None)

        try:
            if callable(create_many):
                try:
                    await create_many(missing_models)
                except TypeError:
                    await create_many(models=missing_models)
            elif callable(create_one):
                for model in missing_models:
                    try:
                        await create_one(model=model)
                    except TypeError:
                        await create_one(model)
            else:
                raise RuntimeError(
                    "Atlas Search index creation API is unavailable in this environment. "
                    "Create search indexes manually in Atlas UI/CLI."
                )
        except OperationFailure as exc:
            raise RuntimeError(
                "Atlas rejected automatic index creation (permission/API limitation). "
                "Create required indexes manually in Atlas UI/CLI."
            ) from exc

        created_names = [
            getattr(model, "name", None) or f"index_{idx}"
            for idx, model in enumerate(missing_models, start=1)
        ]
        logger.info("Created Atlas search indexes: %s", created_names)
        self._search_indexes_ready = True

    async def _list_existing_search_indexes(self, collection) -> set[str]:
        """List existing Atlas Search indexes by name."""
        list_indexes = getattr(collection, "list_search_indexes", None)
        if not callable(list_indexes):
            raise RuntimeError(
                "Atlas Search index listing API is unavailable in this environment. "
                "Create search indexes manually in Atlas UI/CLI."
            )

        cursor = list_indexes()
        documents = await cursor.to_list(length=None)
        names: set[str] = set()
        for document in documents:
            if isinstance(document, dict):
                name = document.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
        return names

    def _build_vector_search_index_model(self) -> SearchIndexModel:
        """Build Atlas vector search index definition."""
        return SearchIndexModel(
            name=self.index_name,
            type="vectorSearch",
            definition={
                "fields": [
                    {
                        "numDimensions": self.effective_dimension,
                        "path": "embedding",
                        "similarity": "cosine",
                        "type": "vector",
                    }
                ]
            },
        )

    def _build_text_search_index_model(self) -> SearchIndexModel:
        """Build Atlas full-text search index definition."""
        return SearchIndexModel(
            name=self.search_index_name,
            type="search",
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "text": {
                            "type": "string",
                        }
                    },
                }
            },
        )

    async def validate_backend_configuration(self) -> None:
        """Best-effort backend/index validation.

        Validates vector index dimension when Atlas metadata is available.
        """
        collection = self._get_collection()
        list_indexes = getattr(collection, "list_search_indexes", None)
        if not callable(list_indexes):
            logger.info(
                "Skipping Atlas index validation because list_search_indexes is unavailable."
            )
            return

        cursor = list_indexes()
        documents = await cursor.to_list(length=None)
        vector_index = None
        for document in documents:
            if isinstance(document, dict) and document.get("name") == self.index_name:
                vector_index = document
                break

        if vector_index is None:
            logger.info("Vector search index %s not found during validation", self.index_name)
            return

        definition = vector_index.get("latestDefinition") or vector_index.get("definition") or {}
        index_dims = self._extract_vector_index_dimensions(definition)
        if index_dims is None:
            logger.info(
                "Could not extract numDimensions from index %s during validation",
                self.index_name,
            )
            return

        if int(index_dims) != int(self.effective_dimension):
            raise ValueError(
                "Atlas vector index dimension mismatch: "
                f"index '{self.index_name}' has {index_dims}, "
                f"store effective dimension is {self.effective_dimension}."
            )

    def _extract_vector_index_dimensions(self, definition: Dict[str, Any]) -> Optional[int]:
        """Extract vector index numDimensions from Atlas index definition."""
        fields = definition.get("fields")
        if not isinstance(fields, list):
            return None

        for field in fields:
            if not isinstance(field, dict):
                continue
            if field.get("type") != "vector":
                continue
            if field.get("path") != "embedding":
                continue
            dims = field.get("numDimensions")
            if isinstance(dims, int):
                return dims

        return None
    
    async def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector-only or hybrid search.

        Hybrid mode internally combines vector similarity and Atlas keyword/full-text
        results using RRF.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter dict
            use_hybrid: If True, use hybrid search (vector + keyword with RRF)
        
        Returns:
            List of result dictionaries sorted by relevance:
            [
                {
                    "id": "doc_id",
                    "text": "document text",
                    "metadata": {"key": "value"},
                    "score": 0.95
                },
                ...
            ]
        """
        if self.auto_setup:
            await self.ensure_backend_ready(create_indexes=True)

        # Embed the query
        query_embedding = await self.embedding_model.embed(
            query,
            mode=EmbeddingMode.QUERY.value,
        )
        self._validate_query_embedding(query_embedding)
        
        if use_hybrid:
            # Perform hybrid search with RRF
            return await self._hybrid_search(
                query_embedding, query, k, filter
            )
        else:
            # Perform vector-only search
            return await self._vector_search(query_embedding, k, filter)
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search.
        
        Args:
            query_embedding: Embedded query vector
            k: Number of results to return
            filter: Optional metadata filter
        
        Returns:
            List of result dictionaries
        """
        self._validate_query_embedding(query_embedding)
        collection = self._get_collection()
        
        # Build filter pipeline
        pipeline = []
        
        # Add metadata filter if provided
        if filter:
            filter_doc = {"metadata": {"$eq": filter}}
        else:
            filter_doc = {}
        
        # Vector search stage
        vector_search_stage = {
            "$vectorSearch": {
                "index": self.index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 10,  # Search more candidates for better results
                "limit": k
            }
        }
        
        # Add preFilter if metadata filter provided
        if filter:
            vector_search_stage["$vectorSearch"]["preFilter"] = filter_doc
        
        pipeline.append(vector_search_stage)
        
        # Project results to desired format
        pipeline.append({
            "$project": {
                "id": "$_id",
                "text": "$text",
                "metadata": "$metadata",
                "score": {"$meta": "vectorSearchScore"}
            }
        })
        
        # Execute aggregation
        cursor = await collection.aggregate(pipeline)
        results = await cursor.to_list(length=k)
        
        # Convert ObjectId to string and ensure metadata is dict
        return self._format_results(results)
    
    async def _keyword_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search using Atlas Search.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter
        
        Returns:
            List of result dictionaries
        """
        collection = self._get_collection()
        
        # Build search pipeline
        pipeline = []
        
        # Atlas Search stage
        search_stage = {
            "$search": {
                "index": self.search_index_name,
                "text": {
                    "query": query,
                    "path": "text",
                    "fuzzy": {"maxEdits": 1}
                }
            }
        }
        
        # Add filter if provided
        if filter:
            search_stage["$search"]["filter"] = filter
        
        pipeline.append(search_stage)
        
        # Project results
        pipeline.append({
            "$project": {
                "id": "$_id",
                "text": "$text",
                "metadata": "$metadata",
                "score": {"$meta": "searchScore"}
            }
        })
        
        # Limit results
        pipeline.append({"$limit": k})
        
        # Execute aggregation
        try:
            cursor = await collection.aggregate(pipeline)
            results = await cursor.to_list(length=k)
            return self._format_results(results)
        except OperationFailure as e:
            logger.warning(
                f"Atlas Search failed, falling back to vector-only search: {e}"
            )
            return []
    
    async def _hybrid_search(
        self,
        query_embedding: List[float],
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with Reciprocal Rank Fusion.
        
        Executes both vector and keyword search in parallel, then
        merges results using RRF algorithm.
        
        Args:
            query_embedding: Embedded query vector
            query: Original query text
            k: Number of results to return
            filter: Optional metadata filter
        
        Returns:
            List of merged and ranked result dictionaries
        """
        import asyncio
        from .base import reciprocal_rank_fusion
        
        # Run both searches in parallel
        vector_results, keyword_results = await asyncio.gather(
            self._vector_search(query_embedding, k, filter),
            self._keyword_search(query, k, filter)
        )
        
        # If keyword search failed, return vector results
        if not keyword_results:
            return vector_results
        
        # If vector search failed, return keyword results
        if not vector_results:
            return keyword_results
        
        # Merge using RRF
        merged = reciprocal_rank_fusion(
            [vector_results, keyword_results],
            k=60
        )
        
        # Return top k results
        return merged[:k]
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Automatically generates embeddings using the embedding_model.
        
        Args:
            texts: List of text chunks to store
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
        
        Returns:
            List of document IDs
        """
        if self.auto_setup:
            await self.ensure_backend_ready(create_indexes=True)

        # Validate inputs
        self._validate_inputs(texts, metadatas, ids)
        
        # Generate IDs if not provided
        if ids is None:
            ids = self._generate_ids(len(texts))
        
        # Generate embeddings for all texts
        embeddings = await self.embedding_model.embed_batch(
            texts,
            mode=EmbeddingMode.DOCUMENT.value,
        )
        self._validate_document_embeddings(embeddings)
        
        # Prepare documents
        documents = []
        for i, text in enumerate(texts):
            doc = {
                "_id": ids[i],
                "text": text,
                "embedding": embeddings[i]
            }
            
            # Add metadata if provided
            if metadatas and i < len(metadatas):
                doc["metadata"] = metadatas[i]
            else:
                doc["metadata"] = {}
            
            documents.append(doc)
        
        # Insert into MongoDB
        collection = self._get_collection()
        
        # Use insert_many with ordered=False to continue on errors
        try:
            await collection.insert_many(documents, ordered=False)
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise
        
        return ids
    
    async def delete(self, ids: Optional[List[str]] = None) -> int:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete.
                If None, deletes all documents.
        
        Returns:
            Number of documents deleted
        """
        collection = self._get_collection()
        
        if ids is None:
            # Delete all documents
            result = await collection.delete_many({})
        else:
            # Delete specific documents
            result = await collection.delete_many({"_id": {"$in": ids}})
        
        return result.deleted_count
    
    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve
        
        Returns:
            List of document dictionaries
        """
        collection = self._get_collection()
        
        cursor = collection.find({"_id": {"$in": ids}})
        results = await cursor.to_list(length=None)
        
        return self._format_results(results)
    
    def _format_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Convert MongoDB results to native Python dicts.
        
        Args:
            results: Raw MongoDB query results
        
        Returns:
            List of formatted dictionaries with string IDs
        """
        formatted = []
        for doc in results:
            formatted_doc = {
                "id": str(doc["_id"]),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}) or {}
            }
            
            # Add score if present
            if "score" in doc:
                formatted_doc["score"] = doc["score"]
            
            # Add rrf_score if present
            if "rrf_score" in doc:
                formatted_doc["rrf_score"] = doc["rrf_score"]
            
            formatted.append(formatted_doc)
        
        return formatted
    
    async def close(self):
        """Close MongoDB and embedding model resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None
            self._collection_ready = False
            self._search_indexes_ready = False
            self._backend_validated = False

        try:
            await self.embedding_model.close()
        except Exception as exc:
            logger.warning("Error while closing embedding model: %s", exc)
    
    def __del__(self):
        """Ensure connection cleanup on object deletion (best effort)."""
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            # Never raise from finalizer.
            pass