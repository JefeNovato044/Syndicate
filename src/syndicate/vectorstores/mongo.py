"""MongoDB Atlas vector store implementation.

This module provides a vector store implementation using MongoDB Atlas
with support for hybrid search (vector + keyword) and Reciprocal Rank
Fusion (RRF) for result merging.

IMPORTANT: This implementation requires MongoDB Atlas (not self-hosted
MongoDB) because it uses the $vectorSearch and $search aggregation stages
which are Atlas-only features.

Requirements:
    - pymongo>=4.6.0 (for AsyncMongoClient)
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
        vector_dimension=384
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
from pymongo.errors import OperationFailure, ConfigurationError

from .base import BaseVectorStore
from ..ingestion.embedding_models import EmbeddingModel

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
        vector_dimension: int,
        index_name: str = "vector_index",
        search_index_name: str = "text_index"
    ):
        """
        Args:
            connection_string: MongoDB Atlas connection string
                              Example: "mongodb+srv://user:pass@cluster.mongodb.net/"
            database: Database name
            collection: Collection name for storing vectors
            embedding_model: Embedding model for generating vectors
            vector_dimension: Dimension of embedding vectors (must match model)
            index_name: Name of the vector search index in Atlas
            search_index_name: Name of the Atlas Search index for keyword search
        """
        super().__init__(embedding_model=embedding_model)
        
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self.vector_dimension = vector_dimension
        self.index_name = index_name
        self.search_index_name = search_index_name
        
        self._client: Optional[AsyncMongoClient] = None
        self._collection = None
    
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
    
    async def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector, keyword, or hybrid search.
        
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
        collection = self._get_collection()
        
        # Embed the query
        query_embedding = await self.embedding_model.embed(query)
        
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
        cursor = collection.aggregate(pipeline)
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
                    "fuzzy": {"factor": 0.5}
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
            cursor = collection.aggregate(pipeline)
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
        # Validate inputs
        self._validate_inputs(texts, metadatas, ids)
        
        # Generate IDs if not provided
        if ids is None:
            ids = self._generate_ids(len(texts))
        
        # Generate embeddings for all texts
        embeddings = await self.embedding_model.embed_batch(texts)
        
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
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None
    
    def __del__(self):
        """Ensure connection is closed on deletion."""
        if self._client is not None:
            self._client.close()