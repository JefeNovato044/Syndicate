"""Embedding model interfaces and implementations.

This module provides embedding model abstractions for converting text
into vector representations. It supports both local models and API-based
services.

IMPORTANT: This module ONLY contains embedding generation logic.
Vector storage and retrieval are handled by the vectorstores module.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import asyncio


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.
    
    Embedding models convert text into vector representations (embeddings)
    that can be used for semantic similarity search.
    """
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Default implementation calls embed() for each text sequentially.
        Subclasses should override this for batch optimization.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors (one per input text)
        """
        return await asyncio.gather(*[self.embed(text) for text in texts])
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors.
        
        Returns:
            Integer dimension of embeddings produced by this model
        """
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """Local embedding using sentence-transformers.
    
    This implementation uses the sentence-transformers library to
    generate embeddings locally without API calls.
    
    IMPORTANT: Uses thread pool executor to prevent blocking the
    event loop during CPU-bound matrix multiplication operations.
    
    Example:
        ```python
        from syndicate.ingestion import SentenceTransformerEmbedding
        
        embedding_model = SentenceTransformerEmbedding(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Generate single embedding
        embedding = await embedding_model.embed("Hello world")
        
        # Generate batch embeddings
        embeddings = await embedding_model.embed_batch([
            "First document",
            "Second document"
        ])
        ```
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Popular options:
                       - "all-MiniLM-L6-v2" (fast, 384 dimensions)
                       - "all-mpnet-base-v2" (better quality, 768 dimensions)
                       - "paraphrase-MiniLM-L3-v2" (fast, 384 dimensions)
        """
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load the model to avoid unnecessary imports."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text using thread pool.
        
        Uses asyncio.get_running_loop().run_in_executor() to execute
        the synchronous encoding in a thread pool, preventing blocking
        of the event loop during CPU-bound matrix multiplication.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        loop = asyncio.get_running_loop()

        # Load model and encode inside the thread — prevents blocking on first call
        # (SentenceTransformer.__init__ downloads + loads the model into memory)
        def _encode():
            return self._load_model().encode(text, convert_to_numpy=True)

        embedding = await loop.run_in_executor(None, _encode)

        # Convert numpy array to list for JSON serialization
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using thread pool.
        
        Optimized batch implementation that encodes all texts in a
        single forward pass for better performance.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []
        
        loop = asyncio.get_running_loop()

        # Load model and encode inside the thread — prevents blocking on first call
        def _encode_batch():
            return self._load_model().encode(texts, convert_to_numpy=True)

        embeddings = await loop.run_in_executor(None, _encode_batch)

        # Convert numpy array to list of lists
        return embeddings.tolist()
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI API-based embedding model.
    
    This implementation uses OpenAI's embedding API to generate
    embeddings. Requires an OpenAI API key.
    
    Example:
        ```python
        from syndicate.ingestion import OpenAIEmbedding
        
        embedding_model = OpenAIEmbedding(
            api_key="sk-...",
            model="text-embedding-3-small"
        )
        
        # Generate embedding
        embedding = await embedding_model.embed("Hello world")
        ```
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None
    ):
        """
        Args:
            api_key: OpenAI API key
            model: Model to use. Options:
                   - "text-embedding-3-small" (1536 dimensions, faster, cheaper)
                   - "text-embedding-3-large" (3072 dimensions, better quality)
                   - "text-embedding-ada-002" (1536 dimensions, legacy)
            base_url: Optional custom base URL for API compatibility layers
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None
    
    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        client = self._get_client()
        
        response = await client.embeddings.create(
            model=self.model,
            input=text
        )
        
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Note: OpenAI API has a token limit per request. This implementation
        processes texts in batches to stay within limits.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []
        
        client = self._get_client()
        
        # OpenAI allows up to ~200k tokens per request
        # We'll process in batches of 100 texts to be safe
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension based on model name."""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimension_map.get(self.model, 1536)


class GeminiEmbedding(EmbeddingModel):
    """Google Gemini API-based embedding model.
    
    This implementation uses Google's Gemini API to generate
    embeddings. Requires a Google API key.
    
    Example:
        ```python
        from syndicate.ingestion import GeminiEmbedding
        
        embedding_model = GeminiEmbedding(
            api_key="AIza...",
            model="models/embedding-001"
        )
        
        # Generate embedding
        embedding = await embedding_model.embed("Hello world")
        ```
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "models/embedding-001"
    ):
        """
        Args:
            api_key: Google API key
            model: Model to use. Options:
                   - "models/embedding-001" (768 dimensions)
                   - "models/text-embedding-004" (newer, better quality)
        """
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load the Google Generative AI client."""
        if self._client is None:
            from google.generativeai import GenerativeModel, configure
            configure(api_key=self.api_key)
            self._client = GenerativeModel(self.model)
        return self._client
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Gemini API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        # Run synchronous API call in thread pool
        embedding = await loop.run_in_executor(
            None,
            lambda: self._get_client().embed_content(text)
        )
        
        return embedding.embedding.values
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors (one per input text)
        """
        return await asyncio.gather(*[self.embed(text) for text in texts])
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension based on model name."""
        dimension_map = {
            "models/embedding-001": 768,
            "models/text-embedding-004": 1024
        }
        return dimension_map.get(self.model, 768)


class CohereEmbedding(EmbeddingModel):
    """Cohere API-based embedding model.
    
    This implementation uses Cohere's embedding API to generate
    embeddings. Requires a Cohere API key.
    
    Example:
        ```python
        from syndicate.ingestion import CohereEmbedding
        
        embedding_model = CohereEmbedding(
            api_key="...",
            model="embed-english-v3.0"
        )
        
        # Generate embedding
        embedding = await embedding_model.embed("Hello world")
        ```
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document"
    ):
        """
        Args:
            api_key: Cohere API key
            model: Model to use. Options:
                   - "embed-english-v3.0" (1024 dimensions, English only)
                   - "embed-multilingual-v3.0" (1024 dimensions, multilingual)
                   - "embed-english-light-v3.0" (384 dimensions, faster)
            input_type: Type of input for better embeddings. Options:
                       - "search_document" (for storing documents)
                       - "search_query" (for queries)
                       - "classification" (for classification tasks)
                       - "clustering" (for clustering tasks)
        """
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self._client = None
    
    def _get_client(self):
        """Lazy load the Cohere client."""
        if self._client is None:
            import cohere
            self._client = cohere.AsyncClient(api_key=self.api_key)
        return self._client
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Cohere API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        client = self._get_client()
        
        response = await client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type
        )
        
        return response.embeddings[0].tolist() if hasattr(response.embeddings[0], 'tolist') else response.embeddings[0]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []
        
        client = self._get_client()
        
        # Cohere has a limit of 96 texts per request
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type
            )
            
            embeddings = [
                emb.tolist() if hasattr(emb, 'tolist') else emb
                for emb in response.embeddings
            ]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension based on model name."""
        dimension_map = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384
        }
        return dimension_map.get(self.model, 1024)