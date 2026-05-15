"""Embedding model interfaces and production-ready implementations.

This module provides embedding model abstractions for converting text into
vector representations, with production-focused utilities such as:

- Mode-aware embeddings (query/document/etc.)
- Runtime model/capability introspection
- Dimension configuration and validation
- Retry with exponential backoff for transient provider failures
- Lifecycle cleanup hooks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingMode(str, Enum):
    """Semantic mode for embedding generation."""

    QUERY = "query"
    DOCUMENT = "document"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    SIMILARITY = "similarity"


@dataclass
class EmbeddingModelInfo:
    """Runtime metadata for an embedding model instance."""

    provider: str
    model_name: str
    default_dimension: int
    effective_dimension: int
    supports_dimension_override: bool
    dimension_source: str
    embedding_space_id: str


@dataclass
class EmbeddingModelCapabilities:
    """Capabilities for embedding behavior and operational limits."""

    supported_modes: List[str]
    supports_batching: bool
    supports_dimension_override: bool
    max_batch_size: Optional[int]
    max_input_tokens: Optional[int]


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.

    Embedding models convert text into vector representations that can be
    used for semantic similarity search.
    """

    provider: str = "unknown"
    default_mode: EmbeddingMode = EmbeddingMode.DOCUMENT

    # Best-effort provider limits (if known by implementation).
    max_batch_size: Optional[int] = None
    max_input_tokens: Optional[int] = None

    # Retry policy for transient provider/network failures.
    retry_attempts: int = 2
    retry_base_delay_seconds: float = 0.25

    def __init__(self) -> None:
        self._configured_dimension: Optional[int] = None
        self._dimension_source: str = "default"

    @abstractmethod
    async def embed(
        self,
        text: str,
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[float]:
        """Generate embedding for a single text."""

    async def embed_batch(
        self,
        texts: List[str],
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Default implementation calls embed() for each text concurrently.
        Subclasses should override for provider-native batch optimization.
        """
        prepared_texts = self._prepare_texts(texts)
        if not prepared_texts:
            return []

        embeddings = await asyncio.gather(
            *[self.embed(text, mode=mode) for text in prepared_texts]
        )
        self.validate_embeddings_dimension(embeddings, context="embed_batch")
        return embeddings

    @property
    @abstractmethod
    def default_dimension(self) -> int:
        """Return provider/model default dimension."""

    @property
    def embedding_dimension(self) -> int:
        """Return effective dimension (configured override or default)."""
        if self._configured_dimension is not None:
            return self._configured_dimension
        return self.default_dimension

    @property
    def model_name(self) -> str:
        """Return model identifier for introspection."""
        value = getattr(self, "model", None)
        if isinstance(value, str) and value:
            return value

        value = getattr(self, "_model_name", None)
        if isinstance(value, str) and value:
            return value

        return self.__class__.__name__

    @property
    def supports_dimension_override(self) -> bool:
        """Whether model/provider supports output dimension control."""
        return False

    @property
    def supported_modes(self) -> Set[EmbeddingMode]:
        """Supported embedding modes for this model."""
        return {
            EmbeddingMode.QUERY,
            EmbeddingMode.DOCUMENT,
            EmbeddingMode.CLASSIFICATION,
            EmbeddingMode.CLUSTERING,
            EmbeddingMode.SIMILARITY,
        }

    @property
    def dimension_source(self) -> str:
        """Return source of current effective dimension value."""
        return self._dimension_source

    @property
    def embedding_space_id(self) -> str:
        """Stable identifier used to prevent incompatible vector mixing."""
        return (
            f"{self.provider}:{self.model_name}:"
            f"dim={self.embedding_dimension}"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for diagnostics and observability."""
        info = EmbeddingModelInfo(
            provider=self.provider,
            model_name=self.model_name,
            default_dimension=self.default_dimension,
            effective_dimension=self.embedding_dimension,
            supports_dimension_override=self.supports_dimension_override,
            dimension_source=self.dimension_source,
            embedding_space_id=self.embedding_space_id,
        )
        return asdict(info)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities/limits for this embedding implementation."""
        capabilities = EmbeddingModelCapabilities(
            supported_modes=[mode.value for mode in sorted(self.supported_modes, key=lambda m: m.value)],
            supports_batching=True,
            supports_dimension_override=self.supports_dimension_override,
            max_batch_size=self.max_batch_size,
            max_input_tokens=self.max_input_tokens,
        )
        return asdict(capabilities)

    def configure_dimension(self, dims: int, source: str = "explicit") -> None:
        """Configure effective output dimension.

        Implementations with fixed output dimensions should reject mismatches.
        """
        if dims <= 0:
            raise ValueError("dims must be a positive integer")

        if not self.supports_dimension_override and dims != self.default_dimension:
            raise ValueError(
                f"Model {self.model_name} has fixed dimension {self.default_dimension}; "
                f"cannot configure {dims}."
            )

        self._configured_dimension = dims
        self._dimension_source = source or "explicit"

    def supports_mode(self, mode: EmbeddingMode | str) -> bool:
        """Return whether a mode is supported by this model."""
        normalized = self._normalize_mode(mode)
        return normalized in self.supported_modes

    def validate_embedding_dimension(
        self,
        embedding: Sequence[float],
        context: str = "embedding",
    ) -> None:
        """Validate vector length against effective model dimension."""
        actual = len(embedding)
        expected = self.embedding_dimension
        if actual != expected:
            raise ValueError(
                f"{context} dimension mismatch: expected {expected}, got {actual}."
            )

    def validate_embeddings_dimension(
        self,
        embeddings: Sequence[Sequence[float]],
        context: str = "embeddings",
    ) -> None:
        """Validate all vectors in a batch against effective dimension."""
        for idx, embedding in enumerate(embeddings):
            self.validate_embedding_dimension(embedding, context=f"{context}[{idx}]")

    async def close(self) -> None:
        """Release provider resources (default no-op)."""

    def _normalize_mode(self, mode: EmbeddingMode | str) -> EmbeddingMode:
        if isinstance(mode, EmbeddingMode):
            normalized = mode
        elif isinstance(mode, str):
            try:
                normalized = EmbeddingMode(mode.strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported embedding mode '{mode}'. "
                    f"Supported: {[m.value for m in sorted(self.supported_modes, key=lambda item: item.value)]}."
                ) from exc
        else:
            raise TypeError("mode must be EmbeddingMode or str")

        if normalized not in self.supported_modes:
            raise ValueError(
                f"Mode '{normalized.value}' not supported by {self.model_name}."
            )

        return normalized

    def _prepare_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        cleaned = text.replace("\x00", "").strip()
        if not cleaned:
            raise ValueError("text cannot be empty")

        return cleaned

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        if texts is None:
            raise TypeError("texts cannot be None")

        return [self._prepare_text(text) for text in texts]

    async def _run_with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        operation_name: str,
    ) -> T:
        """Run an async operation with exponential backoff retries."""
        attempts = max(0, self.retry_attempts)

        for attempt in range(attempts + 1):
            try:
                return await operation()
            except Exception as exc:
                retryable = self._is_retryable_error(exc)
                exhausted = attempt >= attempts
                if exhausted or not retryable:
                    raise

                delay = self.retry_base_delay_seconds * (2**attempt)
                logger.warning(
                    "Embedding operation '%s' failed on attempt %s/%s (%s). Retrying in %.2fs.",
                    operation_name,
                    attempt + 1,
                    attempts + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # Unreachable, kept to satisfy static analyzers.
        raise RuntimeError(f"Embedding operation '{operation_name}' failed unexpectedly")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Best-effort transient error classifier."""
        if isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True

        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            if status_code in {408, 409, 429} or status_code >= 500:
                return True

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            if response_status in {408, 409, 429} or response_status >= 500:
                return True

        return False


class SentenceTransformerEmbedding(EmbeddingModel):
    """Local embedding using sentence-transformers."""

    provider = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self):
        """Lazy load the model to avoid unnecessary imports."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(
        self,
        text: str,
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[float]:
        """Generate embedding for a single text using a thread pool."""
        normalized_mode = self._normalize_mode(mode)
        prepared_text = self._prepare_text(text)

        loop = asyncio.get_running_loop()

        def _encode():
            return self._load_model().encode(prepared_text, convert_to_numpy=True)

        embedding = await loop.run_in_executor(None, _encode)
        vector = embedding.tolist()
        self.validate_embedding_dimension(vector, context=f"{normalized_mode.value} embedding")
        return vector

    async def embed_batch(
        self,
        texts: List[str],
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts using a thread pool."""
        normalized_mode = self._normalize_mode(mode)
        prepared_texts = self._prepare_texts(texts)
        if not prepared_texts:
            return []

        loop = asyncio.get_running_loop()

        def _encode_batch():
            return self._load_model().encode(prepared_texts, convert_to_numpy=True)

        embeddings = await loop.run_in_executor(None, _encode_batch)
        vectors = embeddings.tolist()
        self.validate_embeddings_dimension(vectors, context=f"{normalized_mode.value} embedding batch")
        return vectors

    @property
    def default_dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    async def close(self) -> None:
        # sentence-transformers models are in-process objects.
        self._model = None


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI API-based embedding model."""

    provider = "openai"
    max_batch_size = 100

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        retry_attempts: int = 2,
        retry_base_delay_seconds: float = 0.25,
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

        self.retry_attempts = retry_attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds

        if dimensions is not None:
            self.configure_dimension(dimensions)

    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def supports_dimension_override(self) -> bool:
        return self.model.startswith("text-embedding-3")

    @property
    def default_dimension(self) -> int:
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model, 1536)

    def configure_dimension(self, dims: int, source: str = "explicit") -> None:
        if not self.supports_dimension_override and dims != self.default_dimension:
            raise ValueError(
                f"Model {self.model} has fixed dimension {self.default_dimension}; "
                f"cannot configure {dims}."
            )
        super().configure_dimension(dims, source=source)

    async def embed(
        self,
        text: str,
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[float]:
        """Generate embedding using OpenAI API."""
        normalized_mode = self._normalize_mode(mode)
        prepared_text = self._prepare_text(text)
        client = self._get_client()

        async def _operation() -> List[float]:
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": prepared_text,
            }
            if self.supports_dimension_override and self._configured_dimension is not None:
                request_kwargs["dimensions"] = self._configured_dimension

            response = await client.embeddings.create(**request_kwargs)
            vector = list(response.data[0].embedding)
            self.validate_embedding_dimension(vector, context=f"{normalized_mode.value} embedding")
            return vector

        return await self._run_with_retry(_operation, operation_name=f"openai:{self.model}:embed")

    async def embed_batch(
        self,
        texts: List[str],
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with provider-native batching."""
        normalized_mode = self._normalize_mode(mode)
        prepared_texts = self._prepare_texts(texts)
        if not prepared_texts:
            return []

        client = self._get_client()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(prepared_texts), self.max_batch_size):
            batch = prepared_texts[i : i + self.max_batch_size]

            async def _operation() -> List[List[float]]:
                request_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "input": batch,
                }
                if self.supports_dimension_override and self._configured_dimension is not None:
                    request_kwargs["dimensions"] = self._configured_dimension

                response = await client.embeddings.create(**request_kwargs)
                vectors = [list(item.embedding) for item in response.data]
                self.validate_embeddings_dimension(
                    vectors,
                    context=f"{normalized_mode.value} embedding batch chunk",
                )
                return vectors

            vectors = await self._run_with_retry(
                _operation,
                operation_name=f"openai:{self.model}:embed_batch",
            )
            all_embeddings.extend(vectors)

        return all_embeddings

    async def close(self) -> None:
        if self._client is None:
            return

        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result

        self._client = None


class GeminiEmbedding(EmbeddingModel):
    """Google Gemini API-based embedding model."""

    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimensions: Optional[int] = None,
        enable_task_type_mapping: bool = True,
        retry_attempts: int = 2,
        retry_base_delay_seconds: float = 0.25,
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.enable_task_type_mapping = enable_task_type_mapping
        self._client = None

        self.retry_attempts = retry_attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds

        if dimensions is not None:
            self.configure_dimension(dimensions)

    def _get_client(self):
        """Lazy load Google GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def supports_dimension_override(self) -> bool:
        return True

    @property
    def default_dimension(self) -> int:
        # Keep legacy aliases for compatibility with previous docs/examples.
        dimension_map = {
            "gemini-embedding-2": 3072,
            "gemini-embedding-001": 3072,
            "models/embedding-001": 768,
            "models/text-embedding-004": 1024,
        }
        return dimension_map.get(self.model, 3072)

    def _resolve_task_type(self, mode: EmbeddingMode) -> Optional[str]:
        """Map generic modes to Gemini task_type when supported."""
        if not self.enable_task_type_mapping:
            return None

        if self.model not in {"gemini-embedding-001", "models/embedding-001"}:
            return None

        task_type_map = {
            EmbeddingMode.QUERY: "RETRIEVAL_QUERY",
            EmbeddingMode.DOCUMENT: "RETRIEVAL_DOCUMENT",
            EmbeddingMode.CLASSIFICATION: "CLASSIFICATION",
            EmbeddingMode.CLUSTERING: "CLUSTERING",
            EmbeddingMode.SIMILARITY: "SEMANTIC_SIMILARITY",
        }
        return task_type_map.get(mode)

    def _build_embed_config(self, mode: EmbeddingMode) -> Dict[str, Any]:
        config: Dict[str, Any] = {}

        if self._configured_dimension is not None:
            config["output_dimensionality"] = self._configured_dimension

        task_type = self._resolve_task_type(mode)
        if task_type is not None:
            config["task_type"] = task_type

        return config

    async def embed(
        self,
        text: str,
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[float]:
        """Generate embedding using Gemini API."""
        normalized_mode = self._normalize_mode(mode)
        prepared_text = self._prepare_text(text)
        loop = asyncio.get_running_loop()

        def _embed_sync() -> List[float]:
            from google.genai import types

            client = self._get_client()
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "contents": [prepared_text],
            }

            config_kwargs = self._build_embed_config(normalized_mode)
            if config_kwargs:
                request_kwargs["config"] = types.EmbedContentConfig(**config_kwargs)

            response = client.models.embed_content(**request_kwargs)
            embeddings = getattr(response, "embeddings", None)
            if not embeddings:
                raise RuntimeError("Gemini embedding response does not contain embeddings")

            embedding_obj = embeddings[0]
            values = getattr(embedding_obj, "values", None)
            if values is None and isinstance(embedding_obj, dict):
                values = embedding_obj.get("values")

            if values is None:
                raise RuntimeError("Gemini embedding payload is missing vector values")

            return list(values)

        async def _operation() -> List[float]:
            return await loop.run_in_executor(None, _embed_sync)

        vector = await self._run_with_retry(
            _operation,
            operation_name=f"gemini:{self.model}:embed",
        )
        self.validate_embedding_dimension(vector, context=f"{normalized_mode.value} embedding")
        return vector

    async def close(self) -> None:
        if self._client is None:
            return

        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result

        self._client = None


class CohereEmbedding(EmbeddingModel):
    """Cohere API-based embedding model."""

    provider = "cohere"
    max_batch_size = 96

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        query_input_type: str = "search_query",
        retry_attempts: int = 2,
        retry_base_delay_seconds: float = 0.25,
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.query_input_type = query_input_type
        self._client = None

        self.retry_attempts = retry_attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds

    def _get_client(self):
        """Lazy load the Cohere client."""
        if self._client is None:
            import cohere

            self._client = cohere.AsyncClient(api_key=self.api_key)
        return self._client

    @property
    def default_dimension(self) -> int:
        dimension_map = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
        }
        return dimension_map.get(self.model, 1024)

    def _resolve_input_type(self, mode: EmbeddingMode) -> str:
        mode_map = {
            EmbeddingMode.QUERY: self.query_input_type,
            EmbeddingMode.DOCUMENT: self.input_type,
            EmbeddingMode.CLASSIFICATION: "classification",
            EmbeddingMode.CLUSTERING: "clustering",
            EmbeddingMode.SIMILARITY: self.input_type,
        }
        return mode_map.get(mode, self.input_type)

    async def embed(
        self,
        text: str,
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[float]:
        """Generate embedding using Cohere API."""
        normalized_mode = self._normalize_mode(mode)
        prepared_text = self._prepare_text(text)
        input_type = self._resolve_input_type(normalized_mode)
        client = self._get_client()

        async def _operation() -> List[float]:
            response = await client.embed(
                texts=[prepared_text],
                model=self.model,
                input_type=input_type,
            )
            embedding = response.embeddings[0]
            vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            self.validate_embedding_dimension(vector, context=f"{normalized_mode.value} embedding")
            return vector

        return await self._run_with_retry(
            _operation,
            operation_name=f"cohere:{self.model}:embed",
        )

    async def embed_batch(
        self,
        texts: List[str],
        mode: EmbeddingMode | str = EmbeddingMode.DOCUMENT,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with provider-native batching."""
        normalized_mode = self._normalize_mode(mode)
        prepared_texts = self._prepare_texts(texts)
        if not prepared_texts:
            return []

        input_type = self._resolve_input_type(normalized_mode)
        client = self._get_client()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(prepared_texts), self.max_batch_size):
            batch = prepared_texts[i : i + self.max_batch_size]

            async def _operation() -> List[List[float]]:
                response = await client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                )
                vectors = [
                    emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    for emb in response.embeddings
                ]
                self.validate_embeddings_dimension(
                    vectors,
                    context=f"{normalized_mode.value} embedding batch chunk",
                )
                return vectors

            vectors = await self._run_with_retry(
                _operation,
                operation_name=f"cohere:{self.model}:embed_batch",
            )
            all_embeddings.extend(vectors)

        return all_embeddings

    async def close(self) -> None:
        if self._client is None:
            return

        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result

        self._client = None
