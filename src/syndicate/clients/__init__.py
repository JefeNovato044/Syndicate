"""Client module with base Client class for type safety."""

from .base import Client

# Import concrete implementations for public API
from .gemini import GeminiClient
from .openai import OpenAIClient

__all__ = ['Client', 'GeminiClient', 'OpenAIClient']


