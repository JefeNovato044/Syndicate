import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

try:
    from langchain.tools import StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def _clean_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip fields Gemini rejects from a JSON Schema dict (returns a shallow copy).
    Gemini does not accept: additionalProperties, title, $defs, definitions.
    Applied recursively to nested property schemas, array items, and combinators.
    """
    schema = dict(schema)
    for field in ("additionalProperties", "title", "$defs", "definitions"):
        schema.pop(field, None)

    if "properties" in schema:
        schema["properties"] = {
            k: _clean_schema_for_gemini(v)
            for k, v in schema["properties"].items()
        }

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _clean_schema_for_gemini(schema["items"])

    for combinator in ("oneOf", "anyOf", "allOf"):
        if combinator in schema and isinstance(schema[combinator], list):
            schema[combinator] = [
                _clean_schema_for_gemini(s) for s in schema[combinator] if isinstance(s, dict)
            ]

    return schema


class BaseTool(ABC):
    """
    Base class for all tools.
    Provides abstraction layer to convert tools to different provider formats.
    
    Supports both sync and async implementations:
    - Sync: def run(self, **kwargs) -> Any
    - Async: async def run(self, **kwargs) -> Any
    
    The framework automatically handles both cases via run_async().
    """
    name: str
    description: str
    args_schema: Optional[type] = None

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Main entry point for the tool. Subclasses must implement this.
        
        Can be sync or async:
        - Sync: def run(self, **kwargs) -> Any
        - Async: async def run(self, **kwargs) -> Any
        
        The framework will call run_async() which handles both cases.
        """
        raise NotImplementedError("Implement run method")

    async def run_async(self, **kwargs) -> Any:
        """
        Async wrapper that handles both sync and async implementations.
        
        This is the method the framework calls. It automatically detects
        whether the implementation is sync or async and handles accordingly.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
            
        Example:
            # Sync implementation
            class MyTool(BaseTool):
                def run(self, query: str) -> str:
                    return f"Searching: {query}"
            
            # Async implementation
            class MyAsyncTool(BaseTool):
                async def run(self, query: str) -> str:
                    await asyncio.sleep(1)  # Simulate async work
                    return f"Searching: {query}"
        """
        # Check if run is an async function
        if asyncio.iscoroutinefunction(self.run):
            # User implemented async - await it directly
            return await self.run(**kwargs)
        else:
            # User implemented sync - run in thread pool to avoid blocking
            return await asyncio.to_thread(self.run, **kwargs)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Safe execution wrapper with error handling.
        """
        try:
            result = self.run(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def get_gemini_tool_schema(self):
        """
        Returns tool schema in Gemini function calling format.
        Strips fields Gemini rejects (title, $defs, additionalProperties, definitions).
        """
        raw_params = (
            {"type": "object", "properties": {}}
            if self.args_schema is None
            else self.args_schema.model_json_schema()
        )
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _clean_schema_for_gemini(raw_params)
        }

    def get_openai_tool_schema(self):
        """
        Returns tool schema in OpenAI function calling format.
        """
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            }
        }
        
        if self.args_schema:
            schema["function"]["parameters"] = self.args_schema.model_json_schema()
        else:
            schema["function"]["parameters"] = {
                "type": "object",
                "properties": {}
            }
        
        return schema

    def as_langchain_tool(self):
        """
        Returns a langchain tool representation instance.
        Requires langchain to be installed.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Install with: pip install langchain")
        
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema
        )

    def to_format(self, format_type: str):
        """
        Universal converter for different provider formats.
        
        Args:
            format_type: One of 'gemini', 'openai', 'langchain'
            
        Returns:
            Tool schema in the requested format
        """
        converters = {
            "gemini": self.get_gemini_tool_schema,
            "openai": self.get_openai_tool_schema,
            "langchain": self.as_langchain_tool,
        }
        
        if format_type not in converters:
            raise ValueError(
                f"Unknown format: {format_type}. "
                f"Supported formats: {list(converters.keys())}"
            )
        
        return converters[format_type]()