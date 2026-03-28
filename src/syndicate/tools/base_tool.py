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
    schema = _resolve_local_refs_for_gemini(dict(schema))

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
            options = [
                _clean_schema_for_gemini(s)
                for s in schema[combinator]
                if isinstance(s, dict)
            ]
            replacement = _collapse_gemini_combinator_options(options)
            schema.pop(combinator, None)
            if replacement:
                for key, value in replacement.items():
                    if key == "nullable":
                        schema[key] = bool(schema.get(key, False) or value)
                    elif key not in schema:
                        schema[key] = value

    return schema


def _is_null_schema(schema: Dict[str, Any]) -> bool:
    return schema.get("type") == "null" or schema.get("enum") == [None]


def _collapse_gemini_combinator_options(options: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse unsupported OpenAPI combinators to Gemini-compatible schema.

    Gemini function declarations reject `oneOf`/`anyOf`/`allOf` in parameters.
    We preserve the common nullable-union shape (T | null) by converting it to
    `{..., "nullable": true}`. For other unions, we fall back to the first
    non-null branch to keep schema validation strict and accepted by Gemini.
    """
    if not options:
        return {}

    nullable = any(_is_null_schema(option) for option in options)
    non_null_options = [option for option in options if not _is_null_schema(option)]

    if not non_null_options:
        return {"type": "string", "nullable": True}

    selected = dict(non_null_options[0])
    if nullable:
        selected["nullable"] = True

    return selected


def _resolve_local_refs_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve local JSON Schema refs (``#/$defs/...`` and ``#/definitions/...``).

    Gemini's FunctionDeclaration rejects raw ``$ref`` fields in parameters,
    so we inline local refs before final schema cleaning.
    """
    defs: Dict[str, Any] = {}
    defs.update(schema.get("$defs", {}) or {})
    defs.update(schema.get("definitions", {}) or {})

    def _resolve(node: Any, stack: Optional[List[str]] = None) -> Any:
        stack = stack or []

        if isinstance(node, list):
            return [_resolve(item, stack) for item in node]

        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str):
            ref_key = None
            if ref.startswith("#/$defs/"):
                ref_key = ref.split("#/$defs/", 1)[1]
            elif ref.startswith("#/definitions/"):
                ref_key = ref.split("#/definitions/", 1)[1]

            if ref_key and ref_key in defs and ref_key not in stack:
                target = defs[ref_key]
                if isinstance(target, dict):
                    merged = dict(target)
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = v
                    return _resolve(merged, stack + [ref_key])

            # Unresolvable or recursive ref: drop $ref to avoid Gemini validation errors.
            return {
                k: _resolve(v, stack)
                for k, v in node.items()
                if k != "$ref"
            }

        return {
            k: _resolve(v, stack)
            for k, v in node.items()
        }

    return _resolve(schema)


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