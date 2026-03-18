"""RAG search tool for agent integration.

This module provides the RAGSearchTool that bridges the vector store
layer with the agent's tool execution framework. It allows LLMs to
search the knowledge base through tool calls.

Example:
    ```python
    from syndicate.vectorstores import MongoVectorStore
    from syndicate.tools import RAGSearchTool
    
    # Create vector store
    vector_store = MongoVectorStore(...)
    
    # Create search tool
    search_tool = RAGSearchTool(
        vector_store=vector_store,
        top_k=4
    )
    
    # Add to agent
    agent.add_tool(search_tool)
    ```
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base_tool import BaseTool
from ..vectorstores.base import BaseVectorStore


class RAGSearchArgs(BaseModel):
    """Arguments for the RAG search tool."""
    query: str = Field(..., description="The search query. Be specific and include key terms.")
    top_k: int = Field(default=3, description="Number of results to return (1-10).", ge=1, le=10)


class RAGSearchTool(BaseTool):
    """Tool for searching the knowledge base.
    
    This tool is provided to the LLM to execute semantic searches
    against the vector store. The LLM decides when to use this tool
    based on the user's query.
    
    The tool returns formatted search results that include:
    - Document text
    - Metadata (source, page, etc.)
    - Relevance scores
    
    Example tool call by LLM:
        {
            "name": "search_knowledge_base",
            "arguments": {
                "query": "What is the company's remote work policy?",
                "top_k": 3
            }
        }
    """
    
    name: str = "search_knowledge_base"
    description: str = (
        "Search the knowledge base for relevant information. "
        "Use this tool when you need to find specific information, "
        "facts, or context from stored documents. "
        "Provide a clear and specific query to get the best results."
    )
    args_schema = RAGSearchArgs
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        top_k: int = 3,
        use_hybrid: bool = True
    ):
        """
        Args:
            vector_store: Vector store instance for searching
            top_k: Default number of results to return
            use_hybrid: Whether to use hybrid search (vector + keyword)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.use_hybrid = use_hybrid

    def run(self, **kwargs) -> str:
        raise NotImplementedError(
            "RAGSearchTool is async-only. "
            "The framework calls run_async(); never call run() directly."
        )

    async def run_async(self, **kwargs) -> str:
        """Execute RAG search — called by the agent's tool execution framework."""
        args = self.args_schema(**kwargs)
        result = await self._execute(args.query, args.top_k)
        return result.get("formatted", "No results found.")

    async def _execute(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the search query against the vector store.
        
        This method is called by the agent's tool execution framework
        when the LLM decides to use this tool.
        
        Args:
            query: Search query text
            top_k: Optional override for number of results
        
        Returns:
            Dictionary with search results:
            {
                "success": True,
                "results": [...],
                "count": 3,
                "formatted": "Formatted text for LLM..."
            }
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.top_k
        k = max(1, min(k, 10))  # Clamp between 1 and 10
        
        try:
            # Search the vector store
            results = await self.vector_store.search(
                query=query,
                k=k,
                use_hybrid=self.use_hybrid
            )
            
            if not results:
                return {
                    "success": True,
                    "results": [],
                    "count": 0,
                    "formatted": "No relevant information found in the knowledge base."
                }
            
            # Format results for LLM consumption
            formatted_parts = []
            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                source = metadata.get("source", "unknown")
                page = metadata.get("page")
                
                # Build source attribution
                source_str = f"Source: {source}"
                if page:
                    source_str += f", Page: {page}"
                
                score = result.get("score") or result.get("rrf_score")
                score_str = f" (Relevance: {score:.3f})" if score else ""
                
                formatted_parts.append(
                    f"[Result {i}{score_str}]\n"
                    f"{source_str}\n"
                    f"{result['text']}"
                )
            
            formatted = "\n\n".join(formatted_parts)
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "formatted": formatted
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0,
                "formatted": f"Error searching knowledge base: {str(e)}"
            }
    
    def get_result_text(self, result: Dict[str, Any]) -> str:
        """
        Extract text from tool execution result for agent response.
        
        Args:
            result: Result dictionary from execute()
        
        Returns:
            Formatted text to return to the agent/LLM
        """
        if result.get("success"):
            return result.get("formatted", "")
        else:
            return result.get("formatted", "Error searching knowledge base.")