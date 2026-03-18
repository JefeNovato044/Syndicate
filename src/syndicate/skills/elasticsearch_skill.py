# mind/skills/elasticsearch_skill.py
"""
Elasticsearch Manager Skill Module

Provides domain expertise and tools for managing Elasticsearch clusters,
including indexing, searching, querying, and cluster management.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from ..tools.base_tool import BaseTool


# ==================== Elasticsearch Tools ====================

class ESIndexTool(BaseTool):
    """
    Tool for creating or updating an Elasticsearch index.
    """
    name = "es_index"
    description = "Create or update an Elasticsearch index with specified settings and mappings"
    args_schema = None
    
    def run(self, index_name: str, settings: Optional[Dict] = None, 
            mappings: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Create or update an Elasticsearch index.
        
        Args:
            index_name: Name of the index to create/update
            settings: Index settings (number_of_shards, number_of_replicas, etc.)
            mappings: Index mappings (field types, analyzers, etc.)
            
        Returns:
            Result of the index operation
        """
        # This would connect to Elasticsearch and perform the operation
        # For now, return a mock response
        return {
            "success": True,
            "message": f"Index '{index_name}' created/updated",
            "index_name": index_name,
            "settings": settings or {},
            "mappings": mappings or {}
        }


class ESSearchTool(BaseTool):
    """
    Tool for searching documents in Elasticsearch.
    """
    name = "es_search"
    description = "Search documents in Elasticsearch using query DSL"
    args_schema = None
    
    def run(self, index_name: str, query: Dict, size: int = 10, 
            **kwargs) -> Dict[str, Any]:
        """
        Search documents in Elasticsearch.
        
        Args:
            index_name: Name of the index to search
            query: Query DSL dictionary (must, should, filter, etc.)
            size: Number of results to return
            
        Returns:
            Search results
        """
        return {
            "success": True,
            "index_name": index_name,
            "query": query,
            "size": size,
            "results": []  # Would contain actual results from ES
        }


class ESDeleteTool(BaseTool):
    """
    Tool for deleting documents from Elasticsearch.
    """
    name = "es_delete"
    description = "Delete documents from Elasticsearch by ID or query"
    args_schema = None
    
    def run(self, index_name: str, doc_id: Optional[str] = None,
            query: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Delete documents from Elasticsearch.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID to delete (if query is None)
            query: Query to match documents for deletion (if doc_id is None)
            
        Returns:
            Deletion result
        """
        if doc_id:
            return {
                "success": True,
                "message": f"Document '{doc_id}' deleted from index '{index_name}'"
            }
        elif query:
            return {
                "success": True,
                "message": f"Documents matching query deleted from index '{index_name}'",
                "query": query
            }
        else:
            return {
                "success": False,
                "error": "Either doc_id or query must be provided"
            }


class ESGetTool(BaseTool):
    """
    Tool for retrieving a document from Elasticsearch.
    """
    name = "es_get"
    description = "Retrieve a document from Elasticsearch by ID"
    args_schema = None
    
    def run(self, index_name: str, doc_id: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve a document from Elasticsearch.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID to retrieve
            
        Returns:
            Document data
        """
        return {
            "success": True,
            "index_name": index_name,
            "doc_id": doc_id,
            "document": {}  # Would contain actual document data
        }


class ESCreateDocumentTool(BaseTool):
    """
    Tool for creating a new document in Elasticsearch.
    """
    name = "es_create_document"
    description = "Create a new document in an Elasticsearch index"
    args_schema = None
    
    def run(self, index_name: str, document: Dict, doc_id: Optional[str] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Create a new document in Elasticsearch.
        
        Args:
            index_name: Name of the index
            document: Document data to index
            doc_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            Creation result with document ID
        """
        return {
            "success": True,
            "message": "Document created successfully",
            "index_name": index_name,
            "doc_id": doc_id or "auto-generated",
            "document": document
        }


class ESUpdateTool(BaseTool):
    """
    Tool for updating a document in Elasticsearch.
    """
    name = "es_update"
    description = "Update an existing document in Elasticsearch"
    args_schema = None
    
    def run(self, index_name: str, doc_id: str, update: Dict,
            **kwargs) -> Dict[str, Any]:
        """
        Update a document in Elasticsearch.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID to update
            update: Update operations (doc, script, etc.)
            
        Returns:
            Update result
        """
        return {
            "success": True,
            "message": f"Document '{doc_id}' updated in index '{index_name}'",
            "index_name": index_name,
            "doc_id": doc_id,
            "update": update
        }


class ESHealthTool(BaseTool):
    """
    Tool for checking Elasticsearch cluster health.
    """
    name = "es_health"
    description = "Check the health status of an Elasticsearch cluster"
    args_schema = None
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Check Elasticsearch cluster health.
        
        Returns:
            Cluster health status
        """
        return {
            "success": True,
            "status": "green",
            "number_of_nodes": 3,
            "active_shards": 100,
            "relocating_shards": 0,
            "initializing_shards": 0,
            "unassigned_shards": 0
        }


class ESTermAggregationTool(BaseTool):
    """
    Tool for performing term aggregations in Elasticsearch.
    """
    name = "es_term_aggregation"
    description = "Perform a term aggregation on an Elasticsearch index"
    args_schema = None
    
    def run(self, index_name: str, field: str, size: int = 10,
            **kwargs) -> Dict[str, Any]:
        """
        Perform a term aggregation.
        
        Args:
            index_name: Name of the index
            field: Field to aggregate on
            size: Number of unique terms to return
            
        Returns:
            Aggregation results
        """
        return {
            "success": True,
            "index_name": index_name,
            "field": field,
            "size": size,
            "buckets": []  # Would contain actual aggregation results
        }


# ==================== Elasticsearch Skill Module ====================

class ElasticsearchSkill:
    """
    Elasticsearch Manager Skill Module.
    
    Provides domain expertise and tools for managing Elasticsearch clusters.
    """
    
    def __init__(self):
        self.name = "Elasticsearch Manager"
        self.description = "Expertise in Elasticsearch cluster management, indexing, searching, and querying"
        
        # Domain expertise
        self.expertise = """
        You are an Elasticsearch expert with deep knowledge of:
        
        **Indexing & Mapping:**
        - Understanding index templates and component templates
        - Creating optimized mappings for different data types
        - Handling nested objects, arrays, and parent-child relationships
        - Setting up proper analyzers (standard, custom, edge_ngram, etc.)
        
        **Querying & Search:**
        - Full-text search with match, multi_match, and bool queries
        - Filtering with term, range, and prefix queries
        - Aggregations (terms, range, histogram, date_histogram)
        - Sorting, pagination, and highlighting
        - Using query DSL effectively
        
        **Performance Optimization:**
        - Index optimization (refresh_interval, merge policies)
        - Shard allocation strategies
        - Query optimization (using filters instead of queries where possible)
        - Caching strategies (query cache, request cache)
        
        **Cluster Management:**
        - Monitoring cluster health (green/yellow/red status)
        - Managing shards and replicas
        - Handling rebalancing and recovery
        - Understanding node roles (master, data, coordinating)
        
        **Best Practices:**
        - Using index lifecycle management (ILM)
        - Implementing rollup jobs for large datasets
        - Proper use of aliases for index rotation
        - Monitoring and alerting setup
        """
        
        # Capabilities
        self.capabilities = [
            "Create and manage Elasticsearch indices",
            "Perform complex searches and queries",
            "Execute aggregations for analytics",
            "Optimize index performance",
            "Monitor cluster health",
            "Manage documents (CRUD operations)",
            "Set up index templates and aliases"
        ]
        
        # Glossary
        self.glossary = {
            "Index": "A collection of documents stored in Elasticsearch",
            "Document": "A single JSON document stored in an index",
            "Shard": "A partition of an index that lives on a single node",
            "Replica": "A copy of a shard for high availability",
            "Mapping": "Defines the data types of fields in an index",
            "Analyzer": "A process that breaks text into tokens for search",
            "Query DSL": "Elasticsearch's domain-specific language for queries",
            "Bool Query": "A query that combines multiple sub-queries (must, should, filter, must_not)",
            "Aggregation": "A way to analyze and summarize data in Elasticsearch",
            "ILM": "Index Lifecycle Management for managing index retention",
            "Alias": "A pointer to one or more indices for easier access",
            "Refresh": "The process that makes changes visible to search",
            "Merge": "The process of merging segments into larger segments"
        }
        
        # Bundled tools
        self.tools = [
            ESIndexTool(),
            ESSearchTool(),
            ESDeleteTool(),
            ESGetTool(),
            ESCreateDocumentTool(),
            ESUpdateTool(),
            ESHealthTool(),
            ESTermAggregationTool()
        ]
        
        # MCP servers
        self.mcp_servers = ["elasticsearch-mcp-server"]
    
    def get_skill_module(self):
        """
        Get the SkillModule instance for this skill.
        
        Returns:
            SkillModule instance
        """
        from .skill_module import SkillModule
        
        return SkillModule(
            name=self.name,
            description=self.description,
            expertise=self.expertise,
            capabilities=self.capabilities,
            glossary=self.glossary,
            tools=self.tools,
            mcp_servers=self.mcp_servers,
            priority=10
        )


# ==================== Factory Function ====================

def create_elasticsearch_skill() -> ElasticsearchSkill:
    """
    Factory function to create an Elasticsearch skill instance.
    
    Returns:
        ElasticsearchSkill instance
        
    Example:
        es_skill = create_elasticsearch_skill()
        skill_module = es_skill.get_skill_module()
    """
    return ElasticsearchSkill()
