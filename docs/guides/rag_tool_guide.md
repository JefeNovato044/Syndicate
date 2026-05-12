# RAGSearchTool Guide

This guide covers how to use `RAGSearchTool` directly, bundle it with `KnowledgeBaseSkill`, and customize it via subclassing. It also reflects the current vector store contract (`dims`, mode-aware embeddings, and backend readiness utilities).

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Usage: RAGSearchTool](#basic-usage-ragsearchtool)
3. [Using KnowledgeBaseSkill](#using-knowledgebaseskill)
4. [Vector Store Readiness and Dimensions](#vector-store-readiness-and-dimensions)
5. [Filtering with `default_filter`](#filtering-with-default_filter)
6. [Customizing Output with `format_results()`](#customizing-output-with-format_results)
7. [Customizing Per-Result with `_format_single_result()`](#customizing-per-result-with-_format_single_result)
8. [Building a Fully Custom RAG Tool](#building-a-fully-custom-rag-tool)
9. [Migration: `get_result_text()` is Deprecated](#migration-get_result_text-is-deprecated)

---

## Overview

`RAGSearchTool` is the async-only search bridge between Syndicate's vector store layer and the agent's tool execution framework. Unlike the old `BaseRAGMemory` approach (which injected context silently into the prompt), `RAGSearchTool` gives the LLM a **tool** it can call when it needs information from the knowledge base.

```python
from syndicate.tools import RAGSearchTool

search_tool = RAGSearchTool(
    vector_store=vector_store,
    top_k=4,
    use_hybrid=True,
    default_filter=None  # optional metadata filter
)

agent.add_tool(search_tool)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_store` | `BaseVectorStore` | **required** | Vector store instance (e.g., `MongoVectorStore`) |
| `top_k` | `int` | `3` | Default number of results to return |
| `use_hybrid` | `bool` | `True` | Use hybrid search (vector + keyword) |
| `default_filter` | `Dict[str, Any]` | `None` | Metadata filter to scope all searches |

---

## Basic Usage: RAGSearchTool

### Step 1: Set up the Vector Store

```python
import os
from syndicate.ingestion import SentenceTransformerEmbedding
from syndicate.vectorstores import MongoVectorStore

async def setup_vector_store():
    embedding_model = SentenceTransformerEmbedding(
        model_name="all-MiniLM-L6-v2"  # 384 dimensions
    )
    
    vector_store = MongoVectorStore(
        connection_string=os.getenv("MONGODB_ATLAS_URI"),
        database="syndicate_demo",
        collection="knowledge_base",
        embedding_model=embedding_model,
        # Optional explicit override. Omit `dims` to trust model defaults.
        dims=384,
        index_name="vector_index",
        search_index_name="text_index"
    )

    # Optional bootstrap: attempts collection/index creation via Mongo API.
    # If unavailable, keep Atlas UI/CLI setup as developer responsibility.
    await vector_store.ensure_backend_ready(create_indexes=True)
    
    return vector_store
```

## Vector Store Readiness and Dimensions

`RAGSearchTool` inherits dimension and lifecycle behavior from the underlying vector store.

- `MongoVectorStore` embeds documents in `document` mode and queries in `query` mode.
- If `dims` is omitted, the store uses the embedding model's effective dimension.
- If `dims` conflicts with a fixed-dimension model, initialization fails fast.
- If your model supports dimension override, Syndicate reconfigures the model and emits a warning when store `dims` differs.

### Readiness Patterns

```python
# Recommended: validate/provision once at startup
await vector_store.ensure_backend_ready(create_indexes=True)

# Alternative: lazy readiness checks on search/write calls
vector_store = MongoVectorStore(
    ...,
    auto_setup=True,
)
```

### Step 2: Create and Add the Tool

```python
from syndicate.tools import RAGSearchTool
from syndicate.agents import GenericAgent
from syndicate.clients import GeminiClient

vector_store = await setup_vector_store()

# Create the search tool
search_tool = RAGSearchTool(
    vector_store=vector_store,
    top_k=4,
    use_hybrid=True
)

# Create agent and add the tool
llm_client = GeminiClient(model_name="gemini-2.5-flash")
agent = GenericAgent(llm_client=llm_client)
agent.add_tool(search_tool)

# Now the agent can call `search_knowledge_base` when needed
response = await agent.invoke(
    "What is the company's remote work policy?",
    owner_id="user123",
    chat_id="chat456"
)
```

When the agent decides to search, it makes a tool call like:

```json
{
    "name": "search_knowledge_base",
    "arguments": {
        "query": "remote work policy",
        "top_k": 3
    }
}
```

### Step 3: Default Output Format

By default, `RAGSearchTool` formats results like this:

```
[Result 1 (Relevance: 0.923)]
Source: employee_handbook.pdf, Page: 12
Remote employees are entitled to a home office stipend of $500 per month...

[Result 2 (Relevance: 0.847)]
Source: hr_policies.pdf, Page: 5
Employees working remotely must maintain core hours of 10am-3pm EST...
```

---

## Using KnowledgeBaseSkill

`KnowledgeBaseSkill` bundles `RAGSearchTool` with behavioral instructions that are auto-injected into the agent's system prompt. This teaches the agent **when** and **how** to use the knowledge base.

```python
from syndicate.skills import KnowledgeBaseSkill

kb_skill = KnowledgeBaseSkill(
    vector_store=vector_store,
    top_k=4,
    use_hybrid=True,
    domain="company HR policies"
)

agent.install_skill(kb_skill)
```

### What KnowledgeBaseSkill Adds

1. **RAGSearchTool** — the actual search tool
2. **Expertise text** — instructions about when to search, how to interpret results
3. **Capabilities list** — used for agent self-awareness
4. **Glossary** — RAG-related term definitions

### Customizing Instructions

You can customize the expertise text in three ways:

```python
# Option A: Replace the entire expertise
kb_skill = KnowledgeBaseSkill(
    vector_store=vector_store,
    domain="legal documents",
    instructions_template="You are a legal research assistant. "
                          "Always cite the source document name.",
    instructions_mode="replace"
)

# Option B: Append custom instructions to the default
kb_skill = KnowledgeBaseSkill(
    vector_store=vector_store,
    domain="technical documentation",
    instructions_template="When searching for API references, include code snippets.",
    instructions_mode="append"
)

# Option C: Use a callable for dynamic expertise
def build_expertise(domain):
    return f"You specialize in {domain}. Always prioritize recent documents."

kb_skill = KnowledgeBaseSkill(
    vector_store=vector_store,
    domain="customer support tickets",
    expertise_builder=build_expertise
)
```

---

## Filtering with `default_filter`

Use `default_filter` to scope all searches to a specific subset of your knowledge base. This is useful when you have a single vector store with documents from multiple domains.

```python
# Only search documents tagged with category='policies'
policy_tool = RAGSearchTool(
    vector_store=vector_store,
    top_k=3,
    default_filter={"category": "policies"}
)

# Only search documents from a specific department
engineering_tool = RAGSearchTool(
    vector_store=vector_store,
    top_k=5,
    default_filter={"department": "engineering", "version": "v2"}
)
```

When `default_filter` is set, every search automatically includes the filter. The filter is passed directly to `vector_store.search()` and interpreted by the selected backend.

### Combining with KnowledgeBaseSkill

`KnowledgeBaseSkill` creates its own `RAGSearchTool` internally. To add a filter, subclass:

```python
class FilteredKnowledgeBaseSkill(KnowledgeBaseSkill):
    def __init__(self, vector_store, default_filter=None, **kwargs):
        top_k = kwargs.get("top_k", 4)
        use_hybrid = kwargs.get("use_hybrid", True)

        # Initialize standard skill first
        super().__init__(vector_store=vector_store, **kwargs)

        # Replace default tool with a filtered version
        self.tools = [
            RAGSearchTool(
                vector_store=vector_store,
                top_k=top_k,
                use_hybrid=use_hybrid,
                default_filter=default_filter,
            )
        ]
```

---

## Customizing Output with `format_results()`

Override `format_results()` to completely control the output format. This hook receives the raw execution result dict and returns a string.

### Signature

```python
def format_results(self, execution_result: Dict[str, Any]) -> str:
    """
    Args:
        execution_result: Dict with keys: success, results, count, error
    
    Returns:
        Formatted string to return to the agent/LLM
    """
```

### Example: Citation-Style Output

```python
from syndicate.tools import RAGSearchTool

class CitationRAGSearchTool(RAGSearchTool):
    """Returns results with numbered citations for academic-style responses."""
    
    name = "search_knowledge_base_citations"
    description = (
        "Search the knowledge base and return results with numbered citations."
    )
    
    def format_results(self, execution_result):
        if not execution_result.get("success"):
            error = execution_result.get("error")
            return f"Search failed: {error}" if error else "Search failed."
        
        results = execution_result.get("results", [])
        if not results:
            return "No relevant information found."
        
        citations = []
        for i, r in enumerate(results):
            source = r.get("metadata", {}).get("source", "unknown")
            text = r.get("text", "")[:300]  # truncate long texts
            score = r.get("score") or r.get("rrf_score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            citations.append(
                f"[{i+1}] {text}... "
                f"(source: {source}, score: {score_str})"
            )
        
        return "\n".join(citations)
```

### Example: JSON Output

```python
import json

class JSONRAGSearchTool(RAGSearchTool):
    """Returns results as a JSON string for programmatic consumption."""
    
    name = "search_knowledge_base_json"
    description = (
        "Search the knowledge base and return results as JSON."
    )
    
    def format_results(self, execution_result):
        return json.dumps(execution_result, indent=2)
```

---

## Customizing Per-Result with `_format_single_result()`

Override `_format_single_result()` when you only need to change how **individual results** are formatted, without touching the overall structure.

### Signature

```python
def _format_single_result(
    self,
    result: Dict[str, Any],
    index: int
) -> str:
    """
    Args:
        result: Single result dict (id, text, metadata, score)
        index: 1-based result number
    
    Returns:
        Formatted string block for this result
    """
```

### Example: Styled Results

```python
from syndicate.tools import RAGSearchTool

class StyledRAGSearchTool(RAGSearchTool):
    """Returns results with decorative styling."""
    
    name = "search_knowledge_base_styled"
    description = "Search the knowledge base with styled output."
    
    def _format_single_result(self, result, index):
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        score = result.get("score") or result.get("rrf_score")
        
        source_str = f"Source: {source}"
        if page:
            source_str += f", Page: {page}"
        
        score_str = f" [Relevance: {score:.3f}]" if score else ""
        
        return (
            f"{'='*60}\n"
            f"  Result {index}{score_str}\n"
            f"{'='*60}\n"
            f"{source_str}\n\n"
            f"{result['text']}"
        )
```

### Default Implementation

The default `_format_single_result()` produces:

```
[Result 1 (Relevance: 0.923)]
Source: employee_handbook.pdf, Page: 12
Remote employees are entitled to a home office stipend...
```

---

## Building a Fully Custom RAG Tool

You can build a completely custom RAG tool by subclassing `RAGSearchTool` and overriding any combination of methods. Here's a comprehensive example:

### Example: Multi-Source RAG Tool

```python
from syndicate.tools import RAGSearchTool
from typing import Any, Dict

class MultiSourceRAGSearchTool(RAGSearchTool):
    """
    A custom RAG tool that groups results by source document
    and provides a summary count per source.
    """
    
    name = "search_multi_source"
    description = (
        "Search the knowledge base and group results by source document. "
        "Useful for finding information across multiple documents."
    )
    
    def format_results(self, execution_result: Dict[str, Any]) -> str:
        if not execution_result.get("success"):
            error = execution_result.get("error")
            return f"Search failed: {error}" if error else "Search failed."
        
        results = execution_result.get("results", [])
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Group results by source
        sources = {}
        for r in results:
            source = r.get("metadata", {}).get("source", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(r)
        
        # Build grouped output
        parts = []
        for i, (source, docs) in enumerate(sources.items(), 1):
            parts.append(f"--- Source {i}: {source} ({len(docs)} results) ---")
            for j, doc in enumerate(docs, 1):
                score = doc.get("score") or doc.get("rrf_score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                page = doc.get("metadata", {}).get("page")
                page_str = f" (Page {page})" if page else ""
                parts.append(
                    f"  [{i}.{j}] Relevance: {score_str}{page_str}\n"
                    f"  {doc['text'][:500]}"
                )
        
        return "\n\n".join(parts)


# Usage
rag_tool = MultiSourceRAGSearchTool(
    vector_store=vector_store,
    top_k=10,
    use_hybrid=True,
    default_filter={"category": "articles"}
)

agent.add_tool(rag_tool)
```

### Example: FAQ-Style RAG Tool

```python
class FAQSearchTool(RAGSearchTool):
    """
    A RAG tool optimized for FAQ-style Q&A.
    Returns concise Q&A pairs when available.
    """
    
    name = "search_faq"
    description = (
        "Search the FAQ knowledge base. "
        "Returns question-answer pairs when available."
    )
    
    def _format_single_result(self, result, index):
        metadata = result.get("metadata", {})
        question = metadata.get("question", "Unknown question")
        answer = result.get("text", "No answer found.")
        score = result.get("score") or result.get("rrf_score")
        
        score_str = f" (Relevance: {score:.3f})" if score else ""
        
        return (
            f"[Result {index}{score_str}]\n"
            f"Q: {question}\n"
            f"A: {answer}"
        )
```

---

## Migration: `get_result_text()` is Deprecated

The `get_result_text()` method is deprecated. If you have a subclass that overrides it, you should migrate to one of the new extension points.

### Old Way (Deprecated)

```python
class OldCustomTool(RAGSearchTool):
    def get_result_text(self, result):
        # Old: format a single result
        return f"RESULT: {result['text']}"
```

### New Way: Override `_format_single_result()`

```python
class NewCustomTool(RAGSearchTool):
    def _format_single_result(self, result, index):
        # New: per-result formatting
        return f"RESULT: {result['text']}"
```

### New Way: Override `format_results()`

```python
class NewCustomTool(RAGSearchTool):
    def format_results(self, execution_result):
        # New: full output formatting
        if not execution_result.get("success"):
            return "Search failed."
        results = execution_result.get("results", [])
        if not results:
            return "No results found."
        return "\n".join(
            f"RESULT: {r['text']}" for r in results
        )
```

When `get_result_text()` is called, it emits a `DeprecationWarning`:

```
DeprecationWarning: get_result_text() is deprecated. 
Override format_results() or _format_single_result() instead.
```

---

## Summary: Which Extension Point Should You Use?

| Goal | Override |
|------|----------|
| Change per-result styling (borders, prefixes, etc.) | `_format_single_result()` |
| Completely change output structure (JSON, citations, grouped) | `format_results()` |
| Add a filter to scope all searches | `default_filter` constructor param |
| Bundle tool with behavioral instructions | `KnowledgeBaseSkill` |

---

## File Reference

| File | Purpose |
|------|---------|
| [`src/syndicate/tools/rag_tool.py`](src/syndicate/tools/rag_tool.py) | `RAGSearchTool` source |
| [`src/syndicate/skills/rag_skill.py`](src/syndicate/skills/rag_skill.py) | `KnowledgeBaseSkill` source |
| [`src/syndicate/vectorstores/base.py`](src/syndicate/vectorstores/base.py) | `BaseVectorStore` interface |
| [`src/syndicate/vectorstores/mongo.py`](src/syndicate/vectorstores/mongo.py) | `MongoVectorStore` implementation |
| [`examples/rag_knowledge_base_example.py`](examples/rag_knowledge_base_example.py) | Full usage examples |
| [`docs/guides/agentic-rag.md`](docs/guides/agentic-rag.md) | End-to-end agentic RAG tutorial |
