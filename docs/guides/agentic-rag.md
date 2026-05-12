# Building Agentic RAG Solutions with Syndicate

This guide explains how to build Retrieval-Augmented Generation (RAG) capabilities into your AI agents using Syndicate's Agentic RAG architecture. 

> **Important Note:** Syndicate has moved away from the old framework-managed memory injection approach (`BaseRAGMemory`). Instead, it uses an **Agentic** approach where the LLM is equipped with tools and skills to actively search knowledge bases only when needed.

## Why Agentic RAG?

1. **Control**: The LLM decides *when* to search, avoiding polluting the prompt with irrelevant context.
2. **Modularity**: Vector stores live independently of the agent and can be shared across multiple agents.
3. **Transparency**: The retrieval acts as a standard tool call, visible in the agent's thought process and execution logs.

## Prerequisites

To follow this tutorial, you'll need to install Syndicate with the `rag` extras, which includes tools for embeddings and text splitting:

```bash
pip install "syndicate[rag]"

# Optional: For specialized embeddings
pip install "syndicate[rag,embeddings-openai]"
```

*For this guide, we'll assume you have a MongoDB Atlas cluster available for vector storage, though the principles apply to any built-in `BaseVectorStore`.*

## Core Components

- **Vector Store** (e.g., `MongoVectorStore`): Stores document chunks and handles hybrid/semantic search.
- **Embedding Model** (e.g., `SentenceTransformerEmbedding`): Converts text into vector embeddings with mode-aware generation (`document` for ingestion, `query` for retrieval).
- **Tools & Skills**: `RAGSearchTool` enables raw search, while `KnowledgeBaseSkill` bundles the tool with LLM behavioral instructions.

---

## Step-by-Step Tutorial

### 1. Set up the Vector Store

First, we need to initialize our embedding model and connect to a vector database. We'll use MongoDB Atlas with `all-MiniLM-L6-v2` embeddings.

```python
import os
from syndicate.ingestion import SentenceTransformerEmbedding
from syndicate.vectorstores import MongoVectorStore

async def setup_store():
    # 1. Create embedding model (runs locally)
    embedding_model = SentenceTransformerEmbedding(
        model_name="all-MiniLM-L6-v2" # 384 dimensions
    )
    
    # 2. Connect to vector store
    vector_store = MongoVectorStore(
        connection_string=os.getenv("MONGODB_ATLAS_URI"),
        database="syndicate_demo",
        collection="knowledge_base",
        embedding_model=embedding_model,
        # Optional explicit override. Omit `dims` to use model defaults.
        dims=384,
        index_name="vector_index",
        search_index_name="text_index"
    )

    # Optional: provision collection/indexes through Mongo API when available.
    # If Atlas API/permissions do not allow it, set up indexes manually in Atlas.
    await vector_store.ensure_backend_ready(create_indexes=True)
    
    return vector_store
```

### 1.1 Dimension and Embedding Mode Contract

- `MongoVectorStore` automatically uses `document` embeddings during `add_texts()` and `query` embeddings during `search()`.
- If `dims` is omitted, the store uses the embedding model's effective dimension.
- If `dims` is provided and the model has fixed output dimensions, mismatches fail fast with `ValueError`.
- If `dims` is provided and the model supports dimension override, Syndicate reconfigures the model and emits a warning so the override is visible.

### 1.2 Backend Bootstrap Options

You can choose either eager startup provisioning or lazy auto-setup:

```python
# Option A (recommended): provision/validate once during startup
await vector_store.ensure_backend_ready(create_indexes=True)

# Option B: defer provisioning checks until reads/writes
vector_store = MongoVectorStore(
    ...,
    auto_setup=True,
)
```

### 2. Ingesting Documents

Before an agent can search, you need knowledge in the database. Syndicate provides utilities like `RecursiveCharacterTextSplitter` to chunk your data.

```python
from syndicate.ingestion import RecursiveCharacterTextSplitter

async def ingest_data(vector_store):
    documents = [
        "Syndicate allows employees to work from anywhere.",
        "Employees get 15 vacation days in their first year."
    ]
    
    # Split text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
        
    # Add to vector store (embeddings are auto-generated)
    doc_ids = await vector_store.add_texts(texts=all_chunks)
    print(f"Added {len(doc_ids)} chunks to the database.")
```

### 3. Equipping the Agent with a Knowledge Base Skill

To give an agent access to your data, you don't just hand it the database—you give it a **Skill**. The `KnowledgeBaseSkill` wraps the retrieval tool and auto-injects instructions into the agent's system prompt on how and when to use it.

```python
from syndicate.agents import GenericAgent
from syndicate.clients import GeminiClient
from syndicate.skills import KnowledgeBaseSkill

async def create_hr_agent(vector_store):
    client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"), model_name="gemini-1.5-pro")
    
    agent = GenericAgent(
        name="HR Assistant",
        llm_client=client,
        system_prompt="You are a helpful HR assistant. Answer questions accurately."
    )
    
    # Bundle the tool and prompt instructions into a single Skill
    kb_skill = KnowledgeBaseSkill(
        vector_store=vector_store,
        top_k=4,
        use_hybrid=True, # Combines Vector + BM25 keyword search using RRF
        domain="company HR policies",
        additional_instructions="Always search the knowledge base for policy questions and cite your sources."
    )
    
    # Install the skill dynamically
    agent.install_skill(kb_skill)
    
    return agent
```

### 3.1 Customizing KnowledgeBaseSkill Instructions (Localized or Domain-Specific)

You can now customize the internal expertise prompt without subclassing the skill.

- `instructions_template`: Provide custom base instructions (supports `{domain}` placeholder)
- `instructions_mode`: Choose how to apply template (`"replace"` or `"append"`)
- `expertise_builder`: Provide a callable for dynamic prompt generation

```python
from syndicate.skills import KnowledgeBaseSkill

# Option A: replace default generic prompt entirely
skill_es = KnowledgeBaseSkill(
    vector_store=vector_store,
    domain="Ley Federal del Trabajo",
    instructions_template=(
        "Eres un asistente legal enfocado en {domain}. "
        "Siempre cita articulos y fuentes. "
        "Si no hay evidencia suficiente, dilo explicitamente."
    ),
    instructions_mode="replace",
)

# Option B: keep default guidance and append custom policy
skill_append = KnowledgeBaseSkill(
    vector_store=vector_store,
    domain="company HR policies",
    instructions_template="Always answer in Spanish and include source citations.",
    instructions_mode="append",
)

# Option C: generate expertise dynamically
skill_dynamic = KnowledgeBaseSkill(
    vector_store=vector_store,
    domain="internal SOP docs",
    expertise_builder=lambda domain: f"Use only verified evidence from {domain}.",
    instructions_mode="replace",
)
```

### 4. Running the Agent

Now just talk to the agent! It will autonomously query the vector database if it detects its internal knowledge isn't enough.

```python
async def main():
    store = await setup_store()
    await ingest_data(store)
    
    agent = await create_hr_agent(store)
    
    # The agent will decide to use the knowledge base tool here!
    response = await agent.invoke("How much vacation time do I get as a new hire?")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Migrating from BaseRAGMemory

If you used Syndicate's older implementation (`BaseRAGMemory`), building RAG meant passing the memory wrapper explicitly to intercept the prompt:

**Old Way (Deprecated):**
```python
# Magic retrieval injected into prompt context, LLM had no control
agent = GenericAgent(memory=BaseRAGMemory(vector_store)) 
```

**New Way (Agentic):**
```python
# LLM receives a tool and instructions, and calls `search_knowledge_base` itself
skill = KnowledgeBaseSkill(vector_store=vector_store, domain="docs")
agent.install_skill(skill)
```

This Agentic approach drastically improves contextual fidelity and reduces token costs.
