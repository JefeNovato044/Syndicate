# Syndicate

> **Pre-release (Beta)**: This framework is in active development. APIs may change before stable 1.0.

> A modular, plug-and-play AI agent framework inspired by neural skill implants. Build intelligent agents by simply plugging in components—no manual configuration required.

## Philosophy

**"Just plug in. No wiring needed."**

Syndicate embraces a motherboard design philosophy where components auto-configure based on what's connected. The core inspiration comes from Cyberpunk 2077 neural chips and cyberware systems: abilities are installed, upgraded, and combined without rebuilding the whole body.

That idea maps directly to the framework design: skills, tools, memory backends, and model providers can be plugged in or swapped with minimal friction. You compose capabilities like a cyberware loadout while keeping a stable core runtime.

The framework prioritizes:
- **Zero-boilerplate setup** - Components auto-detect and configure themselves
- **Modularity** - Swap components in/out at runtime
- **Provider-agnostic** - Works with any LLM provider (Gemini, OpenAI, Ollama, LM Studio, vLLM, …)
- **Extensibility** - Add skills, tools, and memory implementations easily

## Requirements

- **Python 3.10+** (Compatible with most modern AI/ML libraries)
- **Recommended runtime: Python 3.11**
- LLM API Key (OpenAI, Gemini) or local provider (Ollama)

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Optional Extras

| Use Case | Install Command |
|----------|-----------------|
| PostgreSQL memory backend | `pip install "syndicate[postgres]"` |
| RAG with local embeddings (`sentence-transformers`) | `pip install "syndicate[rag]"` |
| RAG with OpenAI embeddings | `pip install "syndicate[rag,embeddings-openai]"` |
| RAG with Gemini embeddings | `pip install "syndicate[rag,embeddings-gemini]"` |
| RAG with Cohere embeddings | `pip install "syndicate[rag,embeddings-cohere]"` |
| All RAG extras | `pip install "syndicate[all-rag]"` |

## Quick Start

### 1-Minute Example

```python
import asyncio
import os
from syndicate.agents import GenericAgent
from syndicate.clients.openai import OpenAIClient

async def main():
    # OpenAI cloud example (for local providers, change base_url/model accordingly)
    client = OpenAIClient(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name="gpt-4o"
    )
    agent = GenericAgent(llm_client=client)
    
    response = await agent.invoke("Hello, Syndicate!")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Design

### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BaseAgent                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   LLM Client │  │    Memory    │  │    Skills        │   │
│  │  (auto-detect)│  │  (short-term)│  │  (domain expert) │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │    Tools     │  │   Registry   │  │  Vision/Audio    │   │
│  │ (provider-   │  │  (discovery) │  │  (optional)      │   │
│  │  agnostic)   │  └──────────────┘  └──────────────────┘   │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

## Production Hardening

Syndicate is built for production reliability. Key resilience and observability features are documented in detail:

- **[Resilience & Observability](docs/production-hardening/resilience-and-observability.md)**
  - Tool Execution Policies (Timeout, Retries, Backoff)
  - Lifecycle Hooks and Observers
  - Structured Logging

- **[Multi-Agent Discovery](docs/production-hardening/multi-agent-discovery.md)**
  - Agent2Agent (A2A) v1.0.0 Manifests
  - Capability Discovery (Skills, Interfaces)
  - Interoperability with Swarms (FR-005)

- **[Memory Rollback & Regeneration](docs/guides/memory-rollback.md)**
  - Soft vs Hard Deletion
  - Regenerating Agent responses
  - Backend support (Mongo, SQL, Local)

- **Previous Hardening Notes**
  - Per-request runtime snapshots in `BaseAgent`
  - Single-active-bucket guarantees in persistent memory
  - OpenAI tool argument parsing hardening

### Component Layers

| Layer | Purpose | Examples |
|-------|---------|----------|
| **LLM Client** | Provider abstraction | `GeminiClient`, `OpenAIClient` |
| **Memory** | Conversation storage | `LocalMemory`, `MongoMemory`, `SqlitePostgresMemory` |
| **Tools** | Function execution | `BaseTool`, `CurrentWeatherTool`, `AgentAsTool` |
| **Skills** | Domain expertise | `SkillModule` |
| **Agents** | Orchestration | `BaseAgent`, `GenericAgent` |

### Key Design Patterns

1. **Auto-Detection** - `BaseAgent._detect_provider()` automatically identifies LLM providers from client attributes or class names

2. **Provider-Agnostic Tools** - `BaseTool.to_format()` converts tools to provider-specific formats (Gemini, OpenAI, LangChain)

3. **Skill Injection** - Skills are sorted by priority and injected into the system prompt at runtime via `install_skill()`

4. **Agent-to-Agent Delegation** - Agents can be wrapped as tools for other agents using `AgentAsTool(agent)`

5. **Bucket-Based Memory** - Long conversations are segmented into buckets with automatic summarization on rollover

## Testing

Run the regression suite to validate concurrency and client hardening behavior:

```bash
python -m unittest tests/test_regressions.py -v
```

This suite includes coverage for:
- delegation memory isolation (including concurrent delegation)
- MCP sub-tool contract behavior
- OpenAI streaming tool-call flush behavior
- OpenAI malformed tool-argument fallback + warning logs
- OpenAI HTTP connection pool limit configuration
- Gemini message/response contract behavior
- LocalMemory rollover and tenant isolation behavior
- SqlitePostgresMemory concurrent bucket-creation safety
- SqlitePostgresMemory custom `table_name` concurrency path
- MongoMemory duplicate-key race fallback behavior

## Troubleshooting

### Missing API Key or Auth Errors

- Ensure `.env` is loaded when using API providers.
- Verify provider-specific variables such as `GEMINI_API_KEY` or OpenAI-compatible credentials.

### Local OpenAI-Compatible Provider Fails

- Confirm `base_url` includes `/v1` for providers that require OpenAI-compatible paths.
- For local servers like Ollama/LM Studio, verify host/port and model name are available.

### SQLite/PostgreSQL Memory Connection Errors

- SQLite URL must include async driver: `sqlite+aiosqlite:///./chat_history.db`.
- PostgreSQL URL must use asyncpg: `postgresql+asyncpg://user:password@host:5432/dbname`.

### MongoDB Memory Errors

- Check `MONGO_URI` connectivity and database permissions.
- Validate network access from your runtime environment to the MongoDB cluster.

## Usage Patterns

### Declaring a Client

Syndicate ships two built-in clients. Both are drop-in replaceable.

```python
from syndicate.clients.gemini import GeminiClient
from syndicate.clients.openai import OpenAIClient
from dotenv import load_dotenv
import os

load_dotenv()

# Google Gemini
gemini_client = GeminiClient(
    model_name="gemini-2.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")
)

# OpenAI-compatible (OpenAI, Ollama, LM Studio, vLLM, …)
openai_client = OpenAIClient(
    model_name="llama3",
    base_url="http://localhost:11434/v1",   # Ollama example
    api_key="ollama"                         # any string for local servers
)
```

## Example Index

Use these runnable examples as guided entry points:

| Example | File | Focus |
|---------|------|-------|
| Basic async invocation | `examples/async_usage_example.py` | Minimal agent usage |
| OpenAI-compatible providers | `examples/openai_usage_example.py` | OpenAI, Ollama, LM Studio-style clients |
| Runtime facade for services | `examples/runtime_interface_example.py` | `AgentInterface` pattern for web backends |
| Custom multi-agent setup | `examples/custom_agent_example.py` | Specialization and composition |
| SQLite persistent memory | `examples/sqlite_memory_example.py` | Bucketed persistence with SQL |
| RAG knowledge base flow | `examples/rag_knowledge_base_example.py` | Retrieval-augmented workflows |
| Streaming output check | `examples/streaming_test.py` | Stream chunk behavior |

## Examples

### 1. Simple Agent with Memory

```python
from syndicate.agents import GenericAgent
from syndicate.clients.openai import OpenAIClient
from syndicate.memory import LocalMemory

client = OpenAIClient(model_name="llama3", base_url="http://localhost:11434/v1")

agent = GenericAgent(
    llm_client=client,
    system_prompt="You are a helpful assistant.",
    memory=LocalMemory(rollover_enabled=False)
)

# Async
response = await agent.invoke("Hello!")

# Sync (outside async contexts)
response = agent.invoke_sync("Hello!")
print(response)
```

### 2. Streaming with Thinking

```python
async for chunk in agent.stream("Who are you?", include_thinking=True):
    if chunk.thinking:
        print(chunk.thinking, end="", flush=True)  # reasoning traces
    if chunk.content:
        print(chunk.content, end="", flush=True)   # final response
```

### 3. Agent with Tools

```python
from syndicate.agents import GenericAgent
from syndicate.clients.gemini import GeminiClient
from syndicate.tools.weather_tool import CurrentWeatherTool

client = GeminiClient(model_name="gemini-2.0-flash-exp", api_key="...")

agent = GenericAgent(
    llm_client=client,
    system_prompt="You are a weather assistant.",
    tools=[CurrentWeatherTool(api_key="...")]
)

response = await agent.invoke("What's the weather in Tokyo?")
print(response)
```

### 4. Agent with Skills

```python
from syndicate.agents import GenericAgent
from syndicate.clients.openai import OpenAIClient
from syndicate.skills.examples import create_git_skill, create_python_skill

client = OpenAIClient(model_name="llama3", base_url="http://localhost:11434/v1")

agent = GenericAgent(
    llm_client=client,
    system_prompt="You are a helpful assistant.",
    skills=[create_python_skill()]
)

# Install additional skills at runtime
agent.install_skill(create_git_skill())

response = await agent.invoke("What are your capabilities?")
print(response)
```

### 5. Agent-to-Agent Delegation

Wrap specialist agents as tools so a coordinator can delegate work to them.

```python
from syndicate.agents import GenericAgent
from syndicate.clients.openai import OpenAIClient
from syndicate.memory import LocalMemory
from syndicate.tools import AgentAsTool

client = OpenAIClient(model_name="llama3", base_url="http://localhost:11434/v1")

# Specialist agents
math_agent = GenericAgent(
    llm_client=client,
    name="MathExpert",
    system_prompt="You are a math expert. Solve problems step by step.",
    memory=LocalMemory(rollover_enabled=False),
)

translator_agent = GenericAgent(
    llm_client=client,
    name="Translator",
    system_prompt="You are a professional translator.",
    memory=LocalMemory(rollover_enabled=False),
)

# Coordinator delegates to specialists via AgentAsTool
coordinator = GenericAgent(
    llm_client=client,
    name="Coordinator",
    system_prompt=(
        "You are a coordinator. Delegate math tasks to MathExpert "
        "and translation tasks to Translator."
    ),
    tools=[AgentAsTool(math_agent), AgentAsTool(translator_agent)],
    memory=LocalMemory(rollover_enabled=False),
)

response = await coordinator.invoke("What is 142 * 37? Then translate the result to French.")
print(response)
```

### 6. Shared Memory Pool

Multiple agents can share the same `LocalMemory` instance, giving each agent visibility into conversations started by the others.

```python
from syndicate.agents import GenericAgent
from syndicate.memory import LocalMemory

shared_memory = LocalMemory(rollover_enabled=False)

agent_a = GenericAgent(llm_client=client, system_prompt="You are Agent A.", memory=shared_memory)
agent_b = GenericAgent(llm_client=client, system_prompt="You are a data analyst.", memory=shared_memory)

await agent_a.invoke("My name is Alice and I work on AI.")
await agent_b.invoke("Who are you and what is the conversation about?")
```

### 7. MongoDB Memory with Summarization

`MongoMemory` provides persistent, multi-tenant storage with automatic bucket rollover and summarization.

```python
from pymongo import AsyncMongoClient
from syndicate.agents import GenericAgent
from syndicate.clients.gemini import GeminiClient
from syndicate.memory import MongoMemory
from syndicate.memory.summarizers import create_default_summarizer
import os

llm_client   = GeminiClient(model_name="gemini-2.5-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))
mongo_client = AsyncMongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
database     = mongo_client["my_app"]

# Any LLM client can act as the summarizer
summarizer = create_default_summarizer(llm_client)

mongo_memory = MongoMemory(
    database=database,
    collection_name="chat_history",
    rollover_enabled=True,
    max_interactions_per_bucket=3,   # summarize every 3 exchanges
    summarizer=summarizer,
    preserve_closed_buckets=True     # keep full history in DB
)

agent = GenericAgent(
    llm_client=llm_client,
    system_prompt="You are a helpful assistant with persistent memory.",
    memory=mongo_memory
)

response = await agent.invoke("Hi! My name is Jose.")
print(response)

# Inspect memory statistics
stats = await mongo_memory.get_stats()
print(f"Buckets: {stats['total_buckets']} total, {stats['active_buckets']} active")

# Browse buckets
buckets = await mongo_memory.get_all_buckets("default", "default")
for bucket in buckets:
    print(f"  position={bucket.position} active={bucket.is_active} msgs={len(bucket.messages)}")
    if bucket.summary:
        print(f"  summary: {bucket.summary[:80]}…")

# Clear a conversation
await mongo_memory.clear("default", "default")
```

### 8. SQLite/PostgreSQL Memory with Summarization

`SqlitePostgresMemory` provides persistent, multi-tenant storage using SQLite or PostgreSQL with automatic bucket rollover and summarization.

**Installation:**

```bash
# For SQLite support (included by default)
uv add sqlalchemy[asyncio] aiosqlite

# For PostgreSQL support
uv add --optional postgres sqlalchemy[asyncio] asyncpg
```

**SQLite Usage:**

```python
from syndicate.agents import GenericAgent
from syndicate.clients.gemini import GeminiClient
from syndicate.memory import SqlitePostgresMemory
from syndicate.memory.summarizers import create_default_summarizer
import os

llm_client = GeminiClient(model_name="gemini-2.5-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))

# SQLite - file-based database
sqlite_memory = SqlitePostgresMemory(
    database_url="sqlite+aiosqlite:///./chat_history.db",
    table_name="chat_buckets",
    rollover_enabled=True,
    max_interactions_per_bucket=5,
    summarizer=create_default_summarizer(llm_client),
    preserve_closed_buckets=True
)

agent = GenericAgent(
    llm_client=llm_client,
    system_prompt="You are a helpful assistant with persistent memory.",
    memory=sqlite_memory
)

response = await agent.invoke("Hi! My name is Carlos.")
print(response)

# Inspect memory statistics
stats = await sqlite_memory.get_stats()
print(f"Buckets: {stats['total_buckets']} total, {stats['active_buckets']} active")

# Clear a conversation
await sqlite_memory.clear("default", "default")
```

**PostgreSQL Usage:**

```python
from syndicate.memory import SqlitePostgresMemory

# PostgreSQL - server-based database
postgres_memory = SqlitePostgresMemory(
    database_url="postgresql+asyncpg://user:password@localhost:5432/mydb",
    table_name="chat_buckets",
    rollover_enabled=True,
    max_interactions_per_bucket=10,
    summarizer=create_default_summarizer(llm_client)
)

agent = GenericAgent(
    llm_client=llm_client,
    system_prompt="You are a helpful assistant.",
    memory=postgres_memory
)
```

### 9. Multi-Tenant Memory

```python
# Different owners/sessions share the same memory store but stay isolated
await agent.invoke("My name is Alice", owner_id="alice", chat_id="session1")
await agent.invoke("My name is Bob",   owner_id="bob",   chat_id="session1")

history = await agent.get_history(owner_id="alice", chat_id="session1")
```

## API Reference

### BaseAgent

Core agent class. Implements a Template Method Pattern with a Hybrid API.

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_client` | `Client` | LLM provider client (required) |
| `system_prompt` | `str` | Agent instructions |
| `memory` | `BaseChatMemory` | Conversation storage |
| `tools` | `list` | Tools the agent can call |
| `skills` | `list[SkillModule]` | Domain expertise modules |
| `name` | `str` | Agent identifier |
| `max_iterations` | `int` | Max tool-calling iterations (default `5`) |
| `verbose` | `bool` | Enable debug logging |

**Key methods:**

| Method | Description |
|--------|-------------|
| `await invoke(input, owner_id, chat_id)` | Main async interaction |
| `invoke_sync(input, owner_id, chat_id)` | Blocking version for sync contexts |
| `stream(input, include_thinking)` | Async generator yielding `StreamChunk` |
| `install_skill(skill)` | Add a `SkillModule` at runtime (chainable) |
| `await get_history(owner_id, chat_id)` | Retrieve raw conversation history |

### Operational Interface (Web/Service Runtime)

For production web services, depend on the minimal operational contract rather than the full mutable agent API.

```python
from syndicate import AgentInterface
from syndicate.agents import GenericAgent

agent = GenericAgent(...)

# Runtime-only facade exposing invoke/stream/invoke_sync
runtime: AgentInterface = agent.as_runtime()
result = await runtime.invoke("Hello", owner_id="tenant", chat_id="session")
```

This helps separate setup-time mutators (`install_skill`, `add_tool`, `set_system_prompt`, etc.) from request-time execution in concurrent services.

#### Why this matters for web backends

- Clear dependency boundary: handlers/services can accept `AgentInterface` instead of `BaseAgent`.
- Runtime safety by design: `AgentRuntime` exposes only `invoke`, `stream`, and `invoke_sync`.
- Easier testing: mock a tiny interface instead of constructing full agent graphs.

#### FastAPI-style pattern

```python
from fastapi import Depends, FastAPI
from syndicate import AgentInterface
from syndicate.agents import GenericAgent

app = FastAPI()

def get_runtime() -> AgentInterface:
    # Build once at startup in real apps; simplified here for readability.
    agent = GenericAgent(...)
    return agent.as_runtime()

@app.post("/chat")
async def chat(payload: dict, runtime: AgentInterface = Depends(get_runtime)):
    text = await runtime.invoke(
        payload["message"],
        owner_id=payload.get("owner_id", "default"),
        chat_id=payload.get("chat_id", "default"),
    )
    return {"response": text}
```

See `examples/runtime_interface_example.py` for a framework-agnostic runtime wrapper and mock-friendly pattern.

### GenericAgent

Zero-boilerplate agent, inherits everything from `BaseAgent`. Ideal for prototypes, interactive notebooks, and quick experiments.

```python
from syndicate.agents import GenericAgent

agent = GenericAgent(
    llm_client=client,
    system_prompt="Your instructions",
    tools=[...],
    skills=[...],
    memory=memory
)
```

### LLM Clients

#### GeminiClient

```python
from syndicate.clients.gemini import GeminiClient

# API key
client = GeminiClient(model_name="gemini-2.5-flash-lite", api_key="...")

# Vertex AI (service account)
client = GeminiClient(
    model_name="gemini-1.5-pro",
    service_account_credentials="/path/to/sa.json",
    project="my-gcp-project",
    location="us-central1"
)
```

#### OpenAIClient

Supports any OpenAI-compatible endpoint — cloud or local.

```python
from syndicate.clients.openai import OpenAIClient

# Ollama
client = OpenAIClient(base_url="http://localhost:11434/v1", model_name="llama3", api_key="ollama")

# LM Studio
client = OpenAIClient(base_url="http://localhost:1234/v1", model_name="mistral")

# OpenAI cloud
client = OpenAIClient(base_url="https://api.openai.com/v1", api_key="sk-...", model_name="gpt-4o")
```

### Memory

Conversation memory abstraction:

1. **`BaseChatMemory`** - Sequential conversation history with bucket-based rollover and optional summarization

| Implementation | Storage | Persistence | Notes |
|---------------|---------|-------------|-------|
| `LocalMemory` | In-process dict | ❌ | Default; great for tests and notebooks |
| `MongoMemory` | MongoDB | ✅ | Multi-tenant, with rollover & summarization |
| `SqlitePostgresMemory` | SQLite/PostgreSQL | ✅ | Persistent relational backend with rollover & summarization |

For semantic retrieval (RAG), use the vector store layer (`BaseVectorStore`, `MongoVectorStore`) with tools/skills such as `RAGSearchTool` and `KnowledgeBaseSkill`.

`KnowledgeBaseSkill` supports prompt customization without subclassing via `instructions_template`, `instructions_mode` (`replace` or `append`), and `expertise_builder` for dynamic domain-specific instructions. See `docs/guides/agentic-rag.md` for examples.

### Skills (SkillModule)

Inject domain expertise into agents via structured system-prompt sections.

```python
from syndicate.skills import create_skill_module

skill = create_skill_module(
    name="Kubernetes Expert",
    description="Deep expertise in Kubernetes cluster management",
    expertise="You know deployments, pods, services, HPA, secrets ...",
    capabilities=["Diagnose pod issues", "Review manifests", "Optimize resources"],
    glossary={"HPA": "Horizontal Pod Autoscaler", "VPA": "Vertical Pod Autoscaler"},
    priority=5   # higher priority skills are injected first
)

agent.install_skill(skill)
```

Built-in example skills live in `syndicate.skills.examples`:

```python
from syndicate.skills.examples import (
    create_kubernetes_skill,
    create_python_skill,
    create_git_skill,
)
```

### Tools (BaseTool)

Define tools with a Pydantic schema for automatic validation and provider-format conversion.

```python
from syndicate.tools.base_tool import BaseTool
from pydantic import BaseModel, Field

class MyArgs(BaseModel):
    query: str = Field(..., description="Search query")

class MyTool(BaseTool):
    name = "web_search"
    description = "Search the web for information."
    args_schema = MyArgs

    def run(self, **kwargs) -> str:
        return f"Results for: {kwargs['query']}"
```

### AgentAsTool

Wraps any agent as a callable tool for hierarchical / multi-agent architectures.

```python
from syndicate.tools import AgentAsTool

tool = AgentAsTool(specialist_agent)
# tool.name  →  "delegate_to_<agent_name>"
coordinator = GenericAgent(llm_client=client, tools=[tool], ...)
```

## Project Structure

```
src/syndicate/
├── agents/
│   ├── base.py          # BaseAgent (motherboard design, Template Method)
│   └── generic.py       # GenericAgent (zero-boilerplate)
├── clients/
│   ├── base.py          # Client ABC
│   ├── gemini.py        # GeminiClient (API key + Vertex AI)
│   └── openai.py        # OpenAIClient (OpenAI, Ollama, LM Studio, vLLM)
├── memory/
│   ├── base.py          # BaseChatMemory ABC + rollover/summarization logic
│   ├── local.py         # LocalMemory (in-process)
│   ├── mongo.py         # MongoMemory (persistent, multi-tenant)
│   ├── sqlite_postgres.py # SqlitePostgresMemory (SQLite/PostgreSQL)
│   └── summarizers.py   # Bucket summarization utilities
├── skills/
│   ├── skill_module.py  # SkillModule + create_skill_module()
│   ├── examples.py      # Built-in skill examples
│   ├── rag_skill.py     # KnowledgeBaseSkill
│   ├── elasticsearch_skill.py # Elasticsearch domain skill
│   └── registry.py      # SkillRegistry
├── tools/
│   ├── base_tool.py     # BaseTool abstraction
│   ├── agent_tool.py    # AgentAsTool wrapper
│   ├── rag_tool.py      # RAG retrieval tool
│   └── weather_tool.py  # CurrentWeatherTool (example)
├── mcp.py                   # MCPSessionManager, MCPSubTool
├── communication_models.py  # Message, ToolCall, StreamChunk, …
└── registry.py              # AgentRegistry
```

## MCP Integration

Syndicate has built-in support for MCP (Model Context Protocol) servers, enabling agents to discover and use tools from external services without any custom glue code.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Syndicate Agent                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │  LLM Client │  │   Memory    │  │       Skills         │  │
│  └─────────────┘  └─────────────┘  └──────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         MCPSessionManager (Persistent Sessions)      │     │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────────┐   │     │
│  │  │ Filesystem │  │ PostgreSQL │  │    GitHub     │   │     │
│  │  └────────────┘  └────────────┘  └───────────────┘   │     │
│  └──────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose | Module |
|-----------|---------|--------|
| `MCPSessionManager` | Registers servers, owns long-lived sessions, discovers sub-tools | `syndicate.mcp` |
| `MCPSubTool` | One discovered MCP sub-tool exposed as a normal Syndicate tool | `syndicate.mcp` |

### Quick Start

```python
from syndicate.mcp import MCPSessionManager
from syndicate.agents import GenericAgent

mgr = MCPSessionManager()
mgr.register(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
)

# Start sessions and discover tools once
await mgr.start()

# Attach discovered MCP tools to an agent
agent = GenericAgent(
    llm_client=client,
    tools=mgr.get_tools("filesystem"),
)

# ... use agent ...

# Shutdown cleanup
await mgr.close()
```

Note: there is currently no dedicated MCP example in `examples/`; use this section as the reference pattern.

## Roadmap

The canonical roadmap lives in `ROADMAP.md`.

Current highlights:
- [x] MongoDB + SQLite/PostgreSQL persistent memory with bucket rollover and summarization
- [x] MCP integration via persistent `MCPSessionManager` sessions and discovered `MCPSubTool`s
- [x] OpenAI-compatible and Gemini client support with hardened streaming/tool-call parsing
- [ ] Telemetry middleware for token/cost observability
- [ ] Native Anthropic client support
- [ ] Sensorial foundation + vision/speech/audio adapters
- [ ] Elasticsearch memory and vector store

## License

MIT License - see LICENSE file for details
