# 🗺️ Roadmap

The Syndicate core is designed to be the ultimate, lightweight "Motherboard" for agentic workflows. Our roadmap focuses on expanding execution capabilities, memory, and provider agnosticism, while keeping the framework purely headless.

### Phase 1: The Foundation (Current)
- [x] **Provider Auto-Detection:** Seamless switching between Gemini, OpenAI-compatible APIs, and local models.
- [x] **Tool Auto-Formatting:** Write a tool once; the framework formats it for the specific LLM.
- [x] **Skill Modules ("Skill Chips"):** Hot-swappable domain expertise.
- [x] **Basic Delegation:** Agents calling other agents as tools.
- [x] **Core Concurrency Hardening:** Request-local runtime snapshots in `BaseAgent` to prevent cross-request state contamination.
- [x] **Memory State Integrity:** Single-active-bucket constraints and race-safe bucket creation for SQL/Mongo memory backends.
- [x] **Unified Async & Streaming:** Fully asynchronous core with standardized `StreamChunk` yielding across all providers.

### Next Priorities
- [ ] **Telemetry Middleware (MVP first):** Provider-agnostic token usage, cost monitoring, latency metrics, and request-level observability hooks.
- [ ] **Native Anthropic Client Support:** Claude-compatible client with parity for sync/async/streaming, tools, and usage accounting.
- [ ] **Multimodal Sensorial Foundation:** Unified content-part abstraction to support text + image + audio inputs/outputs without provider lock-in.
- [ ] **Vision, Speech, and Audio Client Adapters:** Provider adapters built on the sensorial foundation for seamless multimodal usage.
- [ ] **Elasticsearch Memory + Vector Store:** Persistent searchable memory and vector recall with tenant-safe indexing patterns.

### Phase 2: Advanced Orchestration (The "Collective")
- [ ] **Complex Topologies:** Move beyond simple 1:1 delegation to support Agent Swarms and parallel task execution (handling I/O bound tools concurrently).
- [ ] **Cross-Agent Memory (The Blackboard):** Allow specialized agents to read/write to a shared memory space during complex problem-solving.
- [ ] **State Serialization:** Ability to export and hydrate an exact agent's configuration (installed skills, tools, and memory state) via JSON blueprints.

### Phase 3: Production & Persistence
- [x] **Provider Payload Hardening:** Defensive tool-argument parsing for OpenAI-compatible responses.
- [ ] **Long-Term Memory Stores:** Native integrations for dedicated vector databases and search engines (Elasticsearch, Milvus) to give agents persistent, cross-session recall.
- [ ] **Unified Telemetry:** Provider-agnostic token counting, cost tracking, and rate-limit management.
- [ ] **Anthropic Client Parity:** First-class Claude support with consistent client contract behavior.
- [ ] **Sandboxed Execution:** Safer local execution environments for agents writing and testing their own code.

### Phase 4: The Ecosystem & Observability
- [ ] **Skill Module Registry:** A standard for packaging and sharing custom `SkillModules` across different projects.
- [ ] **Lifecycle Event Hooks:** Comprehensive callback system (`on_agent_start`, `on_tool_execute`, `on_stream_chunk`). *Note: This is the critical infrastructure that will eventually allow external visual builders and UIs to "listen" to the framework without bloating the core code.*
- [ ] **Sensorial Plugin Layer:** Standard extension points for modality-specific clients and preprocessors.