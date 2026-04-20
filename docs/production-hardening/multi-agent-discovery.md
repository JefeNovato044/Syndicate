# Multi-Agent Discovery (Agent2Agent)

Documentation for Syndicate's implementation of the **Agent2Agent (A2A) v1.0.0** protocol.

## Overview (FR-005)

Syndicate agents include native support for the A2A protocol, allowing them to broadcast their capabilities in a standardized human-and-machine-readable format called an **Agent Card**. 

This card enables:
- **Automatic Discovery**: In secondary swarm networks or multi-agent environments.
- **Interoperability**: Connecting Syndicate agents to agents built with other A2A-compliant frameworks.
- **Dynamic Skill Mapping**: Exporting internal tools and skills as structured capability descriptions.

---

## The Agent Card (Manifest)

The `AgentCard` contains essential metadata about the agent, its communication preferences, and its skills.

### Generating a Manifest

You can generate an A2A-compliant manifest from any `BaseAgent` instance using the `get_manifest()` method:

```python
from syndicate.agents import GenericAgent

agent = GenericAgent(
    name="ResearchBuddy",
    system_prompt="You are an expert researcher.",
    tools=[WebSearchTool(), DatabaseTool()]
)

# Export A2A Agent Card
manifest = agent.get_manifest()

print(f"Agent Name: {manifest.name}")
print(f"Skills: {[s.name for s in manifest.skills]}")
print(f"Streaming Support: {manifest.capabilities.streaming}")

# Get as raw A2A JSON
print(manifest.to_json())
```

---

## Data Models

Syndicate uses Pydantic models in `communication_models.py` to ensure A2A compliance.

### A2ASkill
Each tool installed in the agent is exported as an `A2ASkill`:
- **id**: The technical name of the tool.
- **name**: A formatted, human-readable version of the id.
- **description**: Extracted from the tool's docstring.
- **Input/Output Modes**: Defaults to `application/json`.

### A2ACapabilities
Describes what the agent *can* do:
- `streaming`: Always `True` for Syndicate (native `stream()` support).
- `pushNotifications`: Currently `False` (reserved for future webhook support).

### A2ASupportedInterface
Defines how to reach the agent. While `get_manifest()` populates the skills and capabilities, the `supportedInterfaces` (URLs, protocols) are typically added by the deployment layer (e.g., a FastAPI adapter).

---

## Interaction Model

Syndicate's internal `stream()` and `invoke()` flows are designed to map cleanly to A2A request/response structures, facilitating the use of the **Unified Envelope** (FR-004).
