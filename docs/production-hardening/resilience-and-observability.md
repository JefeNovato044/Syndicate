# Production Hardening

Documentation for Syndicate's resilience and reliability features.

## Tool Execution Policies (FR-001)

Syndicate provides built-in resilience for tool execution through `ToolExecutionPolicy`. This allows you to define timeouts, retries, and backoff strategies per tool or globally.

### Usage

```python
from syndicate.tools import BaseTool, ToolExecutionPolicy, ToolBackoffPolicy

class MyTool(BaseTool):
    # Define a custom policy for this tool
    execution_policy = ToolExecutionPolicy(
        timeout=30.0,
        max_retries=3,
        backoff_policy=ToolBackoffPolicy(
            strategy="exponential",
            base_delay=1.0,
            max_delay=10.0
        )
    )

    async def run(self, **kwargs):
        # Implementation
        pass
```

### Policy Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `60.0` | Maximum seconds to wait for tool execution. |
| `max_retries` | `int` | `0` | Number of times to retry on failure. |
| `backoff_policy` | `ToolBackoffPolicy` | `None` | Retry delay strategy (`fixed` or `exponential`). |

---

## Observability and Lifecycle Hooks (FR-002)

Syndicate agents emit detailed lifecycle events through the `Observer` protocol. This allows you to plug in logging, tracing, or custom monitoring tools without modifying agent logic.

### The Observer Protocol

Implement the `Observer` protocol to receive events:

```python
from syndicate.protocols import Observer, ObserverEvent

class MyCustomObserver(Observer):
    async def on_request_start(self, event: ObserverEvent):
        print(f"Request started: {event.request_id}")

    async def on_model_call_end(self, event: ObserverEvent):
        print(f"Model call took {event.latency_ms}ms")

    async def on_tool_call_start(self, event: ObserverEvent):
        print(f"Tool {event.tool_name} called")
```

### Built-in Observers

Syndicate ships with standard observers:

- `LoggingObserver`: Outputs structured logs to the Python `logging` system.
- `InMemoryObserver`: Captures events in a list, ideal for testing or debugging.

```python
from syndicate.agents import GenericAgent
from syndicate.observability import LoggingObserver

agent = GenericAgent(
    llm_client=client,
    observers=[LoggingObserver(level="INFO")]
)
```

### Event Metadata

Every `ObserverEvent` contains:
- `request_id`: A unique UUID for the current flow.
- `timestamp`: ISO 8601 timestamp.
- `latency_ms`: Execution time (for `end` hooks).
- `usage`: Token usage data (for model calls).
- `flow`: The calling method (`invoke` or `stream`).
- `metadata`: Flexible dictionary for additional context.
