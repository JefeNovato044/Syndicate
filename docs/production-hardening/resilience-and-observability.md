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

---

## Tool Guardrails (FR-003)

Syndicate provides built-in safety mechanisms to prevent runaway tool loops and manage API concurrency. These guardrails are configured at the agent level and apply to all tools.

### Usage

```python
from syndicate.agents import GenericAgent

agent = GenericAgent(
    llm_client=client,
    max_total_tool_calls=10,        # Hard stop after 10 calls per request
    max_concurrent_tool_calls=2,    # Execute tool calls in batches of 2
)
```

### Guardrail Logic

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_total_tool_calls` | `int` | `10` | The absolute maximum number of tool calls allowed per single `invoke()` or `stream()` call. If reached, the interaction stops. |
| `max_concurrent_tool_calls` | `int` | `5` | The maximum number of tool calls that can run in parallel. Syndicate uses **batching** to process calls in chunks of this size. |

### Terminal Signals

When `max_total_tool_calls` is reached:
- **`invoke()`**: Returns the final content received from the model up to that point.
- **`stream()`**: Emits a final chunk with `finish_reason="guardrail_reached"` to signal the abrupt stop.

This prevents the agent from entering infinite "tool-response-tool" loops that consume tokens and API credits.

---

## Unified Tool Result/Error Envelope (FR-004)

Syndicate now stores tool execution outputs in a canonical JSON envelope for both `invoke()` and `stream()` orchestration paths.

### Envelope Schema

```json
{
    "status": "success",
    "result": {"key": "value"},
    "error": null,
    "metadata": {
        "tool_name": "my_tool",
        "attempt": 1,
        "max_attempts": 1,
        "latency_ms": 12.4,
        "error_type": "ValueError"
    }
}
```

### Field Semantics

| Field | Type | Description |
|------|------|-------------|
| `status` | `"success" \| "error"` | Normalized terminal outcome of the tool execution. |
| `result` | `Any \| null` | Tool return payload for successful execution. |
| `error` | `str \| null` | Error message for failed execution. |
| `metadata` | `dict` | Optional diagnostics such as `attempt`, `latency_ms`, and `error_type`. |

### Backward Compatibility

- Existing non-JSON `Message(role="tool")` content is still accepted by provider decoders.
- Legacy plain-text tool content is interpreted as a successful result payload by compatibility fallbacks.

This keeps historical memory ingestion working while making new tool outputs deterministic and easier to parse in prompts, logs, and observability pipelines.
