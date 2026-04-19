import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional


class LoggingObserver:
    """Observer that emits lifecycle events through the Python logging stack."""

    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self._logger = logger or logging.getLogger("syndicate.observability")
        self._level = level

    async def on_request_start(self, event: Dict[str, Any]) -> None:
        self._log("request_start", event)

    async def on_request_end(self, event: Dict[str, Any]) -> None:
        self._log("request_end", event)

    async def on_model_call_start(self, event: Dict[str, Any]) -> None:
        self._log("model_call_start", event)

    async def on_model_call_end(self, event: Dict[str, Any]) -> None:
        self._log("model_call_end", event)

    async def on_tool_call_start(self, event: Dict[str, Any]) -> None:
        self._log("tool_call_start", event)

    async def on_tool_call_end(self, event: Dict[str, Any]) -> None:
        self._log("tool_call_end", event)

    async def on_error(self, event: Dict[str, Any]) -> None:
        self._logger.warning("event=error payload=%s", event)

    def _log(self, event_name: str, event: Dict[str, Any]) -> None:
        self._logger.log(self._level, "event=%s payload=%s", event_name, event)


class InMemoryObserver:
    """Observer that stores hook payloads in memory for tests and lightweight apps."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    async def on_request_start(self, event: Dict[str, Any]) -> None:
        self._record("on_request_start", event)

    async def on_request_end(self, event: Dict[str, Any]) -> None:
        self._record("on_request_end", event)

    async def on_model_call_start(self, event: Dict[str, Any]) -> None:
        self._record("on_model_call_start", event)

    async def on_model_call_end(self, event: Dict[str, Any]) -> None:
        self._record("on_model_call_end", event)

    async def on_tool_call_start(self, event: Dict[str, Any]) -> None:
        self._record("on_tool_call_start", event)

    async def on_tool_call_end(self, event: Dict[str, Any]) -> None:
        self._record("on_tool_call_end", event)

    async def on_error(self, event: Dict[str, Any]) -> None:
        self._record("on_error", event)

    def _record(self, hook: str, event: Dict[str, Any]) -> None:
        self.events.append({"hook": hook, "event": deepcopy(event)})

    def get_events(self, hook: Optional[str] = None) -> List[Dict[str, Any]]:
        if hook is None:
            return [deepcopy(entry) for entry in self.events]
        return [deepcopy(entry["event"]) for entry in self.events if entry.get("hook") == hook]

    def clear(self) -> None:
        self.events.clear()