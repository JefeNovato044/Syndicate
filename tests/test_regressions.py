import asyncio
import json
import unittest

from syndicate.communication_models import Message
from syndicate.clients.openai import OpenAIClient
from syndicate.mcp import MCPSubTool
from syndicate.tools.agent_tool import AgentAsTool


class _FakeDelegatedAgent:
    def __init__(self):
        self.calls = []

    @property
    def name(self):
        return "FakeDelegatedAgent"

    async def invoke(self, full_message, owner_id="default", chat_id="default"):
        self.calls.append((full_message, owner_id, chat_id, "async"))
        return "ok"

    def invoke_sync(self, full_message, owner_id="default", chat_id="default"):
        self.calls.append((full_message, owner_id, chat_id, "sync"))
        return "ok"


class _FakeMCPResultWithDump:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode="json"):
        return self._payload


class _FakeMCPSession:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def call_tool(self, tool_name, arguments):
        if self._error:
            raise self._error
        return self._result


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncHTTPClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, method, url, json=None):
        return _FakeStreamResponse(self._lines)


class DelegationIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegation_defaults_to_unique_memory_ids(self):
        fake_agent = _FakeDelegatedAgent()
        tool = AgentAsTool(fake_agent)

        await tool.run_async(task="Task A")
        await tool.run_async(task="Task B")

        self.assertEqual(len(fake_agent.calls), 2)
        _, owner_a, chat_a, _ = fake_agent.calls[0]
        _, owner_b, chat_b, _ = fake_agent.calls[1]

        self.assertTrue(owner_a.startswith("delegation:"))
        self.assertTrue(chat_a.startswith("delegation:"))
        self.assertTrue(owner_b.startswith("delegation:"))
        self.assertTrue(chat_b.startswith("delegation:"))
        self.assertNotEqual(owner_a, owner_b)
        self.assertNotEqual(chat_a, chat_b)

    async def test_delegation_respects_explicit_memory_ids(self):
        fake_agent = _FakeDelegatedAgent()
        tool = AgentAsTool(fake_agent)

        await tool.run_async(task="Task", owner_id="tenant-1", chat_id="chat-42")

        self.assertEqual(len(fake_agent.calls), 1)
        _, owner_id, chat_id, _ = fake_agent.calls[0]
        self.assertEqual(owner_id, "tenant-1")
        self.assertEqual(chat_id, "chat-42")


class MCPSubToolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_subtool_preserves_structured_payload(self):
        payload = {"content": [{"type": "text", "text": "hello"}], "isError": False}
        session = _FakeMCPSession(result=_FakeMCPResultWithDump(payload))
        tool = MCPSubTool(
            tool_name="example",
            server_prefix="srv",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            session=session,
        )

        result = await tool.run_async(x=1)
        self.assertEqual(json.loads(result), payload)

    async def test_mcp_subtool_raises_runtime_error_on_failure(self):
        session = _FakeMCPSession(error=ValueError("boom"))
        tool = MCPSubTool(
            tool_name="example",
            server_prefix="srv",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            session=session,
        )

        with self.assertRaises(RuntimeError) as ctx:
            await tool.run_async(x=1)
        self.assertIn("MCP tool 'example' failed", str(ctx.exception))


class OpenAIStreamingFlushTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_flushes_tool_calls_without_done_marker(self):
        client = OpenAIClient(base_url="http://localhost:9999/v1", api_key="x", model_name="test-model")

        # One delta with partial tool-call arguments, no final [DONE] marker.
        delta_line = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"my_tool","arguments":"{\\\"city\\\": \\\"Tokyo\\\"}"}}]},'
            '"finish_reason":null}]}'
        )
        fake_http = _FakeAsyncHTTPClient(lines=[delta_line])
        client._get_async_client = lambda: fake_http

        chunks = []
        async for chunk in client.chat_completion_stream(
            messages=[Message(role="human", content="hi")],
            system_message=Message(role="system", content="sys"),
            tools=None,
        ):
            chunks.append(chunk)

        self.assertGreaterEqual(len(chunks), 1)
        terminal = chunks[-1]
        self.assertTrue(terminal.is_finished)
        self.assertIsNotNone(terminal.tool_calls)
        self.assertEqual(len(terminal.tool_calls), 1)
        self.assertEqual(terminal.tool_calls[0].name, "my_tool")
        self.assertEqual(terminal.tool_calls[0].arguments.get("city"), "Tokyo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
