import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from uuid import uuid4

from syndicate.communication_models import Message, ToolCall, ChatResponse
from syndicate.clients.gemini import GeminiClient
from syndicate.clients.openai import OpenAIClient
from syndicate.memory.local import LocalMemory
from pymongo.errors import DuplicateKeyError
from syndicate.memory.mongo import MongoMemory
from syndicate.memory.sqlite_postgres import SqlitePostgresMemory
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


class OpenAIHardeningTests(unittest.TestCase):
    def test_encode_response_logs_and_falls_back_on_malformed_tool_args(self):
        client = OpenAIClient(base_url="http://localhost:9999/v1", api_key="x", model_name="test-model")

        raw_response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "bad_tool",
                                    "arguments": "{not-valid-json",
                                },
                            }
                        ],
                    }
                }
            ]
        }

        with self.assertLogs("syndicate.clients.openai", level="WARNING") as captured:
            response = client._encode_response(raw_response)

        self.assertIsNotNone(response.tool_calls)
        self.assertEqual(response.tool_calls[0].arguments, {})
        self.assertTrue(
            any("Malformed tool call arguments" in message for message in captured.output),
            msg="Expected malformed tool-call warning log was not emitted",
        )

    def test_get_async_client_configures_connection_pool_limits(self):
        client = OpenAIClient(base_url="http://localhost:9999/v1", api_key="x", model_name="test-model")

        with patch("syndicate.clients.openai.httpx.AsyncClient") as async_client_ctor:
            created = async_client_ctor.return_value
            created.is_closed = False

            returned = client._get_async_client()

        self.assertIs(returned, created)
        kwargs = async_client_ctor.call_args.kwargs
        limits = kwargs.get("limits")

        self.assertIsNotNone(limits)
        self.assertEqual(limits.max_connections, 50)
        self.assertEqual(limits.max_keepalive_connections, 10)
        self.assertEqual(limits.keepalive_expiry, 30)


class GeminiClientContractTests(unittest.IsolatedAsyncioTestCase):
    def test_decode_messages_preserves_tool_contract_and_thought_signature(self):
        client = GeminiClient.__new__(GeminiClient)

        messages = [
            Message(role="system", content="You are a test assistant"),
            Message(role="human", content="what's up"),
            Message(
                role="ai",
                content="",
                tool_calls=[
                    ToolCall(
                        id="weather_id",
                        name="weather",
                        arguments={"city": "Tokyo"},
                        thought_signature="signed-thought",
                    )
                ],
            ),
            Message(role="tool", content="NOT_JSON", tool_call_id="weather"),
        ]

        decoded, system_instruction = client._decode_messages(messages)

        self.assertEqual(system_instruction, "You are a test assistant")
        self.assertEqual(decoded[0]["role"], "user")
        self.assertEqual(decoded[0]["parts"][0]["text"], "what's up")
        self.assertEqual(decoded[1]["role"], "model")
        self.assertEqual(
            decoded[1]["parts"][0]["functionCall"]["name"],
            "weather",
        )
        self.assertEqual(
            decoded[1]["parts"][0]["functionCall"]["args"],
            {"city": "Tokyo"},
        )
        self.assertEqual(decoded[1]["parts"][0]["thoughtSignature"], "signed-thought")
        self.assertEqual(decoded[2]["role"], "user")
        self.assertEqual(
            decoded[2]["parts"][0]["functionResponse"]["response"],
            {"result": "NOT_JSON"},
        )

    async def test_chat_completion_async_builds_image_tools_and_thinking_config(self):
        client = GeminiClient.__new__(GeminiClient)
        client.model_name = "gemini-test"
        client.temperature = 0.3

        fake_generate = AsyncMock(return_value=SimpleNamespace())
        client.client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=fake_generate),
            )
        )

        client._encode_response = lambda raw: ChatResponse(content="ok", role="ai")

        class _FakeTool:
            def to_format(self, provider_type):
                self.seen_provider = provider_type
                return {"name": "weather_tool"}

        fake_tool = _FakeTool()

        with patch("syndicate.clients.gemini.types.ThinkingConfig", side_effect=lambda **kw: kw), patch(
            "syndicate.clients.gemini.types.GenerateContentConfig",
            side_effect=lambda **kw: kw,
        ):
            response = await client.chat_completion_async(
                messages=[
                    Message(role="human", content="first"),
                    Message(role="ai", content="middle"),
                    Message(role="human", content="second"),
                ],
                system_message=Message(role="system", content="sys-msg"),
                image="BASE64_IMAGE",
                tools=[fake_tool],
                thinking_level="medium",
            )

        self.assertEqual(response.content, "ok")
        self.assertEqual(fake_tool.seen_provider, "gemini")

        call_kwargs = fake_generate.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-test")
        self.assertEqual(call_kwargs["config"]["system_instruction"], "sys-msg")
        self.assertEqual(call_kwargs["config"]["tools"], [{"name": "weather_tool"}])
        self.assertEqual(call_kwargs["config"]["thinking_config"]["thinking_level"], "medium")

        contents = call_kwargs["contents"]
        self.assertEqual(contents[-1]["role"], "user")
        self.assertEqual(contents[-1]["parts"][0]["text"], "second")
        self.assertEqual(
            contents[-1]["parts"][1],
            {"inline_data": {"mime_type": "image/jpeg", "data": "BASE64_IMAGE"}},
        )

    def test_encode_response_extracts_tool_call_and_thought_signature(self):
        client = GeminiClient.__new__(GeminiClient)

        function_call = SimpleNamespace(name="weather", args={"city": "Tokyo"})
        part = SimpleNamespace(
            text="hi",
            thought="internal-thinking",
            function_call=function_call,
            thought_signature="signature-123",
        )
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            finish_reason="STOP",
        )
        usage = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            thoughts_token_count=2,
        )
        raw = SimpleNamespace(candidates=[candidate], usage_metadata=usage)

        encoded = client._encode_response(raw)

        self.assertEqual(encoded.content, "hi")
        self.assertEqual(encoded.thinking, "internal-thinking")
        self.assertEqual(encoded.finish_reason, "STOP")
        self.assertEqual(encoded.prompt_tokens, 10)
        self.assertEqual(encoded.completion_tokens, 5)
        self.assertEqual(encoded.total_tokens, 15)
        self.assertEqual(encoded.thinking_tokens, 2)
        self.assertIsNotNone(encoded.tool_calls)
        self.assertEqual(encoded.tool_calls[0].name, "weather")
        self.assertEqual(encoded.tool_calls[0].arguments, {"city": "Tokyo"})
        self.assertEqual(encoded.tool_calls[0].thought_signature, "signature-123")


class LocalMemoryBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_rollover_preserves_closed_bucket_when_enabled(self):
        memory = LocalMemory(
            rollover_enabled=True,
            max_interactions_per_bucket=1,
            preserve_closed_buckets=True,
        )

        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")
        await memory.add_message(Message(role="human", content="u2"), "owner", "chat")

        active = await memory.get_active_bucket("owner", "chat")
        self.assertIsNotNone(active)
        self.assertEqual(active.position, 1)
        self.assertEqual([m.content for m in active.messages], ["u2"])

        buckets = memory.get_all_buckets("owner", "chat")
        self.assertEqual(len(buckets), 2)
        closed_positions = sorted(b.position for b in buckets if not b.is_active)
        self.assertEqual(closed_positions, [0])

    async def test_rollover_drops_closed_bucket_when_not_preserving(self):
        memory = LocalMemory(
            rollover_enabled=True,
            max_interactions_per_bucket=1,
            preserve_closed_buckets=False,
        )

        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")
        await memory.add_message(Message(role="human", content="u2"), "owner", "chat")

        buckets = memory.get_all_buckets("owner", "chat")
        self.assertEqual(len(buckets), 1)
        self.assertTrue(buckets[0].is_active)
        self.assertEqual(buckets[0].position, 1)

    async def test_clear_isolated_to_owner_chat(self):
        memory = LocalMemory()

        await memory.add_message(Message(role="human", content="a"), "owner-1", "chat-1")
        await memory.add_message(Message(role="human", content="b"), "owner-2", "chat-2")

        await memory.clear("owner-1", "chat-1")

        self.assertEqual(await memory.get_history("owner-1", "chat-1"), [])
        history_other = await memory.get_history("owner-2", "chat-2")
        self.assertEqual(len(history_other), 1)
        self.assertEqual(history_other[0].content, "b")


class SqlitePostgresMemoryConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_create_bucket_keeps_single_active_bucket(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "memory.db")
            memory = SqlitePostgresMemory(
                database_url=f"sqlite+aiosqlite:///{db_path}",
            )

            try:
                await asyncio.gather(
                    *[memory.create_bucket("owner", "chat") for _ in range(12)],
                )

                buckets = await memory.get_all_buckets("owner", "chat")
                self.assertGreaterEqual(len(buckets), 1)

                active = [b for b in buckets if b.is_active]
                self.assertEqual(
                    len(active),
                    1,
                    msg="Concurrency created more than one active bucket",
                )

                highest_position = max(b.position for b in buckets)
                self.assertEqual(active[0].position, highest_position)
            finally:
                if memory._engine is not None:
                    await memory._engine.dispose()

    async def test_custom_table_name_supports_concurrent_create_bucket(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "memory_custom.db")
            custom_table = f"chat_buckets_{uuid4().hex[:10]}"
            memory = SqlitePostgresMemory(
                database_url=f"sqlite+aiosqlite:///{db_path}",
                table_name=custom_table,
            )

            try:
                await asyncio.gather(
                    *[memory.create_bucket("owner-custom", "chat-custom") for _ in range(8)],
                )

                active = await memory.get_active_bucket("owner-custom", "chat-custom")
                self.assertIsNotNone(active)

                buckets = await memory.get_all_buckets("owner-custom", "chat-custom")
                self.assertGreaterEqual(len(buckets), 1)
                self.assertEqual(len([b for b in buckets if b.is_active]), 1)
            finally:
                if memory._engine is not None:
                    await memory._engine.dispose()


class _FakeMongoCollectionForDuplicateKey:
    def __init__(self, existing_doc):
        self._existing_doc = existing_doc
        self.insert_calls = 0

    async def find_one(self, query, sort=None):
        if query.get("is_active") is True:
            return self._existing_doc
        return None

    async def update_many(self, query, update):
        return None

    async def insert_one(self, doc):
        self.insert_calls += 1
        raise DuplicateKeyError("duplicate active bucket")


class MongoMemoryConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_bucket_returns_existing_bucket_on_duplicate_key(self):
        existing_doc = {
            "bucket_id": "existing-bucket",
            "owner_id": "owner",
            "chat_id": "chat",
            "messages": [],
            "summary": None,
            "is_active": True,
            "position": 3,
        }
        fake_collection = _FakeMongoCollectionForDuplicateKey(existing_doc)

        memory = MongoMemory.__new__(MongoMemory)
        memory._collection = fake_collection
        memory._indexes_created = True
        memory.preserve_closed_buckets = True

        async def _noop_indexes():
            return None

        memory._ensure_indexes = _noop_indexes

        bucket = await memory.create_bucket("owner", "chat")

        self.assertEqual(fake_collection.insert_calls, 1)
        self.assertEqual(bucket.bucket_id, "existing-bucket")
        self.assertTrue(bucket.is_active)
        self.assertEqual(bucket.position, 3)


class ConcurrentDelegationTests(unittest.IsolatedAsyncioTestCase):
    """Test delegation isolation under concurrent load."""

    async def test_delegation_isolation_under_concurrency(self):
        """Verify unique (owner_id, chat_id) pairs under concurrent delegation."""
        fake_agent = _FakeDelegatedAgent()
        tool = AgentAsTool(fake_agent)

        # Fire 10 concurrent delegations
        results = await asyncio.gather(
            *[tool.run_async(task=f"Task {i}") for i in range(10)],
        )

        self.assertEqual(len(fake_agent.calls), 10)
        self.assertEqual(len(results), 10)

        # Extract (owner_id, chat_id) pairs
        ids = [(call[1], call[2]) for call in fake_agent.calls]

        # All pairs must be unique
        self.assertEqual(len(ids), len(set(ids)),
                         msg="Concurrent delegations produced duplicate isolation IDs")

        # Verify all IDs use delegation prefix
        for owner_id, chat_id in ids:
            self.assertTrue(owner_id.startswith("delegation:"))
            self.assertTrue(chat_id.startswith("delegation:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
