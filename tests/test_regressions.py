import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from uuid import uuid4
from collections.abc import AsyncGenerator

from syndicate.protocols import AgentInterface
from syndicate.agents.runtime import AgentRuntime
from syndicate.agents.base import BaseAgent
from syndicate.communication_models import Message, ToolCall, ChatResponse, StreamChunk
from syndicate.clients.gemini import GeminiClient
from syndicate.clients.openai import OpenAIClient
from syndicate.memory.local import LocalMemory
from pymongo.errors import DuplicateKeyError
from syndicate.memory.mongo import MongoMemory
from syndicate.memory.sqlite_postgres import SqlitePostgresMemory
from syndicate.mcp import MCPSubTool
from syndicate.tools.agent_tool import AgentAsTool
from syndicate.tools.base_tool import _clean_schema_for_gemini


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


class _FakeAgentRuntimeTarget:
    def __init__(self):
        self.calls = []

    async def invoke(self, user_input, owner_id="default", chat_id="default"):
        self.calls.append(("invoke", user_input, owner_id, chat_id))
        return f"echo:{user_input}"

    async def stream(
        self,
        user_input,
        owner_id="default",
        chat_id="default",
        include_thinking=False,
    ) -> AsyncGenerator:
        self.calls.append(("stream", user_input, owner_id, chat_id, include_thinking))
        yield StreamChunk(content="chunk-1", is_finished=False)
        yield StreamChunk(content="", is_finished=True)

    def invoke_sync(self, user_input, owner_id="default", chat_id="default"):
        self.calls.append(("invoke_sync", user_input, owner_id, chat_id))
        return f"sync:{user_input}"

    def install_skill(self, _skill):
        return "should never be visible from runtime facade"


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
            # tool_call_id must match ToolCall.id so the pair isn't treated as orphaned
            Message(role="tool", content="NOT_JSON", tool_call_id="weather_id"),
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

    def test_decode_messages_links_function_response_with_call_id_and_name(self):
        client = GeminiClient.__new__(GeminiClient)

        messages = [
            Message(
                role="ai",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_abc123",
                        name="create_work_center",
                        arguments={"name": "X"},
                    )
                ],
            ),
            Message(
                role="tool",
                content='{"ok": true}',
                tool_call_id="call_abc123",
            ),
        ]

        decoded, _ = client._decode_messages(messages)

        function_call = decoded[0]["parts"][0]["functionCall"]
        function_response = decoded[1]["parts"][0]["functionResponse"]

        self.assertEqual(function_call["id"], "call_abc123")
        self.assertEqual(function_call["name"], "create_work_center")
        self.assertEqual(function_response["id"], "call_abc123")
        self.assertEqual(function_response["name"], "create_work_center")

    def test_decode_messages_drops_orphaned_tool_response_from_closed_bucket(self):
        """Regression: bucket rollover may store ai(tool_calls) in bucket N and the
        matching tool response in bucket N+1.  get_history only returns the active
        bucket, so the tool response appears without a preceding functionCall.
        _decode_messages must silently drop it to avoid a 400 INVALID_ARGUMENT."""
        client = GeminiClient.__new__(GeminiClient)

        # Simulates what MongoMemory.get_history returns when the ai(tool_calls)
        # message for id '8dwjxazu' rolled over into a closed bucket (summarised)
        # and only the tool response survived in the active bucket.
        messages = [
            # Context summary injected from the closed bucket (no functionCall here)
            Message(role="system", content="Previous context: user asked to create a doc"),
            # Orphaned tool response — its ai(tool_calls) is in the closed bucket
            Message(role="tool", content='{"error": "internal"}', tool_call_id="8dwjxazu"),
            # Next iteration: properly paired call + response
            Message(
                role="ai",
                content="",
                tool_calls=[ToolCall(id="7w5yt56j", name="CreateDocument", arguments={})],
            ),
            Message(role="tool", content='{"error": "internal 2"}', tool_call_id="7w5yt56j"),
            # Final text turn
            Message(role="ai", content="Sorry, I can't create the doc right now."),
            # New user message appended by _build_messages
            Message(role="human", content="Can you show me the payload you used?"),
        ]

        decoded, system_instruction = client._decode_messages(messages)

        # System message captured separately
        self.assertIn("Previous context", system_instruction)

        # The orphaned tool response must NOT appear in the Gemini history
        roles = [m["role"] for m in decoded]
        parts_flat = [p for m in decoded for p in m["parts"]]
        func_responses = [p for p in parts_flat if "functionResponse" in p]
        orphan_ids = [r["functionResponse"]["id"] for r in func_responses if r["functionResponse"]["id"] == "8dwjxazu"]
        self.assertEqual(orphan_ids, [], "Orphaned functionResponse must be dropped")

        # The paired call/response for 7w5yt56j must be present and ordered correctly
        model_calls = [m for m in decoded if m["role"] == "model" and any("functionCall" in p for p in m["parts"])]
        user_responses = [m for m in decoded if m["role"] == "user" and any("functionResponse" in p for p in m["parts"])]
        self.assertEqual(len(model_calls), 1)
        self.assertEqual(len(user_responses), 1)
        self.assertEqual(model_calls[0]["parts"][0]["functionCall"]["id"], "7w5yt56j")
        self.assertEqual(user_responses[0]["parts"][0]["functionResponse"]["id"], "7w5yt56j")

        # The model call must immediately precede the user response in the list
        call_idx = decoded.index(model_calls[0])
        response_idx = decoded.index(user_responses[0])
        self.assertEqual(response_idx, call_idx + 1, "functionResponse must follow functionCall immediately")

    def test_decode_messages_drops_dangling_ai_tool_calls_without_responses(self):
        """Regression: if ai(tool_calls) is stored in the active bucket but its tool
        responses are absent (e.g. another mid-batch rollover edge case), the
        dangling functionCall must also be dropped so Gemini never sees a
        functionCall turn with no following functionResponse."""
        client = GeminiClient.__new__(GeminiClient)

        messages = [
            Message(role="human", content="Do something"),
            # Dangling: tool call made, but response never stored (all IDs absent from history)
            Message(
                role="ai",
                content="",
                tool_calls=[ToolCall(id="dangling_id", name="DoThing", arguments={})],
            ),
            # Subsequent normal text response
            Message(role="ai", content="Something went wrong, try again."),
            Message(role="human", content="OK try again"),
        ]

        decoded, _ = client._decode_messages(messages)

        # The dangling functionCall model turn must be absent
        func_calls = [
            p
            for m in decoded
            for p in m["parts"]
            if "functionCall" in p
        ]
        self.assertEqual(func_calls, [], "Dangling functionCall must be dropped")

        # Normal text turns should survive
        text_parts = [p["text"] for m in decoded for p in m["parts"] if "text" in p]
        self.assertIn("Do something", text_parts)
        self.assertIn("Something went wrong, try again.", text_parts)
        self.assertIn("OK try again", text_parts)

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
        ), patch(
            "syndicate.clients.gemini.types.FunctionDeclaration",
            side_effect=lambda **kw: {"_fd": kw},
        ), patch(
            "syndicate.clients.gemini.types.Tool",
            side_effect=lambda **kw: {"_tool": kw},
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
        tools_payload = call_kwargs["config"]["tools"]
        self.assertEqual(len(tools_payload), 1)
        self.assertIn("_tool", tools_payload[0])
        wrapped_declarations = tools_payload[0]["_tool"]["function_declarations"]
        self.assertEqual(len(wrapped_declarations), 1)
        self.assertEqual(
            wrapped_declarations[0]["_fd"]["name"],
            "weather_tool",
        )
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

    def test_encode_response_preserves_function_call_id(self):
        client = GeminiClient.__new__(GeminiClient)

        function_call = SimpleNamespace(id="call_xyz789", name="weather", args={"city": "Tokyo"})
        part = SimpleNamespace(
            text="",
            thought=None,
            function_call=function_call,
            thought_signature=None,
        )
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            finish_reason="STOP",
        )
        raw = SimpleNamespace(candidates=[candidate], usage_metadata=SimpleNamespace())

        encoded = client._encode_response(raw)

        self.assertIsNotNone(encoded.tool_calls)
        self.assertEqual(encoded.tool_calls[0].id, "call_xyz789")

    def test_encode_response_accepts_bytes_thought_signature(self):
        client = GeminiClient.__new__(GeminiClient)

        function_call = SimpleNamespace(name="weather", args={"city": "Tokyo"})
        part = SimpleNamespace(
            text="",
            thought=None,
            function_call=function_call,
            thought_signature=b"\x12\x97sig-bytes",
        )
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            finish_reason="STOP",
        )
        raw = SimpleNamespace(candidates=[candidate], usage_metadata=SimpleNamespace())

        encoded = client._encode_response(raw)

        self.assertIsNotNone(encoded.tool_calls)
        self.assertEqual(encoded.tool_calls[0].thought_signature, b"\x12\x97sig-bytes")

    def test_format_tools_wraps_function_declarations_and_keeps_native_and_callable(self):
        client = GeminiClient.__new__(GeminiClient)

        class _FakeTool:
            def to_format(self, provider_type):
                self.seen_provider = provider_type
                return {
                    "name": "weather_tool",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }

        class _NativeTool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def callable_tool(**kwargs):
            return "ok"
        native_tool = _NativeTool(function_declarations=[])
        fake_tool = _FakeTool()

        with patch("syndicate.clients.gemini.types.FunctionDeclaration", side_effect=lambda **kw: {"_fd": kw}), patch(
            "syndicate.clients.gemini.types.Tool",
            _NativeTool,
        ):
            formatted = client._format_tools([fake_tool, native_tool, callable_tool])

        self.assertIsNotNone(formatted)
        self.assertEqual(fake_tool.seen_provider, "gemini")
        self.assertEqual(len(formatted), 3)
        self.assertIs(formatted[0], native_tool)
        self.assertIs(formatted[1], callable_tool)
        self.assertIsInstance(formatted[2], _NativeTool)
        self.assertEqual(len(formatted[2].kwargs["function_declarations"]), 1)
        self.assertEqual(
            formatted[2].kwargs["function_declarations"][0]["_fd"]["name"],
            "weather_tool",
        )


class LocalMemoryBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_rollover_does_not_split_tool_call_pair_across_buckets(self):
        """Regression: rollover must fire at turn boundary, not mid-interaction.

        If the threshold fires after the ai(tool_calls) message is stored (i.e.
        the bucket is already full when _store_interaction begins), the rollover
        must happen BEFORE writing any messages in the new batch so that the entire
        human → ai(tool_calls) → tool(result) → ai(text) sequence lands in one bucket.
        """
        # Threshold of 1 interaction = 2 messages.  We prime the bucket with one
        # completed interaction so it is exactly at threshold before the tool-call
        # batch arrives.
        memory = LocalMemory(
            rollover_enabled=True,
            max_interactions_per_bucket=1,
            preserve_closed_buckets=True,
        )
        # Prime: fill bucket 0 to threshold
        await memory.add_message(Message(role="human", content="ping"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="pong"), "owner", "chat")

        # Now simulate _store_interaction writing a tool-call turn as a batch.
        # With defer_rollover=True the bucket must NOT split mid-batch.
        # We replicate exactly what _store_interaction does after the upfront rollover.
        if await memory.should_rollover("owner", "chat"):
            await memory.rollover_history("owner", "chat", summarize=False)

        tool_call_batch = [
            Message(role="human", content="create doc"),
            Message(
                role="ai",
                content="",
                tool_calls=[ToolCall(id="tcid1", name="CreateDoc", arguments={})],
            ),
            Message(role="tool", content='{"ok": true}', tool_call_id="tcid1"),
            Message(role="ai", content="Done!"),
        ]
        for msg in tool_call_batch:
            await memory.add_message(msg, "owner", "chat", defer_rollover=True)

        # The entire 4-message batch must be in a single active bucket
        active = await memory.get_active_bucket("owner", "chat")
        self.assertIsNotNone(active)
        roles = [m.role for m in active.messages]
        self.assertEqual(
            roles,
            ["human", "ai", "tool", "ai"],
            "ai(tool_calls) and tool(result) must be in the same bucket",
        )

        # The tool call and its response must both be present
        tool_call_msg = next((m for m in active.messages if m.tool_calls), None)
        tool_result_msg = next((m for m in active.messages if m.role == "tool"), None)
        self.assertIsNotNone(tool_call_msg)
        self.assertIsNotNone(tool_result_msg)
        self.assertEqual(tool_result_msg.tool_call_id, tool_call_msg.tool_calls[0].id)

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


class AgentInterfaceAndRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_facade_forwards_operational_calls(self):
        target = _FakeAgentRuntimeTarget()
        runtime = AgentRuntime(target)

        self.assertTrue(isinstance(runtime, AgentInterface))

        invoke_result = await runtime.invoke("hello", owner_id="o", chat_id="c")
        self.assertEqual(invoke_result, "echo:hello")

        chunks = []
        async for chunk in runtime.stream("stream-it", owner_id="o2", chat_id="c2", include_thinking=True):
            chunks.append(chunk)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].content, "chunk-1")
        self.assertTrue(chunks[-1].is_finished)

        sync_result = runtime.invoke_sync("sync", owner_id="o3", chat_id="c3")
        self.assertEqual(sync_result, "sync:sync")

    async def test_runtime_facade_hides_configuration_methods(self):
        target = _FakeAgentRuntimeTarget()
        runtime = AgentRuntime(target)

        self.assertFalse(hasattr(runtime, "install_skill"))
        self.assertFalse(hasattr(runtime, "add_tool"))
        self.assertFalse(hasattr(runtime, "set_system_prompt"))

    async def test_base_agent_as_runtime_returns_cached_facade(self):
        class _FakeLLMClient:
            provider_type = "openai"

            async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
                return ChatResponse(content="ok", role="ai")

            async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
                yield StreamChunk(content="ok", is_finished=True)

        memory = LocalMemory(rollover_enabled=False)
        agent = BaseAgent(llm_client=_FakeLLMClient(), memory=memory, system_prompt="sys")

        runtime_a = agent.as_runtime()
        runtime_b = agent.as_runtime()

        self.assertIs(runtime_a, runtime_b)
        self.assertTrue(isinstance(runtime_a, AgentInterface))


class GeminiSchemaCleaningTests(unittest.TestCase):
    def test_clean_schema_resolves_local_ref_from_definitions(self):
        schema = {
            "type": "object",
            "properties": {
                "input": {
                    "$ref": "#/definitions/UpdateWorkCenterDataInput"
                }
            },
            "definitions": {
                "UpdateWorkCenterDataInput": {
                    "type": "object",
                    "title": "UpdateWorkCenterDataInput",
                    "properties": {
                        "name": {"type": "string", "title": "Name"}
                    },
                    "required": ["name"],
                }
            },
        }

        cleaned = _clean_schema_for_gemini(schema)

        self.assertNotIn("definitions", cleaned)
        self.assertIn("properties", cleaned)
        self.assertIn("input", cleaned["properties"])
        self.assertNotIn("$ref", cleaned["properties"]["input"])
        self.assertEqual(cleaned["properties"]["input"]["type"], "object")
        self.assertIn("name", cleaned["properties"]["input"]["properties"])

    def test_clean_schema_collapses_oneof_nullable_array_items(self):
        schema = {
            "type": "object",
            "properties": {
                "employees": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                }
            },
        }

        cleaned = _clean_schema_for_gemini(schema)
        item_schema = cleaned["properties"]["employees"]["items"]

        self.assertNotIn("oneOf", item_schema)
        self.assertEqual(item_schema.get("type"), "string")
        self.assertTrue(item_schema.get("nullable"))


class ToolCallEventStreamTests(unittest.IsolatedAsyncioTestCase):
    """Tests that _orchestrate_stream emits ToolCallEvent chunks around tool dispatch."""

    def _make_agent(self, tool_result, tool_error=None):
        """Build a minimal BaseAgent wired with a fake LLM and a fake tool."""
        from syndicate.communication_models import ToolCallEvent

        # One-shot LLM: first call returns a tool_call, second returns final text
        call_count = {"n": 0}

        class _FakeLLM:
            provider_type = "openai"

            async def chat_completion_stream(self, messages, system_message, tools=None, **kw):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First turn: yield a tool call
                    yield StreamChunk(
                        tool_calls=[ToolCall(id="ev_id_1", name="fake_tool", arguments={"x": 1})],
                        is_finished=True,
                        finish_reason="tool_calls",
                    )
                else:
                    # Second turn: final text
                    yield StreamChunk(content="done", is_finished=True, finish_reason="stop")

        class _FakeTool:
            name = "fake_tool"

            async def run_async(self, **kwargs):
                if tool_error:
                    raise RuntimeError(tool_error)
                return tool_result

        memory = LocalMemory()
        agent = BaseAgent.__new__(BaseAgent)
        agent.llm = _FakeLLM()
        agent.memory = memory
        agent.name = "TestAgent"
        agent.system_prompt = "sys"
        agent.tools = [_FakeTool()]
        agent.max_iterations = 5
        agent.skills = []
        agent._as_tool_cache = None
        agent._as_runtime_cache = None
        from contextvars import ContextVar
        agent._request_system_prompt_ctx = ContextVar("sp", default=None)
        agent._request_formatted_tools_ctx = ContextVar("ft", default=None)
        agent._request_tool_map_ctx = ContextVar("tm", default=None)
        return agent

    async def test_stream_emits_start_and_success_events(self):
        from syndicate.communication_models import ToolCallEvent

        agent = self._make_agent(tool_result="tool_output")

        messages = [Message(role="human", content="go")]
        chunks = []
        async for chunk in agent._orchestrate_stream(messages):
            chunks.append(chunk)

        tool_events = [c for c in chunks if c.tool_call is not None]
        self.assertEqual(len(tool_events), 2, "Expected start + success events")

        start_evt = tool_events[0].tool_call
        success_evt = tool_events[1].tool_call

        self.assertEqual(start_evt.status, "start")
        self.assertEqual(start_evt.tool_name, "fake_tool")
        self.assertEqual(start_evt.tool_call_id, "ev_id_1")
        self.assertEqual(start_evt.args, {"x": 1})
        self.assertIsNone(start_evt.result)

        self.assertEqual(success_evt.status, "success")
        self.assertEqual(success_evt.tool_name, "fake_tool")
        self.assertEqual(success_evt.tool_call_id, "ev_id_1")
        self.assertEqual(success_evt.result, "tool_output")

        # Final text chunk must also arrive
        text_chunks = [c for c in chunks if c.content]
        self.assertTrue(any(c.content == "done" for c in text_chunks))

    async def test_stream_emits_error_event_on_tool_failure(self):
        agent = self._make_agent(tool_result=None, tool_error="boom")

        messages = [Message(role="human", content="go")]
        chunks = []
        async for chunk in agent._orchestrate_stream(messages):
            chunks.append(chunk)

        tool_events = [c for c in chunks if c.tool_call is not None]
        statuses = [e.tool_call.status for e in tool_events]
        self.assertIn("start", statuses)
        self.assertIn("error", statuses)

        error_evt = next(e.tool_call for e in tool_events if e.tool_call.status == "error")
        self.assertEqual(error_evt.tool_name, "fake_tool")
        self.assertIsNotNone(error_evt.error)

    async def test_existing_content_chunks_unaffected(self):
        """Consumers that only read chunk.content must receive an unbroken stream."""
        agent = self._make_agent(tool_result="r")

        messages = [Message(role="human", content="go")]
        content = ""
        async for chunk in agent._orchestrate_stream(messages):
            content += chunk.content  # tool_call chunks have content="" by default

        self.assertEqual(content, "done")


if __name__ == "__main__":
    unittest.main(verbosity=2)
