import asyncio
import json
import os
import tempfile
import unittest
import warnings
from datetime import datetime, timezone
from unittest import IsolatedAsyncioTestCase
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from uuid import uuid4
from collections.abc import AsyncGenerator

from syndicate.protocols import AgentInterface
from syndicate.agents.runtime import AgentRuntime
from syndicate.agents.base import BaseAgent
from syndicate.communication_models import Message, ToolCall, ChatResponse, StreamChunk, ToolResultEnvelope
from syndicate.clients.gemini import GeminiClient
from syndicate.clients.openai import OpenAIClient
from syndicate.memory.local import LocalMemory
from syndicate.memory.summarizers import resolve_summarizer
from pymongo.errors import DuplicateKeyError, OperationFailure
from syndicate.memory.mongo import MongoMemory
from syndicate.memory.sqlite_postgres import SqlitePostgresMemory
from syndicate.mcp import MCPSubTool
from syndicate.observability import InMemoryObserver
from syndicate.skills import KnowledgeBaseSkill
from syndicate.tools.agent_tool import AgentAsTool
from syndicate.vectorstores.mongo import MongoVectorStore
from syndicate.tools.base_tool import (
    BaseTool,
    ToolBackoffPolicy,
    ToolExecutionPolicy,
    _clean_schema_for_gemini,
)


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


class _FakeRegenerateClient:
    provider_type = "openai"
    model_name = "fake-regenerate-model"

    async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
        return ChatResponse(content="regenerated answer")


class _FakeEmbeddingModel:
    def __init__(
        self,
        query_embedding=None,
        batch_embeddings=None,
        default_dimension=2,
        supports_dimension_override=True,
        supported_modes=None,
    ):
        self.query_embedding = query_embedding or [0.1, 0.2]
        self.batch_embeddings = batch_embeddings
        self.default_dimension = default_dimension
        self.supports_dimension_override = supports_dimension_override
        self.supported_modes = set(supported_modes or {"query", "document"})
        self._effective_dimension = default_dimension
        self.dimension_source = "default"
        self.closed = False
        self.embed_calls = []
        self.embed_batch_calls = []

    @property
    def model_name(self):
        return "fake-embedding-model"

    @property
    def embedding_dimension(self):
        return self._effective_dimension

    @property
    def embedding_space_id(self):
        return f"fake:{self.model_name}:dim={self.embedding_dimension}"

    def get_model_info(self):
        return {
            "provider": "fake",
            "model_name": self.model_name,
            "default_dimension": self.default_dimension,
            "effective_dimension": self.embedding_dimension,
            "supports_dimension_override": self.supports_dimension_override,
            "dimension_source": self.dimension_source,
            "embedding_space_id": self.embedding_space_id,
        }

    def get_capabilities(self):
        return {
            "supported_modes": sorted(self.supported_modes),
            "supports_batching": True,
            "supports_dimension_override": self.supports_dimension_override,
            "max_batch_size": None,
            "max_input_tokens": None,
        }

    def supports_mode(self, mode):
        mode_value = getattr(mode, "value", str(mode)).lower()
        return mode_value in self.supported_modes

    def configure_dimension(self, dims, source="explicit"):
        if dims <= 0:
            raise ValueError("dims must be a positive integer")
        if not self.supports_dimension_override and dims != self.default_dimension:
            raise ValueError(
                f"Model {self.model_name} has fixed dimension {self.default_dimension}; "
                f"cannot configure {dims}."
            )

        self._effective_dimension = dims
        self.dimension_source = source

    async def embed(self, text, mode="document"):
        self.embed_calls.append(text)
        return self.query_embedding

    async def embed_batch(self, texts, mode="document"):
        self.embed_batch_calls.append(list(texts))
        if self.batch_embeddings is not None:
            return self.batch_embeddings
        return [
            [float(i + idx) for idx in range(self.embedding_dimension)]
            for i in range(len(texts))
        ]

    async def close(self):
        self.closed = True


class _FakeAsyncCursor:
    def __init__(self, results):
        self.results = results
        self.lengths = []

    async def to_list(self, length=None):
        self.lengths.append(length)
        return self.results


class _FakeMongoCollectionForVectorStore:
    def __init__(
        self,
        *,
        aggregate_results=None,
        aggregate_error=None,
        find_results=None,
        delete_count=0,
        search_indexes=None,
        search_index_api_available=True,
        create_search_index_error=None,
    ):
        self.aggregate_results = aggregate_results if aggregate_results is not None else []
        self.aggregate_error = aggregate_error
        self.find_results = find_results if find_results is not None else []
        self.delete_count = delete_count
        self.search_indexes = list(search_indexes or [])
        self.search_index_api_available = search_index_api_available
        self.create_search_index_error = create_search_index_error
        self.aggregate_pipelines = []
        self.insert_many_calls = []
        self.delete_queries = []
        self.find_queries = []
        self.created_search_index_models = []

    def aggregate(self, pipeline):
        self.aggregate_pipelines.append(pipeline)
        if self.aggregate_error is not None:
            raise self.aggregate_error
        return _FakeAsyncCursor(self.aggregate_results)

    async def insert_many(self, documents, ordered=False):
        self.insert_many_calls.append({"documents": documents, "ordered": ordered})

    async def delete_many(self, query):
        self.delete_queries.append(query)
        return SimpleNamespace(deleted_count=self.delete_count)

    def find(self, query):
        self.find_queries.append(query)
        return _FakeAsyncCursor(self.find_results)

    def list_search_indexes(self):
        if not self.search_index_api_available:
            raise AttributeError("list_search_indexes unavailable")
        return _FakeAsyncCursor(self.search_indexes)

    async def create_search_index(self, model=None):
        if not self.search_index_api_available:
            raise AttributeError("create_search_index unavailable")
        if self.create_search_index_error is not None:
            raise self.create_search_index_error

        self.created_search_index_models.append(model)

        document = getattr(model, "document", None)
        if isinstance(document, dict):
            name = document.get("name")
            definition = document.get("definition") or {}
        else:
            name = getattr(model, "name", None)
            definition = getattr(model, "definition", {})

        if not isinstance(name, str) or not name:
            name = f"search-index-{len(self.created_search_index_models)}"

        self.search_indexes.append(
            {
                "name": name,
                "definition": definition if isinstance(definition, dict) else {},
            }
        )
        return name

    async def create_search_indexes(self, models):
        created = []
        for model in models:
            created.append(await self.create_search_index(model=model))
        return created


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
                include_thoughts=True,
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
        self.assertTrue(call_kwargs["config"]["thinking_config"]["include_thoughts"])

        contents = call_kwargs["contents"]
        self.assertEqual(contents[-1]["role"], "user")
        self.assertEqual(contents[-1]["parts"][0]["text"], "second")
        self.assertEqual(
            contents[-1]["parts"][1],
            {"inline_data": {"mime_type": "image/jpeg", "data": "BASE64_IMAGE"}},
        )

    async def test_chat_completion_stream_emits_thinking_tokens_usage(self):
        client = GeminiClient.__new__(GeminiClient)
        client.model_name = "gemini-test"
        client.temperature = 0.3
        client._decode_messages = lambda messages: ([{"role": "user", "parts": [{"text": "hi"}]}], "sys")
        client._format_tools = lambda tools: None

        part = SimpleNamespace(text="hello", thought="internal", function_call=None)
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            finish_reason="STOP",
        )
        usage = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            thoughts_token_count=4,
        )

        async def _chunk_iter():
            yield SimpleNamespace(candidates=[candidate], usage_metadata=usage)

        fake_stream = AsyncMock(return_value=_chunk_iter())
        client.client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content_stream=fake_stream)
            )
        )

        with patch(
            "syndicate.clients.gemini.types.GenerateContentConfig",
            side_effect=lambda **kw: kw,
        ):
            chunks = []
            async for chunk in client.chat_completion_stream(
                messages=[Message(role="human", content="hello")],
                system_message=Message(role="system", content="sys"),
            ):
                chunks.append(chunk)

        self.assertEqual(len(chunks), 1)
        stream_chunk = chunks[0]
        self.assertEqual(stream_chunk.content, "hello")
        self.assertEqual(stream_chunk.thinking, "internal")
        self.assertEqual(stream_chunk.thinking_tokens, 4)
        self.assertEqual(stream_chunk.usage["prompt_tokens"], 10)
        self.assertEqual(stream_chunk.usage["completion_tokens"], 5)
        self.assertEqual(stream_chunk.usage["total_tokens"], 15)
        self.assertEqual(stream_chunk.usage["thinking_tokens"], 4)

    async def test_chat_completion_stream_extracts_thinking_from_thought_marked_text_parts(self):
        client = GeminiClient.__new__(GeminiClient)
        client.model_name = "gemini-test"
        client.temperature = 0.3
        client._decode_messages = lambda messages: ([{"role": "user", "parts": [{"text": "hi"}]}], "sys")
        client._format_tools = lambda tools: None

        thought_part = SimpleNamespace(text="internal rationale", thought=True, function_call=None)
        answer_part = SimpleNamespace(text="final answer", thought=False, function_call=None)
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[thought_part, answer_part]),
            finish_reason="STOP",
        )
        usage = SimpleNamespace(thoughts_token_count=9)

        async def _chunk_iter():
            yield SimpleNamespace(candidates=[candidate], usage_metadata=usage)

        fake_stream = AsyncMock(return_value=_chunk_iter())
        client.client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content_stream=fake_stream)
            )
        )

        with patch(
            "syndicate.clients.gemini.types.GenerateContentConfig",
            side_effect=lambda **kw: kw,
        ):
            chunks = []
            async for chunk in client.chat_completion_stream(
                messages=[Message(role="human", content="hello")],
                system_message=Message(role="system", content="sys"),
            ):
                chunks.append(chunk)

        self.assertEqual(len(chunks), 1)
        stream_chunk = chunks[0]
        self.assertEqual(stream_chunk.content, "final answer")
        self.assertEqual(stream_chunk.thinking, "internal rationale")
        self.assertEqual(stream_chunk.thinking_tokens, 9)

    async def test_chat_completion_stream_sets_include_thoughts_config_when_requested(self):
        client = GeminiClient.__new__(GeminiClient)
        client.model_name = "gemini-test"
        client.temperature = 0.3
        client._decode_messages = lambda messages: ([{"role": "user", "parts": [{"text": "hi"}]}], "sys")
        client._format_tools = lambda tools: None

        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text="ok", thought=False, function_call=None)]),
            finish_reason="STOP",
        )

        async def _chunk_iter():
            yield SimpleNamespace(candidates=[candidate], usage_metadata=SimpleNamespace())

        fake_stream = AsyncMock(return_value=_chunk_iter())
        client.client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content_stream=fake_stream)
            )
        )

        with patch("syndicate.clients.gemini.types.ThinkingConfig", side_effect=lambda **kw: kw), patch(
            "syndicate.clients.gemini.types.GenerateContentConfig",
            side_effect=lambda **kw: kw,
        ):
            async for _ in client.chat_completion_stream(
                messages=[Message(role="human", content="hello")],
                system_message=Message(role="system", content="sys"),
                include_thoughts=True,
            ):
                pass

        call_kwargs = fake_stream.call_args.kwargs
        self.assertTrue(call_kwargs["config"]["thinking_config"]["include_thoughts"])

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

    def test_encode_response_extracts_thinking_from_thought_marked_text_parts(self):
        client = GeminiClient.__new__(GeminiClient)

        thought_part = SimpleNamespace(
            text="internal chain",
            thought=True,
            function_call=None,
            thought_signature=None,
        )
        answer_part = SimpleNamespace(
            text="final",
            thought=False,
            function_call=None,
            thought_signature=None,
        )
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[thought_part, answer_part]),
            finish_reason="STOP",
        )
        usage = SimpleNamespace(thoughts_token_count=6)
        raw = SimpleNamespace(candidates=[candidate], usage_metadata=usage)

        encoded = client._encode_response(raw)

        self.assertEqual(encoded.content, "final")
        self.assertEqual(encoded.thinking, "internal chain")
        self.assertEqual(encoded.thinking_tokens, 6)

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


class LocalMemoryRollbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_message_soft_delete_hides_from_history(self):
        memory = LocalMemory(soft_delete=True)

        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")

        deleted = await memory.delete_message("owner", "chat", index=-1)
        self.assertTrue(deleted)

        history = await memory.get_history("owner", "chat")
        self.assertEqual([m.role for m in history], ["human"])

        active = await memory.get_active_bucket("owner", "chat")
        self.assertIsNotNone(active)
        self.assertEqual(len(active.messages), 2, "soft delete should preserve physical messages")

    async def test_delete_message_hard_delete_removes_from_bucket(self):
        memory = LocalMemory(soft_delete=False)

        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")

        deleted = await memory.delete_message("owner", "chat", index=0)
        self.assertTrue(deleted)

        history = await memory.get_history("owner", "chat")
        self.assertEqual([m.role for m in history], ["ai"])

        active = await memory.get_active_bucket("owner", "chat")
        self.assertIsNotNone(active)
        self.assertEqual(len(active.messages), 1, "hard delete should remove physical message")

    async def test_delete_last_message_role_filter(self):
        memory = LocalMemory(soft_delete=True)

        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")
        await memory.add_message(Message(role="human", content="u2"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a2"), "owner", "chat")

        deleted = await memory.delete_last_message("owner", "chat", role="human")
        self.assertTrue(deleted)

        history = await memory.get_history("owner", "chat")
        self.assertEqual([m.content for m in history], ["u1", "a1", "a2"])

    async def test_delete_message_out_of_range_returns_false(self):
        memory = LocalMemory()
        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")

        deleted = await memory.delete_message("owner", "chat", index=10)
        self.assertFalse(deleted)


class LocalMemoryFullHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_full_history_supports_closed_deleted_and_limit(self):
        memory = LocalMemory(soft_delete=True)

        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")

        first_bucket = await memory.get_active_bucket("owner", "chat")
        self.assertIsNotNone(first_bucket)
        await memory.close_bucket(first_bucket.bucket_id)
        await memory.create_bucket("owner", "chat", position=1)

        await memory.add_message(Message(role="human", content="u2"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a2"), "owner", "chat")
        await memory.delete_message("owner", "chat", index=-1)

        history = await memory.get_full_history("owner", "chat")
        self.assertEqual([m.content for m in history], ["u1", "a1", "u2"])

        with_deleted = await memory.get_full_history("owner", "chat", include_deleted=True)
        self.assertEqual([m.content for m in with_deleted], ["u1", "a1", "u2", "a2"])

        active_only = await memory.get_full_history("owner", "chat", include_closed_buckets=False)
        self.assertEqual([m.content for m in active_only], ["u2"])

        limited = await memory.get_full_history(
            "owner",
            "chat",
            include_deleted=True,
            limit=2,
        )
        self.assertEqual([m.content for m in limited], ["u2", "a2"])


class SqlitePostgresMemoryDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_soft_delete_hides_message_from_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "delete_soft.db")
            memory = SqlitePostgresMemory(
                database_url=f"sqlite+aiosqlite:///{db_path}",
                soft_delete=True,
            )

            try:
                await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
                await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")

                deleted = await memory.delete_message("owner", "chat", index=-1)
                self.assertTrue(deleted)

                history = await memory.get_history("owner", "chat")
                self.assertEqual([m.role for m in history], ["human"])

                active = await memory.get_active_bucket("owner", "chat")
                self.assertIsNotNone(active)
                self.assertEqual(len(active.messages), 2)
            finally:
                if memory._engine is not None:
                    await memory._engine.dispose()


class SqlitePostgresMemoryFullHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_full_history_supports_closed_deleted_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "full_history.db")
            memory = SqlitePostgresMemory(
                database_url=f"sqlite+aiosqlite:///{db_path}",
                soft_delete=True,
            )

            try:
                await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
                await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")

                first_bucket = await memory.get_active_bucket("owner", "chat")
                self.assertIsNotNone(first_bucket)
                await memory.close_bucket(first_bucket.bucket_id)
                await memory.create_bucket("owner", "chat", position=1)

                await memory.add_message(Message(role="human", content="u2"), "owner", "chat")
                await memory.add_message(Message(role="ai", content="a2"), "owner", "chat")
                await memory.delete_message("owner", "chat", index=-1)

                history = await memory.get_full_history("owner", "chat")
                self.assertEqual([m.content for m in history], ["u1", "a1", "u2"])

                with_deleted = await memory.get_full_history("owner", "chat", include_deleted=True)
                self.assertEqual([m.content for m in with_deleted], ["u1", "a1", "u2", "a2"])

                active_only = await memory.get_full_history(
                    "owner",
                    "chat",
                    include_closed_buckets=False,
                )
                self.assertEqual([m.content for m in active_only], ["u2"])

                limited = await memory.get_full_history(
                    "owner",
                    "chat",
                    include_deleted=True,
                    limit=2,
                )
                self.assertEqual([m.content for m in limited], ["u2", "a2"])
            finally:
                if memory._engine is not None:
                    await memory._engine.dispose()

    async def test_hard_delete_removes_message_from_storage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "delete_hard.db")
            memory = SqlitePostgresMemory(
                database_url=f"sqlite+aiosqlite:///{db_path}",
                soft_delete=False,
            )

            try:
                await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
                await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")

                deleted = await memory.delete_message("owner", "chat", index=0)
                self.assertTrue(deleted)

                history = await memory.get_history("owner", "chat")
                self.assertEqual([m.role for m in history], ["ai"])

                active = await memory.get_active_bucket("owner", "chat")
                self.assertIsNotNone(active)
                self.assertEqual(len(active.messages), 1)
            finally:
                if memory._engine is not None:
                    await memory._engine.dispose()


class AgentRegenerateTests(unittest.IsolatedAsyncioTestCase):
    async def test_regenerate_last_ai_replaces_tail(self):
        memory = LocalMemory()
        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="old answer"), "owner", "chat")

        agent = BaseAgent(
            llm_client=_FakeRegenerateClient(),
            memory=memory,
            system_prompt="system",
        )

        response = await agent.regenerate_response(owner_id="owner", chat_id="chat")
        self.assertEqual(response.content, "regenerated answer")

        history = await memory.get_history("owner", "chat")
        self.assertEqual([m.role for m in history], ["human", "ai"])
        self.assertEqual(history[0].content, "u1")
        self.assertEqual(history[1].content, "regenerated answer")

    async def test_regenerate_target_index_truncates_branch(self):
        memory = LocalMemory()
        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")
        await memory.add_message(Message(role="human", content="u2"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a2"), "owner", "chat")

        agent = BaseAgent(
            llm_client=_FakeRegenerateClient(),
            memory=memory,
            system_prompt="system",
        )

        response = await agent.regenerate_response(
            owner_id="owner",
            chat_id="chat",
            target_index=1,
        )
        self.assertEqual(response.content, "regenerated answer")

        history = await memory.get_history("owner", "chat")
        self.assertEqual([m.role for m in history], ["human", "ai"])
        self.assertEqual(history[0].content, "u1")
        self.assertEqual(history[1].content, "regenerated answer")

    async def test_regenerate_raises_without_ai_turn(self):
        memory = LocalMemory()
        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")

        agent = BaseAgent(
            llm_client=_FakeRegenerateClient(),
            memory=memory,
            system_prompt="system",
        )

        with self.assertRaises(ValueError):
            await agent.regenerate_response(owner_id="owner", chat_id="chat")


class AgentFullHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_get_full_history_delegates_to_memory(self):
        memory = LocalMemory(soft_delete=True)

        await memory.add_message(Message(role="human", content="u1"), "owner", "chat")
        await memory.add_message(Message(role="ai", content="a1"), "owner", "chat")

        first_bucket = await memory.get_active_bucket("owner", "chat")
        self.assertIsNotNone(first_bucket)
        await memory.close_bucket(first_bucket.bucket_id)
        await memory.create_bucket("owner", "chat", position=1)

        await memory.add_message(Message(role="human", content="u2"), "owner", "chat")

        agent = BaseAgent(
            llm_client=_FakeRegenerateClient(),
            memory=memory,
            system_prompt="system",
        )

        full_history = await agent.get_full_history(owner_id="owner", chat_id="chat")
        self.assertEqual([m.content for m in full_history], ["u1", "a1", "u2"])

        active_only = await agent.get_full_history(
            owner_id="owner",
            chat_id="chat",
            include_closed_buckets=False,
        )
        self.assertEqual([m.content for m in active_only], ["u2"])


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


class _FakeMongoCollectionForDeleteFlow:
    def __init__(self, active_doc):
        self.active_doc = active_doc
        self.update_calls = []

    async def find_one(self, query, sort=None):
        if (
            query.get("owner_id") == self.active_doc.get("owner_id")
            and query.get("chat_id") == self.active_doc.get("chat_id")
            and query.get("is_active") is True
        ):
            return self.active_doc
        return None

    async def update_one(self, query, update):
        self.update_calls.append((query, update))
        if query.get("bucket_id") != self.active_doc.get("bucket_id"):
            return None
        payload = update.get("$set", {})
        if "messages" in payload:
            self.active_doc["messages"] = payload["messages"]
        return None


class _FakeMongoAsyncIterCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._docs):
            raise StopAsyncIteration
        value = self._docs[self._index]
        self._index += 1
        return value


class _FakeMongoCollectionForFullHistory:
    def __init__(self, docs):
        self.docs = list(docs)

    def find(self, query, sort=None):
        results = [
            doc for doc in self.docs
            if all(doc.get(key) == value for key, value in query.items())
        ]

        if sort:
            field, direction = sort[0]
            reverse = direction < 0
            results.sort(key=lambda doc: doc.get(field, 0), reverse=reverse)

        return _FakeMongoAsyncIterCursor(results)

    async def find_one(self, query, sort=None):
        results = [
            doc for doc in self.docs
            if all(doc.get(key) == value for key, value in query.items())
        ]

        if sort:
            field, direction = sort[0]
            reverse = direction < 0
            results.sort(key=lambda doc: doc.get(field, 0), reverse=reverse)

        if not results:
            return None
        return results[0]


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


class MongoMemoryDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_soft_delete_hides_message_from_history(self):
        active_doc = {
            "bucket_id": "bucket-1",
            "owner_id": "owner",
            "chat_id": "chat",
            "messages": [
                {
                    "role": "human",
                    "content": "u1",
                    "timestamp": datetime.now(timezone.utc),
                },
                {
                    "role": "ai",
                    "content": "a1",
                    "timestamp": datetime.now(timezone.utc),
                },
            ],
            "summary": None,
            "is_active": True,
            "position": 0,
        }
        fake_collection = _FakeMongoCollectionForDeleteFlow(active_doc)

        memory = MongoMemory.__new__(MongoMemory)
        memory._collection = fake_collection
        memory._indexes_created = True
        memory.soft_delete = True

        async def _noop_indexes():
            return None

        memory._ensure_indexes = _noop_indexes

        deleted = await memory.delete_message("owner", "chat", index=-1)
        self.assertTrue(deleted)

        history = await memory.get_history(
            "owner",
            "chat",
            include_context_summary=False,
        )
        self.assertEqual([m.role for m in history], ["human"])
        self.assertEqual(len(active_doc["messages"]), 2)
        self.assertTrue(active_doc["messages"][1].get("$deleted", False))

    async def test_hard_delete_removes_message_from_storage(self):
        active_doc = {
            "bucket_id": "bucket-2",
            "owner_id": "owner",
            "chat_id": "chat",
            "messages": [
                {
                    "role": "human",
                    "content": "u1",
                    "timestamp": datetime.now(timezone.utc),
                },
                {
                    "role": "ai",
                    "content": "a1",
                    "timestamp": datetime.now(timezone.utc),
                },
            ],
            "summary": None,
            "is_active": True,
            "position": 0,
        }
        fake_collection = _FakeMongoCollectionForDeleteFlow(active_doc)

        memory = MongoMemory.__new__(MongoMemory)
        memory._collection = fake_collection
        memory._indexes_created = True
        memory.soft_delete = False

        async def _noop_indexes():
            return None

        memory._ensure_indexes = _noop_indexes

        deleted = await memory.delete_message("owner", "chat", index=0)
        self.assertTrue(deleted)

        history = await memory.get_history(
            "owner",
            "chat",
            include_context_summary=False,
        )
        self.assertEqual([m.role for m in history], ["ai"])
        self.assertEqual(len(active_doc["messages"]), 1)
        self.assertEqual(active_doc["messages"][0]["role"], "ai")


class MongoMemoryFullHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_full_history_supports_closed_deleted_and_limit(self):
        docs = [
            {
                "bucket_id": "bucket-0",
                "owner_id": "owner",
                "chat_id": "chat",
                "messages": [
                    {
                        "role": "human",
                        "content": "u1",
                        "timestamp": datetime.now(timezone.utc),
                    },
                    {
                        "role": "ai",
                        "content": "a1",
                        "timestamp": datetime.now(timezone.utc),
                    },
                ],
                "summary": None,
                "is_active": False,
                "position": 0,
            },
            {
                "bucket_id": "bucket-1",
                "owner_id": "owner",
                "chat_id": "chat",
                "messages": [
                    {
                        "role": "human",
                        "content": "u2",
                        "timestamp": datetime.now(timezone.utc),
                    },
                    {
                        "role": "ai",
                        "content": "a2",
                        "timestamp": datetime.now(timezone.utc),
                        "$deleted": True,
                    },
                ],
                "summary": None,
                "is_active": True,
                "position": 1,
            },
        ]
        fake_collection = _FakeMongoCollectionForFullHistory(docs)

        memory = MongoMemory.__new__(MongoMemory)
        memory._collection = fake_collection
        memory._indexes_created = True

        async def _noop_indexes():
            return None

        memory._ensure_indexes = _noop_indexes

        history = await memory.get_full_history("owner", "chat")
        self.assertEqual([m.content for m in history], ["u1", "a1", "u2"])

        with_deleted = await memory.get_full_history("owner", "chat", include_deleted=True)
        self.assertEqual([m.content for m in with_deleted], ["u1", "a1", "u2", "a2"])

        active_only = await memory.get_full_history(
            "owner",
            "chat",
            include_closed_buckets=False,
        )
        self.assertEqual([m.content for m in active_only], ["u2"])

        limited = await memory.get_full_history(
            "owner",
            "chat",
            include_deleted=True,
            limit=2,
        )
        self.assertEqual([m.content for m in limited], ["u2", "a2"])


class MongoVectorStoreTests(unittest.IsolatedAsyncioTestCase):
    def _build_store(self, embedding_model=None):
        model = embedding_model or _FakeEmbeddingModel()
        store = MongoVectorStore(
            connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
            database="test_db",
            collection="test_collection",
            embedding_model=model,
            dims=2,
        )
        return store, model

    def test_constructor_warns_when_store_dims_override_model_dims(self):
        model = _FakeEmbeddingModel(default_dimension=2, supports_dimension_override=True)

        with self.assertWarns(RuntimeWarning):
            store = MongoVectorStore(
                connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
                database="test_db",
                collection="test_collection",
                embedding_model=model,
                dims=3,
            )

        self.assertEqual(store.effective_dimension, 3)
        self.assertEqual(model.embedding_dimension, 3)
        self.assertEqual(store.dimension_source, "vector_store")

    def test_constructor_fails_on_fixed_dimension_mismatch(self):
        model = _FakeEmbeddingModel(default_dimension=2, supports_dimension_override=False)

        with self.assertRaises(ValueError) as ctx:
            MongoVectorStore(
                connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
                database="test_db",
                collection="test_collection",
                embedding_model=model,
                dims=3,
            )

        self.assertIn("fixed dimension", str(ctx.exception))

    def test_constructor_requires_query_and_document_modes(self):
        model = _FakeEmbeddingModel(supported_modes={"document"})

        with self.assertRaises(ValueError) as ctx:
            MongoVectorStore(
                connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
                database="test_db",
                collection="test_collection",
                embedding_model=model,
                dims=2,
            )

        self.assertIn("missing required modes", str(ctx.exception))

    async def test_search_defaults_to_hybrid_and_embeds_query(self):
        store, model = self._build_store()
        store._collection = object()  # avoid client initialization for this unit test
        store._hybrid_search = AsyncMock(return_value=[{"id": "doc-h"}])
        store._vector_search = AsyncMock(return_value=[{"id": "doc-v"}])

        result = await store.search("benefits policy", k=5, filter={"tenant": "acme"})

        self.assertEqual(result, [{"id": "doc-h"}])
        self.assertEqual(model.embed_calls, ["benefits policy"])
        store._hybrid_search.assert_awaited_once_with([0.1, 0.2], "benefits policy", 5, {"tenant": "acme"})
        store._vector_search.assert_not_awaited()

    async def test_search_vector_only_path_when_hybrid_disabled(self):
        store, model = self._build_store()
        store._collection = object()  # avoid client initialization for this unit test
        store._hybrid_search = AsyncMock(return_value=[{"id": "doc-h"}])
        store._vector_search = AsyncMock(return_value=[{"id": "doc-v"}])

        result = await store.search("vacation days", k=2, use_hybrid=False)

        self.assertEqual(result, [{"id": "doc-v"}])
        self.assertEqual(model.embed_calls, ["vacation days"])
        store._vector_search.assert_awaited_once_with([0.1, 0.2], 2, None)
        store._hybrid_search.assert_not_awaited()

    async def test_search_rejects_query_dimension_mismatch(self):
        model = _FakeEmbeddingModel(query_embedding=[0.1])
        store, _ = self._build_store(embedding_model=model)
        store._collection = object()

        with self.assertRaises(ValueError) as ctx:
            await store.search("benefits policy")

        self.assertIn("Query embedding dimension mismatch", str(ctx.exception))

    async def test_search_realigns_embedding_dimension_after_external_drift(self):
        class _DynamicEmbeddingModel(_FakeEmbeddingModel):
            async def embed(self, text, mode="document"):
                self.embed_calls.append(text)
                return [float(i) for i in range(self.embedding_dimension)]

        model = _DynamicEmbeddingModel(default_dimension=2, supports_dimension_override=True)
        store, _ = self._build_store(embedding_model=model)
        store._collection = object()
        store._hybrid_search = AsyncMock(return_value=[{"id": "doc-h"}])

        # Simulate a shared model instance being reconfigured elsewhere.
        model.configure_dimension(3, source="external")

        result = await store.search("benefits policy")

        self.assertEqual(result, [{"id": "doc-h"}])
        self.assertEqual(model.embedding_dimension, 2)
        self.assertEqual(model.dimension_source, "vector_store")
        store._hybrid_search.assert_awaited_once_with([0.0, 1.0], "benefits policy", 4, None)

    async def test_vector_search_builds_pipeline_with_prefilter(self):
        fake_collection = _FakeMongoCollectionForVectorStore(
            aggregate_results=[
                {
                    "_id": "doc-1",
                    "text": "Remote policy content",
                    "metadata": {"source": "handbook"},
                    "score": 0.92,
                }
            ]
        )
        store, _ = self._build_store()
        store._collection = fake_collection

        result = await store._vector_search([0.3, 0.4], k=3, filter={"source": "handbook"})

        self.assertEqual(len(fake_collection.aggregate_pipelines), 1)
        pipeline = fake_collection.aggregate_pipelines[0]
        vector_stage = pipeline[0]["$vectorSearch"]
        self.assertEqual(vector_stage["queryVector"], [0.3, 0.4])
        self.assertEqual(vector_stage["numCandidates"], 30)
        self.assertEqual(vector_stage["limit"], 3)
        self.assertEqual(
            vector_stage["preFilter"],
            {"metadata": {"$eq": {"source": "handbook"}}},
        )
        self.assertEqual(result[0]["id"], "doc-1")
        self.assertEqual(result[0]["metadata"], {"source": "handbook"})
        self.assertEqual(result[0]["score"], 0.92)

    async def test_keyword_search_returns_empty_when_atlas_search_fails(self):
        fake_collection = _FakeMongoCollectionForVectorStore(
            aggregate_error=OperationFailure("$search unavailable")
        )
        store, _ = self._build_store()
        store._collection = fake_collection

        result = await store._keyword_search("benefits", k=4)

        self.assertEqual(result, [])
        self.assertEqual(len(fake_collection.aggregate_pipelines), 1)

    async def test_hybrid_search_merges_and_limits_results(self):
        store, _ = self._build_store()
        store._vector_search = AsyncMock(
            return_value=[
                {"id": "doc-a", "text": "A", "metadata": {}},
                {"id": "doc-b", "text": "B", "metadata": {}},
            ]
        )
        store._keyword_search = AsyncMock(
            return_value=[
                {"id": "doc-b", "text": "B", "metadata": {}},
                {"id": "doc-c", "text": "C", "metadata": {}},
            ]
        )

        result = await store._hybrid_search([0.5, 0.6], "policy", k=2)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "doc-b")
        self.assertIn("rrf_score", result[0])

    async def test_add_texts_get_by_ids_and_delete_paths(self):
        model = _FakeEmbeddingModel(batch_embeddings=[[0.01, 0.02], [0.03, 0.04]])
        fake_collection = _FakeMongoCollectionForVectorStore(
            find_results=[
                {"_id": "id-1", "text": "Doc 1", "metadata": {"topic": "hr"}},
                {"_id": "id-2", "text": "Doc 2", "metadata": None},
            ],
            delete_count=2,
        )
        store, _ = self._build_store(embedding_model=model)
        store._collection = fake_collection

        ids = await store.add_texts(
            texts=["Doc 1", "Doc 2"],
            metadatas=[{"topic": "hr"}, {"topic": "eng"}],
            ids=["id-1", "id-2"],
        )
        fetched = await store.get_by_ids(["id-1", "id-2"])
        deleted_selected = await store.delete(["id-1", "id-2"])
        deleted_all = await store.delete()

        self.assertEqual(ids, ["id-1", "id-2"])
        self.assertEqual(model.embed_batch_calls, [["Doc 1", "Doc 2"]])
        self.assertEqual(len(fake_collection.insert_many_calls), 1)
        inserted = fake_collection.insert_many_calls[0]
        self.assertFalse(inserted["ordered"])
        self.assertEqual(inserted["documents"][0]["_id"], "id-1")
        self.assertEqual(inserted["documents"][0]["embedding"], [0.01, 0.02])
        self.assertEqual(inserted["documents"][1]["metadata"], {"topic": "eng"})

        self.assertEqual(fake_collection.find_queries, [{"_id": {"$in": ["id-1", "id-2"]}}])
        self.assertEqual(fetched[0]["id"], "id-1")
        self.assertEqual(fetched[1]["metadata"], {})

        self.assertEqual(fake_collection.delete_queries[0], {"_id": {"$in": ["id-1", "id-2"]}})
        self.assertEqual(fake_collection.delete_queries[1], {})
        self.assertEqual(deleted_selected, 2)
        self.assertEqual(deleted_all, 2)

    async def test_add_texts_rejects_document_dimension_mismatch(self):
        model = _FakeEmbeddingModel(batch_embeddings=[[0.01], [0.02]])
        fake_collection = _FakeMongoCollectionForVectorStore()
        store, _ = self._build_store(embedding_model=model)
        store._collection = fake_collection

        with self.assertRaises(ValueError) as ctx:
            await store.add_texts(texts=["Doc 1", "Doc 2"])

        self.assertIn("Document embedding dimension mismatch", str(ctx.exception))

    async def test_close_chains_embedding_model_close(self):
        class _FakeClient:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        store, model = self._build_store()
        fake_client = _FakeClient()
        store._client = fake_client

        await store.close()
        await store.close()  # idempotent/safe repeated close

        self.assertTrue(fake_client.closed)
        self.assertTrue(model.closed)

    async def test_backend_bootstrap_creates_indexes_once_and_is_idempotent(self):
        fake_collection = _FakeMongoCollectionForVectorStore(search_indexes=[])
        store, _ = self._build_store()
        store._collection = fake_collection

        await store.ensure_backend_ready(create_indexes=True)
        await store.ensure_backend_ready(create_indexes=True)

        self.assertEqual(len(fake_collection.created_search_index_models), 2)
        self.assertTrue(store._search_indexes_ready)

    async def test_backend_bootstrap_reports_manual_setup_when_index_api_unavailable(self):
        fake_collection = _FakeMongoCollectionForVectorStore(search_index_api_available=False)
        store, _ = self._build_store()
        store._collection = fake_collection

        with self.assertRaises(RuntimeError) as ctx:
            await store.ensure_backend_ready(create_indexes=True)

        self.assertIn("manually in Atlas UI/CLI", str(ctx.exception))

    async def test_backend_bootstrap_reports_manual_setup_on_permission_failure(self):
        fake_collection = _FakeMongoCollectionForVectorStore(
            search_indexes=[],
            create_search_index_error=OperationFailure("not authorized"),
        )
        store, _ = self._build_store()
        store._collection = fake_collection

        with self.assertRaises(RuntimeError) as ctx:
            await store.ensure_backend_ready(create_indexes=True)

        self.assertIn("manually in Atlas UI/CLI", str(ctx.exception))


class KnowledgeBaseSkillCustomizationTests(unittest.TestCase):
    def _make_skill(self, **kwargs):
        # KnowledgeBaseSkill does not access vector_store at construction time,
        # so a simple sentinel object is sufficient for these unit tests.
        return KnowledgeBaseSkill(vector_store=object(), **kwargs)

    def test_default_expertise_contains_domain_text(self):
        skill = self._make_skill(domain="documentacion interna")

        self.assertIn("documentacion interna", skill.expertise)
        self.assertIn("search_knowledge_base", skill.expertise)
        self.assertEqual(skill.name, "knowledge_base")
        self.assertIn("documentacion interna", skill.description)

    def test_instructions_template_replace_mode_overrides_default_expertise(self):
        skill = self._make_skill(
            domain="leyes laborales",
            instructions_template="Responde en espanol y cita fuentes de {domain}.",
            instructions_mode="replace",
        )

        self.assertIn("Responde en espanol", skill.expertise)
        self.assertIn("leyes laborales", skill.expertise)
        self.assertNotIn("When to Use the Knowledge Base", skill.expertise)

    def test_instructions_template_append_mode_keeps_default_expertise(self):
        skill = self._make_skill(
            domain="politicas HR",
            instructions_template="IMPORTANTE: Responde solo en espanol.",
            instructions_mode="append",
        )

        self.assertIn("When to Use the Knowledge Base", skill.expertise)
        self.assertIn("IMPORTANTE: Responde solo en espanol.", skill.expertise)

    def test_expertise_builder_is_supported(self):
        skill = self._make_skill(
            domain="normativa",
            expertise_builder=lambda domain: f"Guia especializada para {domain}.",
            instructions_mode="replace",
        )

        self.assertEqual(skill.expertise, "Guia especializada para normativa.")

    def test_additional_instructions_still_append(self):
        skill = self._make_skill(
            instructions_template="Base custom text",
            instructions_mode="replace",
            additional_instructions="Anade referencias por articulo.",
        )

        self.assertIn("Base custom text", skill.expertise)
        self.assertIn("Anade referencias por articulo.", skill.expertise)

    def test_template_and_builder_are_mutually_exclusive(self):
        with self.assertRaises(ValueError) as ctx:
            self._make_skill(
                instructions_template="x",
                expertise_builder=lambda domain: "y",
            )

        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_invalid_template_placeholder_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._make_skill(
                instructions_template="Use {unknown}",
            )

        self.assertIn("instructions_template placeholder", str(ctx.exception))


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

    def _make_agent(
        self,
        tool_result,
        tool_error=None,
        first_turn_thinking=None,
        first_turn_thinking_tokens=None,
    ):
        """Build a minimal BaseAgent wired with a fake LLM and a fake tool."""
        # One-shot LLM: first call returns a tool_call, second returns final text
        call_count = {"n": 0}

        class _FakeLLM:
            provider_type = "openai"

            async def chat_completion_stream(self, messages, system_message, tools=None, **kw):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First turn: yield a tool call
                    yield StreamChunk(
                        thinking=first_turn_thinking,
                        thinking_tokens=first_turn_thinking_tokens,
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

    async def test_stream_persists_thinking_metadata_on_tool_call_turn(self):
        agent = self._make_agent(
            tool_result="tool_output",
            first_turn_thinking="inspect weather inputs",
            first_turn_thinking_tokens=11,
        )

        async for _ in agent.stream(
            "go",
            owner_id="thinking-owner",
            chat_id="thinking-chat",
            include_thinking=True,
        ):
            pass

        history = await agent.get_history(owner_id="thinking-owner", chat_id="thinking-chat")
        tool_call_turn = next(
            (message for message in history if message.role == "ai" and message.tool_calls),
            None,
        )

        self.assertIsNotNone(tool_call_turn)
        self.assertEqual(tool_call_turn.thinking, "inspect weather inputs")
        self.assertEqual(tool_call_turn.thinking_tokens, 11)

    async def test_stream_persists_thinking_tokens_on_final_text_turn(self):
        class _ThinkingFinalLLM:
            provider_type = "openai"

            async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
                yield StreamChunk(thinking="draft ", thinking_tokens=3, is_finished=False)
                yield StreamChunk(thinking="answer", thinking_tokens=5, is_finished=False)
                yield StreamChunk(content="done", is_finished=True, finish_reason="stop")

        agent = BaseAgent(
            llm_client=_ThinkingFinalLLM(),
            memory=LocalMemory(rollover_enabled=False),
            system_prompt="sys",
            tools=[],
            max_iterations=4,
        )

        async for _ in agent.stream(
            "go",
            owner_id="final-owner",
            chat_id="final-chat",
            include_thinking=True,
        ):
            pass

        history = await agent.get_history(owner_id="final-owner", chat_id="final-chat")
        ai_messages = [message for message in history if message.role == "ai"]

        self.assertEqual(len(ai_messages), 1)
        self.assertEqual(ai_messages[0].content, "done")
        self.assertEqual(ai_messages[0].thinking, "draft answer")
        self.assertEqual(ai_messages[0].thinking_tokens, 5)

    async def test_public_stream_surface_forwards_tool_call_events(self):
        """Regression: stream() shell was dropping ToolCallEvent chunks because
        it only forwarded chunk.content and chunk.thinking, never chunk.tool_call."""
        agent = self._make_agent(tool_result="tool_output")

        # Use the public stream() API, not _orchestrate_stream directly
        chunks = []
        async for chunk in agent.stream("go", owner_id="test", chat_id="test"):
            chunks.append(chunk)

        tool_events = [c for c in chunks if c.tool_call is not None]
        self.assertEqual(len(tool_events), 2, "Public stream() must forward ToolCallEvent chunks")

        statuses = [e.tool_call.status for e in tool_events]
        self.assertIn("start", statuses)
        self.assertIn("success", statuses)

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

    async def test_stream_include_thinking_auto_sets_gemini_include_thoughts_and_renders(self):
        class _GeminiLikeLLM:
            provider_type = "gemini"

            def __init__(self):
                self.kwargs_seen = None

            async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
                self.kwargs_seen = kwargs
                yield StreamChunk(thinking="internal rationale", is_finished=False)
                yield StreamChunk(content="done", is_finished=True, finish_reason="stop")

        llm = _GeminiLikeLLM()
        agent = BaseAgent(
            llm_client=llm,
            memory=LocalMemory(rollover_enabled=False),
            system_prompt="sys",
            tools=[],
            max_iterations=4,
        )

        chunks = []
        async for chunk in agent.stream("go", include_thinking=True):
            chunks.append(chunk)

        self.assertEqual(chunks[0].thinking, "internal rationale")
        self.assertEqual(str(chunks[0]), "internal rationale")
        self.assertTrue(llm.kwargs_seen.get("include_thoughts"))


class ToolResultEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    class _NoopLLM:
        provider_type = "openai"

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            return ChatResponse(content="ok", role="ai")

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            yield StreamChunk(content="ok", is_finished=True)

    class _StreamingLLM:
        provider_type = "openai"

        def __init__(self):
            self.calls = 0

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            return ChatResponse(content="unused", role="ai")

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield StreamChunk(
                    tool_calls=[
                        ToolCall(
                            id="env_stream_1",
                            name="envelope_success_tool",
                            arguments={"value": 3},
                        )
                    ],
                    is_finished=True,
                    finish_reason="tool_calls",
                )
            else:
                yield StreamChunk(content="done", is_finished=True, finish_reason="stop")

    class _SuccessTool(BaseTool):
        name = "envelope_success_tool"
        description = "Returns success payload"

        async def run(self, value: int = 1, **kwargs):
            return {"ok": True, "value": value}

    class _ErrorTool(BaseTool):
        name = "envelope_error_tool"
        description = "Raises to produce error envelope"

        async def run(self, **kwargs):
            raise ValueError("boom-envelope")

    def _build_agent(self, tools, llm_client=None):
        return BaseAgent(
            llm_client=llm_client or self._NoopLLM(),
            memory=LocalMemory(rollover_enabled=False),
            system_prompt="sys",
            tools=tools,
            max_iterations=4,
        )

    async def test_success_envelope_format(self):
        agent = self._build_agent([self._SuccessTool()])

        tool_messages = await agent._execute_tools_concurrently(
            [ToolCall(id="env_ok_1", name="envelope_success_tool", arguments={"value": 7})]
        )

        self.assertEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0].content)
        envelope = ToolResultEnvelope.model_validate(payload)

        self.assertEqual(envelope.status, "success")
        self.assertEqual(envelope.result, {"ok": True, "value": 7})
        self.assertIsNone(envelope.error)
        self.assertEqual(envelope.metadata.get("tool_name"), "envelope_success_tool")

    async def test_failure_envelope_format(self):
        agent = self._build_agent([self._ErrorTool()])

        tool_messages = await agent._execute_tools_concurrently(
            [ToolCall(id="env_err_1", name="envelope_error_tool", arguments={})]
        )

        self.assertEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0].content)
        envelope = ToolResultEnvelope.model_validate(payload)

        self.assertEqual(envelope.status, "error")
        self.assertIsNone(envelope.result)
        self.assertIn("boom-envelope", envelope.error or "")
        self.assertEqual(envelope.metadata.get("tool_name"), "envelope_error_tool")
        self.assertEqual(envelope.metadata.get("attempt"), 1)
        self.assertEqual(envelope.metadata.get("max_attempts"), 1)

    async def test_stream_path_persists_envelope_json_tool_message(self):
        llm = self._StreamingLLM()
        agent = self._build_agent([self._SuccessTool()], llm_client=llm)
        messages = [Message(role="human", content="run")]

        async for _ in agent._orchestrate_stream(messages):
            pass

        tool_messages = [msg for msg in messages if msg.role == "tool"]
        self.assertEqual(len(tool_messages), 1)

        payload = json.loads(tool_messages[0].content)
        envelope = ToolResultEnvelope.model_validate(payload)
        self.assertEqual(envelope.status, "success")
        self.assertEqual(envelope.result, {"ok": True, "value": 3})
        self.assertEqual(envelope.metadata.get("tool_name"), "envelope_success_tool")

    def test_gemini_decode_keeps_legacy_tool_message_ingestion(self):
        client = GeminiClient.__new__(GeminiClient)
        messages = [
            Message(
                role="ai",
                content="",
                tool_calls=[ToolCall(id="legacy_tc", name="legacy_tool", arguments={})],
            ),
            Message(role="tool", content="NOT_JSON", tool_call_id="legacy_tc"),
        ]

        decoded, _ = client._decode_messages(messages)
        response_payload = decoded[1]["parts"][0]["functionResponse"]["response"]

        self.assertEqual(response_payload, {"result": "NOT_JSON"})

    def test_gemini_decode_accepts_canonical_tool_envelope(self):
        client = GeminiClient.__new__(GeminiClient)
        messages = [
            Message(
                role="ai",
                content="",
                tool_calls=[ToolCall(id="env_tc", name="envelope_tool", arguments={})],
            ),
            Message(
                role="tool",
                content=json.dumps(
                    {
                        "status": "success",
                        "result": {"ok": True},
                        "error": None,
                        "metadata": {"tool_name": "envelope_tool"},
                    }
                ),
                tool_call_id="env_tc",
            ),
        ]

        decoded, _ = client._decode_messages(messages)
        response_payload = decoded[1]["parts"][0]["functionResponse"]["response"]

        self.assertEqual(response_payload.get("status"), "success")
        self.assertEqual(response_payload.get("result"), {"ok": True})


class SummarizerAgentCompatibilityTests(IsolatedAsyncioTestCase):
    class _DummyLLM:
        provider_type = "openai"

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            return ChatResponse(content="unused", role="ai")

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            yield StreamChunk(content="unused", is_finished=True)

    async def test_resolve_summarizer_uses_agent_invoke_contract(self):
        agent = BaseAgent(llm_client=self._DummyLLM(), memory=LocalMemory())

        with patch.object(agent, "invoke", new_callable=AsyncMock) as mocked_invoke:
            mocked_invoke.return_value = "summary-ok"

            result = await resolve_summarizer(
                agent,
                [Message(role="human", content="hola")],
            )

        self.assertEqual(result, "summary-ok")
        mocked_invoke.assert_awaited_once()

    async def test_resolve_summarizer_rejects_non_string_agent_response(self):
        agent = BaseAgent(llm_client=self._DummyLLM(), memory=LocalMemory())

        with patch.object(agent, "invoke", new_callable=AsyncMock) as mocked_invoke:
            mocked_invoke.return_value = ChatResponse(content="not-string-contract", role="ai")

            with self.assertRaises(ValueError) as ctx:
                await resolve_summarizer(
                    agent,
                    [Message(role="human", content="hola")],
                )

        self.assertIn("Expected a string response", str(ctx.exception))


class ObserverLifecycleHookTests(unittest.IsolatedAsyncioTestCase):
    class _LifecycleLLM:
        provider_type = "openai"
        model_name = "observer-test-model"

        def __init__(self):
            self.stream_calls = 0

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            return ChatResponse(
                content="invoke-ok",
                role="ai",
                finish_reason="stop",
                prompt_tokens=4,
                completion_tokens=2,
                total_tokens=6,
            )

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            self.stream_calls += 1
            if self.stream_calls == 1:
                yield StreamChunk(
                    tool_calls=[
                        ToolCall(
                            id="obs_tc_1",
                            name="observer_tool",
                            arguments={"value": 7},
                        )
                    ],
                    is_finished=True,
                    finish_reason="tool_calls",
                )
            else:
                yield StreamChunk(content="stream-ok", is_finished=True, finish_reason="stop")

    class _ObserverTool(BaseTool):
        name = "observer_tool"
        description = "Echoes its value"
        execution_policy = ToolExecutionPolicy(
            max_retries=0,
            backoff=ToolBackoffPolicy(
                strategy="fixed",
                initial_delay_ms=0,
                max_delay_ms=0,
            ),
        )

        async def run(self, value: int = 0, **kwargs):
            return f"value:{value}"

    class _FailingObserver:
        async def on_request_start(self, event):
            raise RuntimeError("observer failed")

        async def on_request_end(self, event):
            raise RuntimeError("observer failed")

        async def on_model_call_start(self, event):
            raise RuntimeError("observer failed")

        async def on_model_call_end(self, event):
            raise RuntimeError("observer failed")

        async def on_tool_call_start(self, event):
            raise RuntimeError("observer failed")

        async def on_tool_call_end(self, event):
            raise RuntimeError("observer failed")

        async def on_error(self, event):
            raise RuntimeError("observer failed")

    def _build_agent(self, llm_client, observers=None, tools=None):
        return BaseAgent(
            llm_client=llm_client,
            memory=LocalMemory(rollover_enabled=False),
            system_prompt="sys",
            tools=tools or [],
            observers=observers or [],
            max_iterations=4,
        )

    async def test_invoke_success_emits_request_start_and_end(self):
        observer = InMemoryObserver()
        agent = self._build_agent(self._LifecycleLLM(), observers=[observer])

        result = await agent.invoke("hello", owner_id="tenant-a", chat_id="chat-a")

        self.assertEqual(result, "invoke-ok")
        request_start_events = observer.get_events("on_request_start")
        request_end_events = observer.get_events("on_request_end")

        self.assertEqual(len(request_start_events), 1)
        self.assertEqual(len(request_end_events), 1)
        self.assertEqual(request_start_events[0]["flow"], "invoke")
        self.assertEqual(request_start_events[0]["request_id"], request_end_events[0]["request_id"])
        self.assertTrue(request_end_events[0]["success"])

    async def test_stream_emits_model_and_tool_lifecycle_events(self):
        observer = InMemoryObserver()
        agent = self._build_agent(
            self._LifecycleLLM(),
            observers=[observer],
            tools=[self._ObserverTool()],
        )

        chunks = []
        async for chunk in agent.stream("run", owner_id="tenant-b", chat_id="chat-b"):
            chunks.append(chunk)

        self.assertTrue(any(chunk.content == "stream-ok" for chunk in chunks))

        model_start_events = observer.get_events("on_model_call_start")
        model_end_events = observer.get_events("on_model_call_end")
        tool_start_events = observer.get_events("on_tool_call_start")
        tool_end_events = observer.get_events("on_tool_call_end")

        self.assertGreaterEqual(len(model_start_events), 2)
        self.assertEqual(len(model_start_events), len(model_end_events))
        self.assertEqual(len(tool_start_events), 1)
        self.assertEqual(len(tool_end_events), 1)

        tool_start = tool_start_events[0]
        self.assertEqual(tool_start["flow"], "stream")
        self.assertEqual(tool_start["tool_name"], "observer_tool")
        self.assertIn("attempt", tool_start)
        self.assertIn("max_attempts", tool_start)

    async def test_observer_exception_does_not_fail_request(self):
        failing_observer = self._FailingObserver()
        healthy_observer = InMemoryObserver()
        agent = self._build_agent(
            self._LifecycleLLM(),
            observers=[failing_observer, healthy_observer],
        )

        with self.assertLogs("syndicate.agents.base", level="WARNING") as captured:
            result = await agent.invoke("hello")

        self.assertEqual(result, "invoke-ok")
        self.assertEqual(len(healthy_observer.get_events("on_request_end")), 1)
        self.assertTrue(
            any("Observer" in message and "failed" in message for message in captured.output),
            msg="Expected observer failure warning log was not emitted",
        )


class ToolExecutionPolicyResilienceTests(unittest.IsolatedAsyncioTestCase):
    class _NoopLLM:
        provider_type = "openai"

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            return ChatResponse(content="ok", role="ai")

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            yield StreamChunk(content="ok", is_finished=True)

    def _build_agent(self, tools):
        return BaseAgent(
            llm_client=self._NoopLLM(),
            memory=LocalMemory(rollover_enabled=False),
            system_prompt="sys",
            tools=tools,
        )

    async def test_timeout_retried_until_max_retries(self):
        class _TimeoutTool(BaseTool):
            name = "timeout_tool"
            description = "Always times out"
            execution_policy = ToolExecutionPolicy(
                timeout_ms=5,
                max_retries=2,
                backoff=ToolBackoffPolicy(
                    strategy="fixed",
                    initial_delay_ms=0,
                    max_delay_ms=0,
                ),
            )

            def __init__(self):
                self.calls = 0

            async def run(self, **kwargs):
                self.calls += 1
                await asyncio.sleep(0.02)
                return "late"

        tool = _TimeoutTool()
        agent = self._build_agent([tool])

        result = await agent.execute_tool("timeout_tool")

        self.assertFalse(result.get("success"))
        self.assertEqual(tool.calls, 3)
        self.assertIn("timed out", result.get("error", ""))

    async def test_non_retryable_exception_fails_fast(self):
        class _FailFastTool(BaseTool):
            name = "fail_fast_tool"
            description = "Raises non-retryable exception"
            execution_policy = ToolExecutionPolicy(
                max_retries=3,
                retryable_errors=["TimeoutError"],
                backoff=ToolBackoffPolicy(
                    strategy="fixed",
                    initial_delay_ms=0,
                    max_delay_ms=0,
                ),
            )

            def __init__(self):
                self.calls = 0

            async def run(self, **kwargs):
                self.calls += 1
                raise ValueError("boom")

        tool = _FailFastTool()
        agent = self._build_agent([tool])

        result = await agent.execute_tool("fail_fast_tool")

        self.assertFalse(result.get("success"))
        self.assertEqual(tool.calls, 1)
        self.assertIn("boom", result.get("error", ""))

    async def test_policy_retries_support_async_and_sync_tools(self):
        class _FlakyAsyncTool(BaseTool):
            name = "flaky_async_tool"
            description = "Fails once then succeeds (async)"
            execution_policy = ToolExecutionPolicy(
                max_retries=1,
                retryable_errors=["ValueError"],
                backoff=ToolBackoffPolicy(
                    strategy="fixed",
                    initial_delay_ms=0,
                    max_delay_ms=0,
                ),
            )

            def __init__(self):
                self.calls = 0

            async def run(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("transient")
                return "async-ok"

        class _FlakySyncTool(BaseTool):
            name = "flaky_sync_tool"
            description = "Fails once then succeeds (sync)"
            execution_policy = ToolExecutionPolicy(
                max_retries=1,
                retryable_errors=["ValueError"],
                backoff=ToolBackoffPolicy(
                    strategy="fixed",
                    initial_delay_ms=0,
                    max_delay_ms=0,
                ),
            )

            def __init__(self):
                self.calls = 0

            def run(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("transient")
                return "sync-ok"

        async_tool = _FlakyAsyncTool()
        sync_tool = _FlakySyncTool()
        agent = self._build_agent([async_tool, sync_tool])

        async_result = await agent.execute_tool("flaky_async_tool")
        sync_result = await agent.execute_tool("flaky_sync_tool")

        self.assertTrue(async_result.get("success"))
        self.assertEqual(async_result.get("result"), "async-ok")
        self.assertEqual(async_tool.calls, 2)

        self.assertTrue(sync_result.get("success"))
        self.assertEqual(sync_result.get("result"), "sync-ok")
        self.assertEqual(sync_tool.calls, 2)


class ToolGuardrailTests(unittest.IsolatedAsyncioTestCase):
    class _InvokeGuardrailLLM:
        provider_type = "openai"

        def __init__(self, tool_call_batch_size):
            self.tool_call_batch_size = tool_call_batch_size
            self.calls = 0

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResponse(
                    content="",
                    role="ai",
                    tool_calls=[
                        ToolCall(
                            id=f"guardrail_tc_{i}",
                            name="guardrail_tool",
                            arguments={"idx": i},
                        )
                        for i in range(self.tool_call_batch_size)
                    ],
                    finish_reason="tool_calls",
                )
            return ChatResponse(content="done", role="ai", finish_reason="stop")

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            yield StreamChunk(content="unused", is_finished=True, finish_reason="stop")

    class _StreamGuardrailLLM:
        provider_type = "openai"

        def __init__(self, tool_call_batch_size):
            self.tool_call_batch_size = tool_call_batch_size
            self.calls = 0

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            return ChatResponse(content="unused", role="ai")

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield StreamChunk(
                    tool_calls=[
                        ToolCall(
                            id=f"stream_guardrail_tc_{i}",
                            name="guardrail_tool",
                            arguments={"idx": i},
                        )
                        for i in range(self.tool_call_batch_size)
                    ],
                    is_finished=True,
                    finish_reason="tool_calls",
                )
            else:
                yield StreamChunk(content="done", is_finished=True, finish_reason="stop")

    class _CountingTool(BaseTool):
        name = "guardrail_tool"
        description = "Counts invocations"

        def __init__(self):
            self.calls = 0

        async def run(self, idx: int = 0, **kwargs):
            self.calls += 1
            return f"ok:{idx}"

    class _ConcurrencyTrackingTool(BaseTool):
        name = "concurrency_tool"
        description = "Tracks concurrent executions"

        def __init__(self):
            self.calls = 0
            self.active = 0
            self.peak_active = 0
            self._lock = asyncio.Lock()

        async def run(self, idx: int = 0, **kwargs):
            async with self._lock:
                self.calls += 1
                self.active += 1
                if self.active > self.peak_active:
                    self.peak_active = self.active

            try:
                await asyncio.sleep(0.03)
                return f"ok:{idx}"
            finally:
                async with self._lock:
                    self.active -= 1

    class _ConcurrencyLLM:
        provider_type = "openai"

        def __init__(self, tool_call_batch_size):
            self.tool_call_batch_size = tool_call_batch_size
            self.calls = 0

        async def chat_completion_async(self, messages, system_message, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResponse(
                    content="",
                    role="ai",
                    tool_calls=[
                        ToolCall(
                            id=f"conc_tc_{i}",
                            name="concurrency_tool",
                            arguments={"idx": i},
                        )
                        for i in range(self.tool_call_batch_size)
                    ],
                    finish_reason="tool_calls",
                )
            return ChatResponse(content="done", role="ai", finish_reason="stop")

        async def chat_completion_stream(self, messages, system_message, tools=None, **kwargs):
            yield StreamChunk(content="unused", is_finished=True, finish_reason="stop")

    def _build_agent(self, llm_client, tools, **kwargs):
        return BaseAgent(
            llm_client=llm_client,
            memory=LocalMemory(rollover_enabled=False),
            system_prompt="sys",
            tools=tools,
            max_iterations=5,
            **kwargs,
        )

    async def test_invoke_guardrail_max_total_tool_calls_blocks_large_batch(self):
        tool = self._CountingTool()
        agent = self._build_agent(
            self._InvokeGuardrailLLM(tool_call_batch_size=2),
            tools=[tool],
            max_total_tool_calls=1,
        )

        result = await agent.invoke("run")

        self.assertIn("Guardrail reached", result)
        self.assertIn("max_total_tool_calls", result)
        self.assertEqual(tool.calls, 0)

    async def test_stream_guardrail_emits_terminal_guardrail_reason(self):
        tool = self._CountingTool()
        agent = self._build_agent(
            self._StreamGuardrailLLM(tool_call_batch_size=2),
            tools=[tool],
            max_total_tool_calls=1,
        )

        chunks = []
        async for chunk in agent.stream("run"):
            chunks.append(chunk)

        self.assertGreaterEqual(len(chunks), 1)
        terminal = chunks[-1]
        self.assertTrue(terminal.is_finished)
        self.assertEqual(terminal.finish_reason, "guardrail_reached")
        self.assertIn("max_total_tool_calls", terminal.content)
        self.assertEqual(tool.calls, 0)

    async def test_max_concurrent_tool_calls_respected(self):
        tool = self._ConcurrencyTrackingTool()
        agent = self._build_agent(
            self._ConcurrencyLLM(tool_call_batch_size=3),
            tools=[tool],
            max_concurrent_tool_calls=2,
        )

        result = await agent.invoke("run")

        self.assertEqual(result, "done")
        self.assertEqual(tool.calls, 3)
        self.assertLessEqual(tool.peak_active, 2)


class RAGSearchToolExtensibilityTests(unittest.IsolatedAsyncioTestCase):
    """Tests for RAGSearchTool extension points: format_results, _format_single_result, default_filter."""

    def _make_fake_vector_store(self, results=None, search_error=None):
        """Build a minimal fake vector store for testing."""
        fake = _FakeAsyncHTTPClient.__new__(type("_FakeVectorStore", (), {}))
        fake.search = AsyncMock(return_value=results)
        return fake

    def _sample_results(self):
        return [
            {"id": "doc1", "text": "Alpha result", "score": 0.95, "metadata": {"source": "A"}},
            {"id": "doc2", "text": "Beta result", "score": 0.82, "metadata": {"source": "B"}},
        ]

    async def test_default_format_behavior_unchanged(self):
        """Default format_results produces the standard multi-result block."""
        from syndicate.tools.rag_tool import RAGSearchTool

        store = self._make_fake_vector_store(results=self._sample_results())
        tool = RAGSearchTool(vector_store=store, top_k=3)
        execution = await tool._execute("test query")
        output = tool.format_results(execution)

        self.assertIn("Alpha result", output)
        self.assertIn("Beta result", output)
        self.assertIn("Relevance: 0.950", output)
        self.assertIn("Relevance: 0.820", output)

    async def test_subclass_can_override_format_results(self):
        """A subclass can replace format_results entirely."""

        from syndicate.tools.rag_tool import RAGSearchTool

        class _CustomFormatTool(RAGSearchTool):
            def format_results(self, execution_result):
                results = execution_result.get("results", [])
                return f"Found {len(results)} documents"

        store = self._make_fake_vector_store(results=self._sample_results())
        tool = _CustomFormatTool(vector_store=store)
        execution = await tool._execute("test query")
        output = tool.format_results(execution)

        self.assertEqual(output, "Found 2 documents")

    async def test_subclass_can_override_format_single_result(self):
        """A subclass can override _format_single_result for per-result styling."""

        from syndicate.tools.rag_tool import RAGSearchTool

        class _StyledTool(RAGSearchTool):
            def _format_single_result(self, result, index):
                text = result.get("text", "")
                return f"[{index}] {text.upper()}"

        store = self._make_fake_vector_store(results=self._sample_results())
        tool = _StyledTool(vector_store=store)
        execution = await tool._execute("test query")
        output = tool.format_results(execution)

        self.assertIn("[1] ALPHA RESULT", output)
        self.assertIn("[2] BETA RESULT", output)

    async def test_filter_forwarding_with_default_filter(self):
        """default_filter is forwarded to vector_store.search()."""
        from syndicate.tools.rag_tool import RAGSearchTool

        store = self._make_fake_vector_store(results=[])
        tool = RAGSearchTool(vector_store=store, default_filter={"department": "engineering"})
        await tool._execute("test query")

        store.search.assert_called_once()
        call_kwargs = store.search.call_args[1]
        self.assertEqual(call_kwargs.get("filter"), {"department": "engineering"})

    async def test_filter_forwarding_without_default_filter(self):
        """When default_filter is None, filter=None is forwarded."""
        from syndicate.tools.rag_tool import RAGSearchTool

        store = self._make_fake_vector_store(results=[])
        tool = RAGSearchTool(vector_store=store)
        await tool._execute("test query")

        store.search.assert_called_once()
        call_kwargs = store.search.call_args[1]
        self.assertIsNone(call_kwargs.get("filter"))

    async def test_empty_results_message(self):
        """format_results returns 'No results found.' for empty lists."""
        from syndicate.tools.rag_tool import RAGSearchTool

        store = self._make_fake_vector_store(results=[])
        tool = RAGSearchTool(vector_store=store)
        execution = await tool._execute("test query")
        output = tool.format_results(execution)

        self.assertEqual(output, "No relevant information found in the knowledge base.")

    async def test_error_in_format_results_is_not_caught(self):
        """Errors inside format_results propagate (not swallowed)."""

        from syndicate.tools.rag_tool import RAGSearchTool

        class _BrokenTool(RAGSearchTool):
            def format_results(self, execution_result):
                raise ValueError("formatting broken")

        store = self._make_fake_vector_store(results=self._sample_results())
        tool = _BrokenTool(vector_store=store)
        execution = await tool._execute("test query")

        with self.assertRaises(ValueError) as ctx:
            tool.format_results(execution)
        self.assertEqual(str(ctx.exception), "formatting broken")

    async def test_deprecation_warning_on_get_result_text(self):
        """Calling get_result_text emits a DeprecationWarning."""
        from syndicate.tools.rag_tool import RAGSearchTool

        store = self._make_fake_vector_store(results=self._sample_results())
        tool = RAGSearchTool(vector_store=store)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tool.get_result_text({"text": "hello", "score": 0.5})
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("get_result_text", str(w[0].message))


if __name__ == "__main__":
    unittest.main(verbosity=2)
