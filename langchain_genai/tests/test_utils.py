"""Unit tests for langchain_genai._utils conversion functions."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_genai._utils import convert_messages_to_dicts, convert_response_to_chat_result


# ---------------------------------------------------------------------------
# Tests for convert_messages_to_dicts  (T006)
# ---------------------------------------------------------------------------


class TestConvertMessagesToDicts:
    """Tests for convert_messages_to_dicts."""

    def test_human_message(self) -> None:
        msgs: list[BaseMessage] = [HumanMessage(content="hello")]
        result = convert_messages_to_dicts(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_ai_message_plain(self) -> None:
        msgs: list[BaseMessage] = [AIMessage(content="world")]
        result = convert_messages_to_dicts(msgs)
        assert result == [{"role": "assistant", "content": "world"}]

    def test_system_message(self) -> None:
        msgs: list[BaseMessage] = [SystemMessage(content="be nice")]
        result = convert_messages_to_dicts(msgs)
        assert result == [{"role": "system", "content": "be nice"}]

    def test_tool_message(self) -> None:
        msgs: list[BaseMessage] = [
            ToolMessage(content="result data", tool_call_id="call_abc")
        ]
        result = convert_messages_to_dicts(msgs)
        assert result == [
            {"role": "tool", "content": "result data", "tool_call_id": "call_abc"}
        ]

    def test_multi_part_content_passthrough(self) -> None:
        """Multi-part content blocks (list) pass through as-is."""
        content_blocks = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msgs: list[BaseMessage] = [HumanMessage(content=content_blocks)]
        result = convert_messages_to_dicts(msgs)
        assert result == [{"role": "user", "content": content_blocks}]

    def test_ai_message_with_tool_calls(self) -> None:
        """AIMessage.tool_calls are converted to OpenAI format."""
        msgs: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"location": "Paris"},
                        "id": "call_123",
                        "type": "tool_call",
                    }
                ],
            )
        ]
        result = convert_messages_to_dicts(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
        tc = result[0]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["id"] == "call_123"
        assert tc[0]["type"] == "function"
        assert tc[0]["function"]["name"] == "get_weather"
        assert json.loads(tc[0]["function"]["arguments"]) == {"location": "Paris"}

    def test_ai_message_with_additional_kwargs_tool_calls(self) -> None:
        """Fall back to additional_kwargs['tool_calls'] if .tool_calls is empty."""
        raw_tc = [
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}',
                },
            }
        ]
        msgs: list[BaseMessage] = [
            AIMessage(content="", additional_kwargs={"tool_calls": raw_tc})
        ]
        result = convert_messages_to_dicts(msgs)
        assert result[0]["tool_calls"] == raw_tc

    def test_unsupported_message_type_raises(self) -> None:
        """Unsupported message types raise TypeError."""

        class FakeMessage(BaseMessage):
            type: str = "fake"

        msgs: list[BaseMessage] = [FakeMessage(content="bad")]  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="Unsupported message type"):
            convert_messages_to_dicts(msgs)

    def test_multiple_messages(self) -> None:
        msgs: list[BaseMessage] = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        result = convert_messages_to_dicts(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_empty_list(self) -> None:
        assert convert_messages_to_dicts([]) == []


# ---------------------------------------------------------------------------
# Tests for convert_response_to_chat_result  (T007)
# ---------------------------------------------------------------------------


def _make_response(
    content: str = "Hello!",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4.1-nano",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    include_usage: bool = True,
) -> SimpleNamespace:
    """Build a fake ChatCompletion-like response object."""
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )
    usage = (
        SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        if include_usage
        else None
    )
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model,
    )


class TestConvertResponseToChatResult:
    """Tests for convert_response_to_chat_result."""

    def test_plain_text_response(self) -> None:
        resp = _make_response(content="Paris is the capital of France.")
        result = convert_response_to_chat_result(resp)

        assert len(result.generations) == 1
        gen = result.generations[0]
        assert gen.message.content == "Paris is the capital of France."
        assert gen.message.type == "ai"
        assert gen.generation_info == {"finish_reason": "stop"}
        assert result.llm_output == {"model_name": "gpt-4.1-nano"}

    def test_usage_metadata(self) -> None:
        resp = _make_response(prompt_tokens=12, completion_tokens=8, total_tokens=20)
        result = convert_response_to_chat_result(resp)
        ai = result.generations[0].message
        assert ai.usage_metadata is not None
        assert ai.usage_metadata["input_tokens"] == 12
        assert ai.usage_metadata["output_tokens"] == 8
        assert ai.usage_metadata["total_tokens"] == 20

    def test_response_with_tool_calls(self) -> None:
        tool_call = SimpleNamespace(
            id="call_001",
            type="function",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"location": "Paris"}',
            ),
        )
        # model_dump is called on non-dict tool_calls; provide it
        tool_call.model_dump = lambda: {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        }
        resp = _make_response(content="", tool_calls=[tool_call])
        result = convert_response_to_chat_result(resp)

        ai = result.generations[0].message
        assert len(ai.tool_calls) == 1
        assert ai.tool_calls[0]["name"] == "get_weather"
        assert ai.tool_calls[0]["args"] == {"location": "Paris"}
        assert ai.tool_calls[0]["id"] == "call_001"

    def test_no_usage(self) -> None:
        resp = _make_response(include_usage=False)
        result = convert_response_to_chat_result(resp)
        ai = result.generations[0].message
        assert ai.usage_metadata is None

    def test_none_content(self) -> None:
        resp = _make_response(content=None)  # type: ignore[arg-type]
        # Manually set content to None on the message
        resp.choices[0].message.content = None
        result = convert_response_to_chat_result(resp)
        assert result.generations[0].message.content == ""

    def test_missing_model(self) -> None:
        resp = _make_response()
        resp.model = None
        result = convert_response_to_chat_result(resp)
        assert result.llm_output == {}
