"""Unit tests for langchain_genai.chat_model.GenAIChatModel."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel as PydanticBaseModel

from langchain_genai.chat_model import GenAIChatModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_completion(
    content: str = "Hello!",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4.1-nano",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> SimpleNamespace:
    """Build a fake ChatCompletion-like response."""
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


@pytest.fixture()
def mock_client() -> MagicMock:
    """Return a mocked GenAiApiClient."""
    client = MagicMock()
    client.get_genai_completion.return_value = _fake_completion()
    return client


@pytest.fixture()
def chat_model(mock_client: MagicMock) -> GenAIChatModel:
    """Return a GenAIChatModel with a mocked client."""
    with patch("langchain_genai.chat_model.GenAiApiClient", return_value=mock_client):
        model = GenAIChatModel(model="gpt-4.1-nano")
    return model


# ---------------------------------------------------------------------------
# T010 — Tests for GenAIChatModel._generate
# ---------------------------------------------------------------------------


class TestGenAIChatModelGenerate:
    """Tests for GenAIChatModel._generate via invoke."""

    def test_single_message_invocation(
        self, chat_model: GenAIChatModel, mock_client: MagicMock
    ) -> None:
        result = chat_model.invoke([HumanMessage(content="Hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

        # Verify the client was called with correct args.
        mock_client.get_genai_completion.assert_called_once()
        call_kwargs = mock_client.get_genai_completion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4.1-nano"
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    def test_multi_message_conversation(
        self, chat_model: GenAIChatModel, mock_client: MagicMock
    ) -> None:
        msgs = [
            SystemMessage(content="Be helpful."),
            HumanMessage(content="What is 2+2?"),
        ]
        result = chat_model.invoke(msgs)
        assert isinstance(result, AIMessage)
        call_kwargs = mock_client.get_genai_completion.call_args
        assert len(call_kwargs.kwargs["messages"]) == 2
        assert call_kwargs.kwargs["messages"][0]["role"] == "system"
        assert call_kwargs.kwargs["messages"][1]["role"] == "user"

    def test_temperature_and_max_tokens_forwarding(
        self, mock_client: MagicMock
    ) -> None:
        with patch("langchain_genai.chat_model.GenAiApiClient", return_value=mock_client):
            model = GenAIChatModel(
                model="gpt-4.1-nano", temperature=0.5, max_tokens=200
            )
        model.invoke([HumanMessage(content="test")])
        call_kwargs = mock_client.get_genai_completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 200

    def test_model_kwargs_passthrough(self, mock_client: MagicMock) -> None:
        with patch("langchain_genai.chat_model.GenAiApiClient", return_value=mock_client):
            model = GenAIChatModel(
                model="gpt-4.1-nano", model_kwargs={"top_p": 0.9, "seed": 42}
            )
        model.invoke([HumanMessage(content="test")])
        call_kwargs = mock_client.get_genai_completion.call_args
        assert call_kwargs.kwargs["top_p"] == 0.9
        assert call_kwargs.kwargs["seed"] == 42

    def test_usage_metadata_present(
        self, chat_model: GenAIChatModel, mock_client: MagicMock
    ) -> None:
        result = chat_model.invoke([HumanMessage(content="Hi")])
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] == 10
        assert result.usage_metadata["output_tokens"] == 5
        assert result.usage_metadata["total_tokens"] == 15

    def test_llm_type(self, chat_model: GenAIChatModel) -> None:
        assert chat_model._llm_type == "genai-chat"

    def test_stop_sequences_forwarded(
        self, chat_model: GenAIChatModel, mock_client: MagicMock
    ) -> None:
        chat_model.invoke([HumanMessage(content="test")], stop=["END"])
        call_kwargs = mock_client.get_genai_completion.call_args
        assert call_kwargs.kwargs["stop"] == ["END"]


# ---------------------------------------------------------------------------
# T019 — Tests for bind_tools
# ---------------------------------------------------------------------------


class TestBindTools:
    """Tests for GenAIChatModel.bind_tools."""

    def test_bind_tools_with_function(self, chat_model: GenAIChatModel) -> None:
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Sunny in {location}"

        bound = chat_model.bind_tools([get_weather])
        # The bound model should have tools in its kwargs.
        assert "tools" in bound.kwargs
        assert len(bound.kwargs["tools"]) == 1
        assert bound.kwargs["tools"][0]["function"]["name"] == "get_weather"

    def test_bind_tools_with_pydantic_model(
        self, chat_model: GenAIChatModel
    ) -> None:
        class Weather(PydanticBaseModel):
            location: str
            temperature: float

        bound = chat_model.bind_tools([Weather])
        assert "tools" in bound.kwargs
        assert bound.kwargs["tools"][0]["function"]["name"] == "Weather"

    def test_bind_tools_with_dict_schema(
        self, chat_model: GenAIChatModel
    ) -> None:
        tool_dict = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a term",
                "parameters": {
                    "type": "object",
                    "properties": {"term": {"type": "string"}},
                    "required": ["term"],
                },
            },
        }
        bound = chat_model.bind_tools([tool_dict])
        assert "tools" in bound.kwargs

    def test_tool_choice_string_name(self, chat_model: GenAIChatModel) -> None:
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        bound = chat_model.bind_tools([my_tool], tool_choice="my_tool")
        assert bound.kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "my_tool"},
        }

    def test_tool_choice_bool_true(self, chat_model: GenAIChatModel) -> None:
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        bound = chat_model.bind_tools([my_tool], tool_choice=True)
        assert bound.kwargs["tool_choice"] == "required"

    def test_tool_choice_any(self, chat_model: GenAIChatModel) -> None:
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        bound = chat_model.bind_tools([my_tool], tool_choice="any")
        assert bound.kwargs["tool_choice"] == "required"


# ---------------------------------------------------------------------------
# T020 — Tests for with_structured_output
# ---------------------------------------------------------------------------


class TestWithStructuredOutput:
    """Tests for GenAIChatModel.with_structured_output."""

    def test_pydantic_schema_produces_pydantic_parser(
        self, chat_model: GenAIChatModel
    ) -> None:
        class Capital(PydanticBaseModel):
            city: str
            country: str

        chain = chat_model.with_structured_output(Capital)
        # The chain should be a RunnableSequence ending with a PydanticToolsParser.
        assert isinstance(chain, RunnableSequence)
        assert isinstance(chain.last, PydanticToolsParser)

    def test_dict_schema_produces_json_parser(
        self, chat_model: GenAIChatModel
    ) -> None:
        dict_schema = {
            "name": "info",
            "description": "Info extraction",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
        chain = chat_model.with_structured_output(dict_schema)
        assert isinstance(chain, RunnableSequence)
        assert isinstance(chain.last, JsonOutputKeyToolsParser)

    def test_include_raw_produces_runnable_map(
        self, chat_model: GenAIChatModel
    ) -> None:
        class Capital(PydanticBaseModel):
            city: str
            country: str

        chain = chat_model.with_structured_output(Capital, include_raw=True)
        # include_raw wraps with RunnableMap | parser — should be a RunnableSequence
        assert isinstance(chain, RunnableSequence)


# ---------------------------------------------------------------------------
# T022 — Tests for _agenerate (async)
# ---------------------------------------------------------------------------


class TestAgenerate:
    """Async tests for GenAIChatModel._agenerate via ainvoke."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_ai_message(
        self, chat_model: GenAIChatModel, mock_client: MagicMock
    ) -> None:
        result = await chat_model.ainvoke([HumanMessage(content="Hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] == 10
        mock_client.get_genai_completion.assert_called_once()


# ---------------------------------------------------------------------------
# T024 — Tests for _get_ls_params (LangSmith tracing)
# ---------------------------------------------------------------------------


class TestGetLsParams:
    """Tests for GenAIChatModel._get_ls_params."""

    def test_basic_params(self, chat_model: GenAIChatModel) -> None:
        params = chat_model._get_ls_params()
        assert params["ls_provider"] == "genai"
        assert params["ls_model_name"] == "gpt-4.1-nano"
        assert params["ls_model_type"] == "chat"

    def test_temperature_included(self, mock_client: MagicMock) -> None:
        with patch("langchain_genai.chat_model.GenAiApiClient", return_value=mock_client):
            model = GenAIChatModel(model="gpt-4.1-nano", temperature=0.7)
        params = model._get_ls_params()
        assert params["ls_temperature"] == 0.7

    def test_max_tokens_included(self, mock_client: MagicMock) -> None:
        with patch("langchain_genai.chat_model.GenAiApiClient", return_value=mock_client):
            model = GenAIChatModel(model="gpt-4.1-nano", max_tokens=512)
        params = model._get_ls_params()
        assert params["ls_max_tokens"] == 512

    def test_stop_included(self, chat_model: GenAIChatModel) -> None:
        params = chat_model._get_ls_params(stop=["END", "STOP"])
        assert params["ls_stop"] == ["END", "STOP"]

    def test_no_optional_fields_when_none(
        self, chat_model: GenAIChatModel
    ) -> None:
        params = chat_model._get_ls_params()
        assert params.get("ls_temperature") is None
        assert "ls_max_tokens" not in params
        assert "ls_stop" not in params