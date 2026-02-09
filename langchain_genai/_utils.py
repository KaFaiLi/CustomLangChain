"""Message conversion utilities for langchain_genai.

Provides helpers to convert between LangChain message types and
OpenAI-format dictionaries, and to convert OpenAI ChatCompletion
responses into LangChain ChatResult objects.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers.openai_tools import parse_tool_call
from langchain_core.outputs import ChatGeneration, ChatResult


def convert_messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert a list of LangChain messages to OpenAI-format dicts.

    Args:
        messages: LangChain message objects to convert.

    Returns:
        List of dicts with ``role``, ``content``, and optional fields.

    Raises:
        TypeError: If a message type is not supported.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            d: dict[str, Any] = {"role": "assistant", "content": msg.content}
            # Collect tool_calls from the canonical .tool_calls attribute or
            # from additional_kwargs (older LangChain paths).
            tool_calls = _extract_tool_calls(msg)
            if tool_calls:
                d["tool_calls"] = tool_calls
            result.append(d)
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        else:
            raise TypeError(
                f"Unsupported message type: {type(msg).__name__}"
            )
    return result


def _extract_tool_calls(msg: AIMessage) -> list[dict[str, Any]] | None:
    """Return OpenAI-format tool_calls from an AIMessage, or *None*."""
    # Prefer the canonical .tool_calls list (list[ToolCall] dicts).
    if msg.tool_calls:
        return [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": (
                        tc["args"]
                        if isinstance(tc["args"], str)
                        else __import__("json").dumps(tc["args"])
                    ),
                },
            }
            for tc in msg.tool_calls
        ]
    # Fall back to additional_kwargs["tool_calls"] (raw OpenAI format).
    raw = msg.additional_kwargs.get("tool_calls")
    if raw:
        return raw  # type: ignore[return-value]
    return None


def convert_response_to_chat_result(response: Any) -> ChatResult:
    """Convert an OpenAI ChatCompletion response to a LangChain ChatResult.

    Args:
        response: The response object returned by
            ``GenAiApiClient.get_genai_completion``.

    Returns:
        A ``ChatResult`` containing one ``ChatGeneration`` with an ``AIMessage``.
    """
    choice = response.choices[0]
    message = choice.message

    content: str = message.content or ""

    # Parse tool_calls if present.
    tool_calls: list[dict[str, Any]] = []
    if getattr(message, "tool_calls", None):
        for raw_tc in message.tool_calls:
            tc_dict = raw_tc if isinstance(raw_tc, dict) else raw_tc.model_dump()
            parsed = parse_tool_call(tc_dict)
            if parsed is not None:
                tool_calls.append(parsed)

    # Build usage metadata.
    usage_metadata = None
    if getattr(response, "usage", None) is not None:
        usage = response.usage
        usage_metadata = {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    ai_message = AIMessage(
        content=content,
        tool_calls=tool_calls,
        usage_metadata=usage_metadata,  # type: ignore[arg-type]
    )

    generation_info: dict[str, Any] = {}
    if getattr(choice, "finish_reason", None) is not None:
        generation_info["finish_reason"] = choice.finish_reason

    generation = ChatGeneration(
        message=ai_message,
        generation_info=generation_info,
    )

    llm_output: dict[str, Any] = {}
    if getattr(response, "model", None) is not None:
        llm_output["model_name"] = response.model

    return ChatResult(
        generations=[generation],
        llm_output=llm_output,
    )
