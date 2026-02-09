"""GenAIChatModel — LangChain BaseChatModel backed by GenAiApiClient.

This module provides ``GenAIChatModel``, a thin wrapper that delegates
chat completions to :class:`genai_api_client.GenAiApiClient` while
exposing the full LangChain ``BaseChatModel`` interface.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Callable, Optional, Sequence, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

# GenAiApiClient lives at repo root, which should be on sys.path.
from genai_api_client import GenAiApiClient

from langchain_genai._utils import convert_messages_to_dicts, convert_response_to_chat_result


class GenAIChatModel(BaseChatModel):
    """LangChain chat model backed by :class:`GenAiApiClient`.

    All network calls are synchronous and delegate to
    ``GenAiApiClient.get_genai_completion``.  Async support is provided
    via ``run_in_executor`` (see ``_agenerate``).

    Args:
        model: Azure OpenAI deployment / model name (required).
        temperature: Sampling temperature; ``None`` omits from request.
        max_tokens: Maximum completion tokens; ``None`` omits from request.
        config_path: Path to ``config.yaml``.  ``None`` uses the client's
            built-in default.
        model_kwargs: Extra keyword arguments forwarded to every
            completion call (e.g. ``top_p``, ``frequency_penalty``).

    Example:
        .. code-block:: python

            from langchain_genai import GenAIChatModel
            from langchain_core.messages import HumanMessage

            model = GenAIChatModel(model="gpt-4.1-nano")
            response = model.invoke([HumanMessage(content="Hello!")])
            print(response.content)
    """

    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    config_path: Optional[str] = None
    model_kwargs: dict[str, Any] = {}

    # --- internal state (excluded from Pydantic serialization) ---
    _client: GenAiApiClient = None  # type: ignore[assignment]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Instantiate the underlying GenAiApiClient."""
        super().model_post_init(__context)
        self._client = (
            GenAiApiClient(config_path=self.config_path)
            if self.config_path
            else GenAiApiClient()
        )

    # ------------------------------------------------------------------
    # BaseChatModel interface
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "genai-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call GenAiApiClient.get_genai_completion and return a ChatResult.

        Args:
            messages: The prompt messages.
            stop: Optional stop sequences.
            run_manager: Callback manager (unused by wrapper).
            **kwargs: Extra params merged with ``model_kwargs``.

        Returns:
            A ``ChatResult`` with one ``ChatGeneration`` containing an
            ``AIMessage``.
        """
        msg_dicts = convert_messages_to_dicts(messages)

        # Build the parameter dict.
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if stop:
            params["stop"] = stop

        # model_kwargs are defaults; explicit kwargs override them.
        merged = {**self.model_kwargs, **params, **kwargs}

        response = self._client.get_genai_completion(
            messages=msg_dicts,
            model=self.model,
            **merged,
        )
        return convert_response_to_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of ``_generate`` using ``run_in_executor``.

        Delegates to the synchronous ``_generate`` in a thread-pool
        executor so that the event loop is not blocked.
        """
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Return LangSmith tracing metadata for this model invocation."""
        params = LangSmithParams(
            ls_provider="genai",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=self.temperature,
        )
        if self.max_tokens is not None:
            params["ls_max_tokens"] = self.max_tokens
        if stop:
            params["ls_stop"] = stop
        return params

    # ------------------------------------------------------------------
    # Tool calling  (T017)
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | dict | bool | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind tool definitions to this model.

        Args:
            tools: Tool definitions (dicts, Pydantic classes, callables,
                or ``BaseTool`` instances).
            tool_choice: Which tool to require.  Accepts a tool name
                string, ``True`` / ``"any"`` / ``"required"`` to force a
                call, ``"auto"`` / ``"none"`` for default behaviour, or
                a raw dict.
            **kwargs: Additional keyword arguments passed to ``bind``.

        Returns:
            A ``Runnable`` with the tools bound as kwargs.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        # Collect tool names for choice normalisation.
        tool_names: list[str] = []
        for t in formatted_tools:
            if "function" in t:
                tool_names.append(t["function"]["name"])
            elif "name" in t:
                tool_names.append(t["name"])

        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice == "any":
                    tool_choice = "required"
                # "auto", "none", "required" pass through as-is.
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            kwargs["tool_choice"] = tool_choice

        return super().bind(tools=formatted_tools, **kwargs)

    # ------------------------------------------------------------------
    # Structured output  (T018)
    # ------------------------------------------------------------------

    def with_structured_output(
        self,
        schema: type | dict[str, Any],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Return a Runnable that parses model output into *schema*.

        Uses the ``function_calling`` method: binds the schema as a tool
        and attaches an output parser.

        Args:
            schema: A Pydantic class or JSON-schema dict describing the
                desired output structure.
            include_raw: If ``True``, return a dict with ``raw``,
                ``parsed``, and ``parsing_error`` keys.
            **kwargs: Extra keyword arguments forwarded to ``bind``.

        Returns:
            A ``Runnable`` producing instances of *schema* (or a dict
            when *include_raw* is ``True``).
        """
        tool_def = convert_to_openai_tool(schema)
        tool_name = tool_def["function"]["name"]

        llm = self.bind_tools([schema], tool_choice=tool_name, **kwargs)

        is_pydantic = isinstance(schema, type) and _is_pydantic_class(schema)

        if is_pydantic:
            output_parser: Runnable = PydanticToolsParser(
                tools=[schema],  # type: ignore[list-item]
                first_tool_only=True,
            )
        else:
            output_parser = JsonOutputKeyToolsParser(
                key_name=tool_name, first_tool_only=True
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback

        return llm | output_parser


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_pydantic_class(obj: Any) -> bool:
    """Return ``True`` if *obj* is a Pydantic model class."""
    try:
        from pydantic import BaseModel as PydanticBaseModel

        return isinstance(obj, type) and issubclass(obj, PydanticBaseModel)
    except ImportError:
        return False