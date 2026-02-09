from __future__ import annotations

import yaml
from pathlib import Path
from openai import AzureOpenAI

from typing import Any, Dict, List, Optional, Union

CONFIG_PATH = Path(__file__).parent / "config.yaml"

EmbeddingInput = Union[
    str,
    List[str],
    List[int],
    List[List[int]],
]

def _is_multi_input(inp: EmbeddingInput) -> bool:
    """
    Determine whether 'inp' represents multiple inputs.
    - str -> single
    - List[str] -> multi
    - List[int] -> single (token array)
    - List[List[int]] -> multi
    """
    if isinstance(inp, str):
        return False
    if not inp:
        # empty inputs are invalid per API; treat as multi for downstream checks.
        return True
    first = inp[0]
    return not isinstance(first, int)
    # int -> List[int] (single token array); otherwise multi


def _truncate_input(inp: EmbeddingInput, max_length: int) -> EmbeddingInput:
    """
    Non-standard convenience truncation.
    - str / List[str]: truncates by characters.
    - List[int] / List[List[int]]: truncates by token count.
    """
    if max_length <= 0:
        return inp
    
    if isinstance(inp, str):
        return inp[:max_length]
    
    # List[...] cases
    if not inp:
        return inp
    
    if isinstance(inp[0], int):
        # List[int]
        return inp[:max_length] # Token truncation
    else:
        # List[str] or List[List[int]]
        out = []
        for item in inp:
            if isinstance(item, str):
                out.append(item[:max_length])
            else:
                out.append(item[:max_length])
        return out


def _compact_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove keys from the input dictionary where the value is None.
    
    This function helps in cleaning the payload before sending it to the OpenAI API,
    because the API treats keys with None values differently from omitted keys.
    
    Parameters:
        d (Dict[str, Any]): The input dictionary to be compacted.
        
    Returns:
        Dict[str, Any]: A new dictionary with all keys having None values removed.
    """
    return {k: v for k, v in d.items() if v is not None}

class GenAiApiClient:
    def __init__(self, config_path: Optional[str] = None):
        cfg_path = Path(config_path) if config_path else CONFIG_PATH
        with open(cfg_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        azure_cfg = self._config["azure"]
        self.endpoint = azure_cfg["endpoint"]
        self.api_version = azure_cfg["api_version"]
        self.default_deployment = azure_cfg.get("deployment")

        self._openai_client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=azure_cfg["api_key"],
        )

    def get_genai_embeddings(
        self,
        request: Optional[Dict[str, Any]] = None,
        *,
        input: Optional[EmbeddingInput] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        encoding_format: Optional[str] = None,
        user: Optional[str] = None,
        text: Optional[str] = None,
        embeddings_only: bool = False,
        max_length: Optional[int] = None,
        **openai_params: Any,
    ):

        payload: Dict[str, Any] = dict(request or {})
        payload.update(openai_params)

        if payload.get("input") is None:
            if input is not None:
                payload["input"] = input
            elif text is not None:
                payload["input"] = text

        if payload.get("model") is None and model is not None:
            payload["model"] = model

        if payload.get("dimensions") is None and dimensions is not None:
            payload["dimensions"] = dimensions
        if payload.get("encoding_format") is None and encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if payload.get("user") is None and user is not None:
            payload["user"] = user

        if not payload.get("model"):
            raise ValueError("Missing required field: 'model'.")
        if not payload.get("input"):
            raise ValueError("Missing required field: 'input'.")

        inp = payload["input"]
        if isinstance(inp, str) and not inp.strip():
            raise ValueError("Input cannot be an empty string.")
        if isinstance(inp, list) and len(inp) == 0:
            raise ValueError("Input cannot be an empty list.")

        if isinstance(inp, list) and inp and not isinstance(inp[0], int):
            if len(inp) > 2048:
                raise ValueError(
                    "Too many inputs: Azure guidance is max 2048 items per request."
                )

        if max_length is not None:
            payload["input"] = _truncate_input(payload["input"], max_length)

        payload = _compact_none(payload)

        response = self._openai_client.embeddings.create(**payload)

        if not embeddings_only:
            return response

        multi = _is_multi_input(payload["input"])
        if not multi:
            return response.data[0].embedding
        return [item.embedding for item in response.data]

    def get_genai_completion(
        self,
        request: Optional[Dict[str, Any]] = None,
        *,
        prompt: Optional[str] = None,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **openai_params: Any,
    ):

        payload: Dict[str, Any] = dict(request or {})
        payload.update(openai_params)

        if "stream" in payload:
            raise ValueError(
                "Streaming is not supported. Please remove the 'stream' parameter."
            )
        if "stream_options" in payload:
            raise ValueError(
                "Streaming is not supported. Please remove the 'stream_options' parameter."
            )

        if payload.get("messages") is None:
            if messages is not None:
                payload["messages"] = messages
            elif prompt is not None:
                built_messages: List[Dict[str, Any]] = []
                if system_message and system_message != "NONE":
                    built_messages.append({"role": "system", "content": system_message})
                built_messages.append({"role": "user", "content": prompt})
                payload["messages"] = built_messages
        
        if payload.get("model") is None and model is not None:
            payload["model"] = model

        if not payload.get("model"):
            raise ValueError("Missing required field: 'model'.")
        if not payload.get("messages"):
            raise ValueError("Missing required field: 'messages'.")

        payload = _compact_none(payload)

        response = self._openai_client.chat.completions.create(**payload)

        return response