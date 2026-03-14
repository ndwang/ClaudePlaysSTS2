"""LLM backend abstraction for multi-provider support."""

from __future__ import annotations
import dataclasses
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ThinkingBlock:
    type: str = "thinking"
    thinking: str = ""
    signature: str = ""

    def model_dump(self):
        return dataclasses.asdict(self)


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""

    def model_dump(self):
        return dataclasses.asdict(self)


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)

    def model_dump(self):
        return dataclasses.asdict(self)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class MessageResponse:
    content: list  # list of ThinkingBlock, TextBlock, ToolUseBlock
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""


class LLMBackend(ABC):
    """Abstract LLM backend for API calls."""

    @abstractmethod
    def call(self, *, model: str, system: str, messages: list[dict],
             max_tokens: int, tools: list[dict] | None = None,
             tool_choice: dict | None = None, thinking: dict | None = None,
             on_reasoning_delta: Callable[[str], None] | None = None,
             stream_reasoning: bool = False,
             cache: bool = True) -> MessageResponse:
        ...

    def get_pricing(self, model: str) -> dict[str, float] | None:
        return None


class AnthropicBackend(LLMBackend):
    PRICING = {
        "claude-sonnet-4": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
        "claude-opus-4":   {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
        "claude-haiku-3":  {"input": 0.80, "output": 4.0, "cache_write": 1.0, "cache_read": 0.08},
    }

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self._anthropic = anthropic

    def call(self, *, model, system, messages, max_tokens, tools=None,
             tool_choice=None, thinking=None,
             on_reasoning_delta=None, stream_reasoning=False,
             cache=True) -> MessageResponse:
        if cache:
            system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if tools:
                tools = [t for t in tools]  # shallow copy
                tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
            messages = self._add_message_cache_breakpoint(messages)
        else:
            system_param = [{"type": "text", "text": system}]

        params = {
            "model": model,
            "system": system_param,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice
        if thinking:
            params["thinking"] = thinking

        log = logging.getLogger(__name__)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                with self.client.messages.stream(**params) as stream:
                    if stream_reasoning and on_reasoning_delta:
                        for event in stream:
                            if event.type == "content_block_delta":
                                if event.delta.type == "thinking_delta":
                                    on_reasoning_delta(event.delta.thinking)
                                elif event.delta.type == "text_delta":
                                    on_reasoning_delta(event.delta.text)
                    message = stream.get_final_message()
                    return self._convert_response(message)
            except (self._anthropic.APITimeoutError, self._anthropic.APIConnectionError,
                    self._anthropic.InternalServerError, self._anthropic.RateLimitError) as e:
                if attempt == max_attempts:
                    raise
                wait = 2 ** attempt
                log.warning("API call failed (attempt %d/%d): %s. Retrying in %ds...",
                            attempt, max_attempts, e, wait)
                time.sleep(wait)

    @staticmethod
    def _add_message_cache_breakpoint(messages: list[dict]) -> list[dict]:
        """Add cache_control to the last content block of the last user message.

        This ensures the entire conversation prefix (all prior turns) gets cached
        for the next API call, so only the new turn needs to be prefilled.
        """
        if not messages:
            return messages

        # Find the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return messages

        # Shallow copy messages so we don't mutate the caller's list
        messages = list(messages)
        msg = messages[last_user_idx]
        content = msg.get("content")

        if isinstance(content, str):
            messages[last_user_idx] = {
                "role": "user",
                "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
            }
        elif isinstance(content, list) and content:
            content = list(content)  # shallow copy
            last_block = content[-1]
            # Handle both dict blocks and SDK objects
            if hasattr(last_block, "model_dump"):
                last_block = last_block.model_dump()
            elif isinstance(last_block, dict):
                last_block = {**last_block}
            last_block["cache_control"] = {"type": "ephemeral"}
            content[-1] = last_block
            messages[last_user_idx] = {"role": "user", "content": content}

        return messages

    def _convert_response(self, message) -> MessageResponse:
        content = []
        for block in message.content:
            if block.type == "thinking":
                content.append(ThinkingBlock(thinking=block.thinking, signature=getattr(block, "signature", "")))
            elif block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
        usage = Usage(
            input_tokens=getattr(message.usage, "input_tokens", 0),
            output_tokens=getattr(message.usage, "output_tokens", 0),
            cache_creation_input_tokens=getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        )
        return MessageResponse(content=content, usage=usage, stop_reason=message.stop_reason)

    def get_pricing(self, model: str) -> dict[str, float] | None:
        for prefix in sorted(self.PRICING, key=len, reverse=True):
            if model.startswith(prefix):
                return self.PRICING[prefix]
        return None


class OpenAIBackend(LLMBackend):
    PRICING = {
        # GPT-5 family
        "gpt-5.4": {"input": 2.50, "output": 15.0, "cache_write": 2.50, "cache_read": 1.25},
        "gpt-5.2": {"input": 1.75, "output": 14.0, "cache_write": 1.75, "cache_read": 0.175},
        "gpt-5.1": {"input": 1.25, "output": 10.0, "cache_write": 1.25, "cache_read": 0.125},
        "gpt-5-mini": {"input": 0.25, "output": 2.0, "cache_write": 0.25, "cache_read": 0.025},
        "gpt-5": {"input": 1.25, "output": 10.0, "cache_write": 1.25, "cache_read": 0.125},
        # GPT-4 family
        "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_write": 2.0, "cache_read": 0.50},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_write": 0.40, "cache_read": 0.10},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cache_write": 0.10, "cache_read": 0.025},
        "gpt-4o": {"input": 2.5, "output": 10.0, "cache_write": 2.5, "cache_read": 1.25},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_write": 0.15, "cache_read": 0.075},
        # Reasoning models
        "o3": {"input": 2.0, "output": 8.0, "cache_write": 2.0, "cache_read": 1.0},
        "o4-mini": {"input": 1.10, "output": 4.40, "cache_write": 1.10, "cache_read": 0.275},
    }

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        import openai
        kwargs: dict = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = openai.OpenAI(**kwargs)
        self._openai = openai

    def call(self, *, model, system, messages, max_tokens, tools=None,
             tool_choice=None, thinking=None,
             on_reasoning_delta=None, stream_reasoning=False,
             cache=True) -> MessageResponse:
        oai_messages = self._convert_messages(system, messages)

        params = {
            "model": model,
            "messages": oai_messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            oai_tools = self._convert_tools(tools)
            params["tools"] = oai_tools

            # Map tool_choice
            if tool_choice:
                tc_type = tool_choice.get("type")
                if tc_type == "any":
                    params["tool_choice"] = "required"
                elif tc_type == "auto":
                    params["tool_choice"] = "auto"

        # For reasoning models (o-series and GPT-5 family), map thinking budget to reasoning effort
        is_reasoning_model = any(model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5"))
        if thinking and is_reasoning_model:
            budget = thinking.get("budget_tokens", 0)
            if budget > 8000:
                params["reasoning"] = {"effort": "high"}
            elif budget > 2000:
                params["reasoning"] = {"effort": "medium"}
            else:
                params["reasoning"] = {"effort": "low"}

        log = logging.getLogger(__name__)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                return self._stream_call(
                    params,
                    on_reasoning_delta if stream_reasoning else None,
                )
            except (self._openai.APITimeoutError, self._openai.APIConnectionError,
                    self._openai.InternalServerError, self._openai.RateLimitError) as e:
                if attempt == max_attempts:
                    raise
                wait = 2 ** attempt
                log.warning("API call failed (attempt %d/%d): %s. Retrying in %ds...",
                            attempt, max_attempts, e, wait)
                time.sleep(wait)

    def _stream_call(self, params, on_reasoning_delta) -> MessageResponse:
        content_text = ""
        reasoning_text = ""
        tool_calls_acc: dict[int, dict] = {}
        usage = Usage()
        stop_reason = ""

        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.usage:
                usage.input_tokens = chunk.usage.prompt_tokens or 0
                usage.output_tokens = chunk.usage.completion_tokens or 0
                # Cache usage if available
                if hasattr(chunk.usage, "prompt_tokens_details") and chunk.usage.prompt_tokens_details:
                    details = chunk.usage.prompt_tokens_details
                    usage.cache_read_input_tokens = getattr(details, "cached_tokens", 0) or 0

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if finish_reason:
                stop_reason = "end_turn" if finish_reason == "stop" else finish_reason

            # Reasoning content (o-series models)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_text += delta.reasoning_content
                if on_reasoning_delta:
                    on_reasoning_delta(delta.reasoning_content)

            # Regular content
            if delta and delta.content:
                content_text += delta.content
                if on_reasoning_delta:
                    on_reasoning_delta(delta.content)

            # Tool calls
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

        # Build normalized content blocks
        blocks: list = []
        if reasoning_text:
            blocks.append(ThinkingBlock(thinking=reasoning_text))
        if content_text:
            blocks.append(TextBlock(text=content_text))
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(
                id=tc["id"] or f"call_{uuid.uuid4().hex[:8]}",
                name=tc["name"],
                input=args,
            ))

        return MessageResponse(content=blocks, usage=usage, stop_reason=stop_reason)

    def _convert_messages(self, system: str, messages: list[dict]) -> list[dict]:
        """Convert Anthropic-format messages to OpenAI format."""
        oai: list[dict] = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")

            if role == "user":
                if isinstance(content, str):
                    oai.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        b = block.model_dump() if hasattr(block, "model_dump") else block
                        if isinstance(b, str):
                            text_parts.append(b)
                        elif b.get("type") == "tool_result":
                            tool_content = b.get("content", "")
                            oai.append({
                                "role": "tool",
                                "tool_call_id": b["tool_use_id"],
                                "content": tool_content if isinstance(tool_content, str) else json.dumps(tool_content),
                            })
                        elif b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                    if text_parts:
                        oai.append({"role": "user", "content": "\n".join(text_parts)})
                else:
                    oai.append({"role": "user", "content": str(content)})

            elif role == "assistant":
                if isinstance(content, str):
                    oai.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        b = block.model_dump() if hasattr(block, "model_dump") else block
                        if isinstance(b, str):
                            text_parts.append(b)
                        elif b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                        elif b.get("type") == "tool_use":
                            tool_calls.append({
                                "id": b["id"],
                                "type": "function",
                                "function": {
                                    "name": b["name"],
                                    "arguments": json.dumps(b.get("input", {})),
                                },
                            })
                        # thinking blocks are dropped (not representable in OpenAI)
                    msg_dict: dict = {"role": "assistant"}
                    msg_dict["content"] = "\n".join(text_parts) if text_parts else None
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                    oai.append(msg_dict)

        return oai

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert Anthropic tool format to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in tools
        ]

    def get_pricing(self, model: str) -> dict[str, float] | None:
        for prefix in sorted(self.PRICING, key=len, reverse=True):
            if model.startswith(prefix):
                return self.PRICING[prefix]
        return None


def create_backend(model: str, base_url: str | None = None,
                    api_key: str | None = None) -> LLMBackend:
    """Create the appropriate backend based on model name.

    For non-Claude models, uses the OpenAI-compatible backend.
    Set base_url to point at any OpenAI-compatible endpoint
    (Ollama, vLLM, Together, LM Studio, etc.).
    """
    if model.startswith("claude") and not base_url:
        return AnthropicBackend()
    return OpenAIBackend(base_url=base_url, api_key=api_key)
