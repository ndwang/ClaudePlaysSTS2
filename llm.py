"""Agent interfaces for STS2 decision-making."""

from __future__ import annotations
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from i18n import t
from state import GameState


KB_FILE = Path(__file__).parent / "knowledge.json"
RUN_STATE_FILE = Path(__file__).parent / "run_state.json"


def _build_tools() -> list[dict]:
    """Build tool definitions with localized descriptions."""
    return [
        {
            "name": "play_action",
            "description": t("tool.play_action.desc"),
            "input_schema": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": t("tool.play_action.index_desc"),
                    },
                    "target": {
                        "type": "integer",
                        "description": t("tool.play_action.target_desc"),
                    },
                },
                "required": ["index"],
            },
        },
        {
            "name": "update_knowledge_base",
            "description": t("tool.update_kb.desc"),
            "input_schema": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "enum": ["in_run", "cross_run"],
                        "description": t("tool.update_kb.store_desc"),
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["set", "delete"],
                        "description": t("tool.update_kb.operation_desc"),
                    },
                    "key": {
                        "type": "string",
                        "description": t("tool.update_kb.key_desc"),
                    },
                    "value": {
                        "type": "string",
                        "description": t("tool.update_kb.value_desc"),
                    },
                },
                "required": ["store", "operation", "key"],
            },
        },
    ]


def _load_cross_run_kb() -> dict[str, str]:
    try:
        return json.loads(KB_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def _save_cross_run_kb(kb: dict[str, str]) -> None:
    KB_FILE.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")


class Agent(ABC):
    last_reasoning: str = ""

    @abstractmethod
    def decide(self, gs: GameState, briefing: str) -> dict:
        """Return {"action": index, "target"?: enemy_index}."""
        ...

    def reset(self) -> None:
        """Called at the start of a new run."""
        pass

    def reflect(self, briefing: str) -> None:
        """Called after game over for post-run reflection."""
        pass


class RandomAgent(Agent):
    """Picks a random command, preferring to play cards / claim rewards before ending turn / proceeding."""

    # Commands that should only be chosen when no other options remain
    TERMINAL_TYPES = {"end_turn", "proceed"}

    def decide(self, gs: GameState, briefing: str) -> dict:
        non_terminal = [
            (i, cmd) for i, cmd in enumerate(gs.commands)
            if cmd.get("type") not in self.TERMINAL_TYPES
        ]
        candidates = non_terminal if non_terminal else list(enumerate(gs.commands))

        idx, cmd = random.choice(candidates)
        decision = {"action": idx}
        if cmd.get("requiresTarget") and gs.combat:
            enemies = gs.combat.enemies
            if enemies:
                decision["target"] = random.choice(enemies).index
        return decision


class LLMAgent(Agent):
    MAX_RETRIES = 5
    SUMMARIZE_THRESHOLD = 60  # message count (~30 exchanges)

    def __init__(self, model: str = "claude-sonnet-4-20250514", thinking_budget: int = 0):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.thinking_budget = thinking_budget
        self.messages: list[dict] = []
        self._pending_tool_use_id: str | None = None
        self._pending_kb_results: list[dict] = []
        self.last_reasoning: str = ""
        self._summary: str = ""
        self.on_reasoning_delta: Callable[[str], None] | None = None

        # Knowledge bases
        self.in_run_kb: dict[str, str] = {}
        self.cross_run_kb: dict[str, str] = _load_cross_run_kb()

    def _api_call(self, stream_reasoning: bool = False, **params):
        """Call the Anthropic API with streaming, retries on transient errors.

        Uses streaming to avoid total-time timeouts (only idle time matters).
        If stream_reasoning is True, emits thinking/text deltas via on_reasoning_delta callback.
        Returns the final accumulated Message object (same shape as non-streaming).
        """
        import anthropic
        log = logging.getLogger(__name__)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                with self.client.messages.stream(**params) as stream:
                    if stream_reasoning and self.on_reasoning_delta:
                        for event in stream:
                            if event.type == "content_block_delta":
                                if event.delta.type == "thinking_delta":
                                    self.on_reasoning_delta(event.delta.thinking)
                                elif event.delta.type == "text_delta":
                                    self.on_reasoning_delta(event.delta.text)
                    return stream.get_final_message()
            except (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.RateLimitError) as e:
                if attempt == max_attempts:
                    raise
                wait = 2 ** attempt
                log.warning("API call failed (attempt %d/%d): %s. Retrying in %ds...", attempt, max_attempts, e, wait)
                time.sleep(wait)

    def _build_system_prompt(self) -> str:
        from prompts import system_prompt
        parts = [system_prompt()]

        if self.cross_run_kb:
            lines = [f"- {k}: {v}" for k, v in self.cross_run_kb.items()]
            parts.append(t("prompt.cross_run_kb_header") + "\n".join(lines))

        if self.in_run_kb:
            lines = [f"- {k}: {v}" for k, v in self.in_run_kb.items()]
            parts.append(t("prompt.in_run_kb_header") + "\n".join(lines))

        return "".join(parts)

    def _build_api_params(self) -> dict:
        """Build API call parameters, varying by whether thinking is enabled."""
        params = {
            "model": self.model,
            "system": self._build_system_prompt(),
            "tools": _build_tools(),
            "messages": self.messages,
        }
        if self.thinking_budget > 0:
            # Extended thinking: must use tool_choice=auto, model may not call tools
            params["max_tokens"] = self.thinking_budget + 4000
            params["tool_choice"] = {"type": "auto"}
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        else:
            # No thinking: can force tool use
            params["max_tokens"] = 1024
            params["tool_choice"] = {"type": "any"}
        return params

    def _extract_reasoning(self, content: list) -> str:
        """Extract reasoning from response: prefer thinking blocks, fall back to text."""
        thinking_parts = []
        text_parts = []
        for block in content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "text" and block.text.strip():
                text_parts.append(block.text.strip())
        return "\n".join(thinking_parts) if thinking_parts else "\n".join(text_parts)

    def _summarize(self) -> None:
        """Compress conversation history by asking the same session to summarize itself."""
        from prompts import summarization_prompt

        if not self.messages:
            return

        # If there's a pending tool_use, provide a dummy result before the summarization request
        if self._pending_tool_use_id:
            tool_results = list(self._pending_kb_results)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": self._pending_tool_use_id,
                "content": t("llm.summarization_requested"),
            })
            self.messages.append({"role": "user", "content": tool_results + [{"type": "text", "text": summarization_prompt()}]})
        else:
            # Ask the model to summarize — it can see its own full history including thinking
            self.messages.append({"role": "user", "content": summarization_prompt()})

        params = {
            "model": self.model,
            "system": self._build_system_prompt(),
            "messages": self.messages,
            "max_tokens": 1024,
        }
        if self.thinking_budget > 0:
            params["max_tokens"] = self.thinking_budget + 1024
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        response = self._api_call(**params)

        # Extract text (skip thinking blocks)
        self._summary = next(
            b.text for b in response.content if b.type == "text"
        )
        self.messages.clear()
        self._pending_tool_use_id = None
        self._pending_kb_results = []

    @staticmethod
    def _serialize_messages(messages: list[dict]) -> list[dict]:
        """Convert messages to JSON-serializable dicts (handles Pydantic content blocks)."""
        out = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                serialized = []
                for block in content:
                    if hasattr(block, "model_dump"):
                        serialized.append(block.model_dump())
                    else:
                        serialized.append(block)
                out.append({"role": msg["role"], "content": serialized})
            else:
                out.append(msg)
        return out

    def save_run_state(self) -> None:
        """Persist conversation state to disk for crash recovery."""
        state = {
            "messages": self._serialize_messages(self.messages),
            "pending_tool_use_id": self._pending_tool_use_id,
            "pending_kb_results": self._pending_kb_results,
            "in_run_kb": self.in_run_kb,
            "last_reasoning": self.last_reasoning,
            "summary": self._summary,
        }
        RUN_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    def load_run_state(self) -> bool:
        """Load conversation state from disk. Returns True if state was loaded."""
        try:
            state = json.loads(RUN_STATE_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return False
        self.messages = state.get("messages", [])
        self._pending_tool_use_id = state.get("pending_tool_use_id")
        self._pending_kb_results = state.get("pending_kb_results", [])
        self.in_run_kb = state.get("in_run_kb", {})
        self.last_reasoning = state.get("last_reasoning", "")
        self._summary = state.get("summary", "")
        return True

    @staticmethod
    def clear_run_state() -> None:
        """Remove saved run state file."""
        RUN_STATE_FILE.unlink(missing_ok=True)

    def _process_kb_update(self, inp: dict) -> str:
        """Process an update_knowledge_base tool call, return result message."""
        store = inp.get("store", "in_run")
        operation = inp.get("operation", "set")
        key = inp.get("key", "")
        value = inp.get("value", "")

        if not key:
            return t("llm.kb_error_no_key")

        kb = self.in_run_kb if store == "in_run" else self.cross_run_kb

        if operation == "set":
            if not value:
                return t("llm.kb_error_no_value")
            kb[key] = value
            if store == "cross_run":
                _save_cross_run_kb(self.cross_run_kb)
            return t("llm.kb_ok_set", store=store, key=key, value=value)
        elif operation == "delete":
            if key in kb:
                del kb[key]
                if store == "cross_run":
                    _save_cross_run_kb(self.cross_run_kb)
                return t("llm.kb_ok_delete", store=store, key=key)
            return t("llm.kb_warn_not_found", key=key, store=store)
        else:
            return t("llm.kb_error_unknown_op", operation=operation)

    def _process_response(self, response, num_commands: int | None) -> dict | None:
        """Process a response, handle KB calls, return action dict if play_action found.

        Returns {"action": idx, ...} on valid play_action, None otherwise.
        Appends any needed tool_result messages to self.messages for retry.
        """
        content = response.content
        self.messages.append({"role": "assistant", "content": content})
        self.last_reasoning = self._extract_reasoning(content)

        # Collect all tool calls
        tool_calls = [b for b in content if b.type == "tool_use"]

        if not tool_calls:
            # No tool call (can happen with thinking mode's tool_choice: auto)
            self.messages.append({
                "role": "user",
                "content": t("llm.must_use_tool"),
            })
            return None

        # Process all tool calls, build tool_results
        results = []
        action_result = None  # Will be set if valid play_action found
        play_action_block = None

        for block in tool_calls:
            if block.name == "update_knowledge_base":
                msg = self._process_kb_update(block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": msg,
                })
            elif block.name == "play_action" and num_commands is not None:
                index = block.input.get("index")
                if isinstance(index, int) and 0 <= index < num_commands:
                    play_action_block = block
                    action_result = {"action": index}
                    if "target" in block.input:
                        action_result["target"] = block.input["target"]
                    # Don't add tool_result yet — it will be the next game state
                else:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": t("llm.invalid_index", index=index, max_index=num_commands - 1),
                        "is_error": True,
                    })

        if action_result and play_action_block:
            # Valid action found. Store KB results to prepend to next state delivery.
            self._pending_tool_use_id = play_action_block.id
            self._pending_kb_results = results
            return action_result

        # No valid play_action — send all results and retry
        if results:
            self.messages.append({"role": "user", "content": results})
        else:
            self.messages.append({
                "role": "user",
                "content": t("llm.must_use_tool"),
            })
        return None

    def decide(self, gs: GameState, briefing: str) -> dict:
        # Guard: if previous decide() failed mid-way, messages may end with a
        # user message.  Pop it so we don't create consecutive user messages.
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()

        # Summarize if history is getting long
        if len(self.messages) >= self.SUMMARIZE_THRESHOLD:
            self._summarize()

        # Build user message with tool_result(s)
        # After summarization, prepend summary to the first user message
        if self._summary:
            briefing = t("prompt.history_summary", summary=self._summary) + briefing
            self._summary = ""

        if self._pending_tool_use_id is None:
            self.messages.append({"role": "user", "content": briefing})
        else:
            tool_results = list(self._pending_kb_results)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": self._pending_tool_use_id,
                "content": briefing,
            })
            self.messages.append({"role": "user", "content": tool_results})
            self._pending_tool_use_id = None
            self._pending_kb_results = []

        num_commands = len(gs.commands)

        for _ in range(self.MAX_RETRIES):
            response = self._api_call(stream_reasoning=True, **self._build_api_params())
            result = self._process_response(response, num_commands)
            if result is not None:
                self.save_run_state()
                return result

        raise RuntimeError(f"Agent failed to provide valid action after {self.MAX_RETRIES} retries")

    def reflect(self, briefing: str) -> None:
        """Post-run reflection: agent reviews the run and updates cross-run KB."""
        from prompts import reflection_prompt

        # Send game over state + reflection prompt
        if self._pending_tool_use_id is None:
            self.messages.append({"role": "user", "content": f"{briefing}\n\n{reflection_prompt()}"})
        else:
            tool_results = list(self._pending_kb_results)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": self._pending_tool_use_id,
                "content": f"{briefing}\n\n{reflection_prompt()}",
            })
            self.messages.append({"role": "user", "content": tool_results})
            self._pending_tool_use_id = None
            self._pending_kb_results = []

        # Use auto tool_choice so agent can call KB tools or just respond with text
        params = {
            "model": self.model,
            "system": self._build_system_prompt(),
            "tools": _build_tools(),
            "messages": self.messages,
            "max_tokens": 2048,
            "tool_choice": {"type": "auto"},
        }
        if self.thinking_budget > 0:
            params["max_tokens"] = self.thinking_budget + 2048
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        for _ in range(3):
            response = self._api_call(stream_reasoning=True, **params)
            content = response.content
            self.messages.append({"role": "assistant", "content": content})
            self.last_reasoning = self._extract_reasoning(content)

            # Process any KB updates
            tool_calls = [b for b in content if b.type == "tool_use"]
            if not tool_calls:
                break  # Agent responded with text only — reflection done

            results = []
            for block in tool_calls:
                if block.name == "update_knowledge_base":
                    msg = self._process_kb_update(block.input)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": msg,
                    })
                elif block.name == "play_action":
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": t("llm.no_action_reflection"),
                        "is_error": True,
                    })

            if results:
                self.messages.append({"role": "user", "content": results})

            # If stop_reason is end_turn or no more tool calls expected, break
            if response.stop_reason == "end_turn":
                break

        self.save_run_state()

    def reset(self) -> None:
        self.messages.clear()
        self._pending_tool_use_id = None
        self._pending_kb_results = []
        self.last_reasoning = ""
        self._summary = ""
        self.in_run_kb.clear()
        self.clear_run_state()
        # cross_run_kb persists — reload in case it was updated externally
        self.cross_run_kb = _load_cross_run_kb()
