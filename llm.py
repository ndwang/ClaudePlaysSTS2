"""LLM decision-making interface for STS2 agent."""

from __future__ import annotations
import json
import anthropic
from prompts import SYSTEM_PROMPT


class LLMAgent:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages: list[dict] = []

    def decide(self, briefing: str, num_commands: int) -> dict:
        """Send the briefing to the LLM and get a structured action back."""
        self.messages.append({"role": "user", "content": briefing})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=self.messages,
        )

        text = response.content[0].text.strip()
        self.messages.append({"role": "assistant", "content": text})

        # Parse the JSON response
        # Handle markdown code blocks if the LLM wraps the JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        decision = json.loads(text)

        # Validate
        action = decision["action"]
        if not isinstance(action, int) or action < 0 or action >= num_commands:
            raise ValueError(f"Invalid command index: {action} (valid: 0-{num_commands - 1})")

        return decision

    def reset(self) -> None:
        """Clear conversation history (e.g., at start of new run)."""
        self.messages.clear()
