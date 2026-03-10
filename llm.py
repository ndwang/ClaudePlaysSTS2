"""Agent interfaces for STS2 decision-making."""

from __future__ import annotations
import json
import random
from abc import ABC, abstractmethod

from state import GameState


class Agent(ABC):
    @abstractmethod
    def decide(self, gs: GameState, briefing: str) -> dict:
        """Return {"action": index, "target"?: enemy_index}."""
        ...

    def reset(self) -> None:
        """Called at the start of a new run."""
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
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        from prompts import SYSTEM_PROMPT
        self.system_prompt = SYSTEM_PROMPT
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages: list[dict] = []

    def decide(self, gs: GameState, briefing: str) -> dict:
        self.messages.append({"role": "user", "content": briefing})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=self.system_prompt,
            messages=self.messages,
        )

        text = response.content[0].text.strip()
        self.messages.append({"role": "assistant", "content": text})

        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        decision = json.loads(text)

        action = decision["action"]
        num_commands = len(gs.commands)
        if not isinstance(action, int) or action < 0 or action >= num_commands:
            raise ValueError(f"Invalid command index: {action} (valid: 0-{num_commands - 1})")

        return decision

    def reset(self) -> None:
        self.messages.clear()
