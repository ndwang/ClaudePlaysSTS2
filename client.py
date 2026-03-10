"""Main loop: wait for decision points, render state, get LLM decision, act."""

from __future__ import annotations
import sys
import time
import json
import logging

from api import STS2API
from state import GameState
from renderer import render
from llm import Agent, RandomAgent, LLMAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


SPEED_DELAYS = {
    "fast": 0,
    "normal": 1.0,
    "slow": 3.0,
}


def run(base_url: str = "http://localhost:57541", agent_type: str = "random", model: str = "claude-sonnet-4-20250514", delay: float = 0):
    api = STS2API(base_url)
    gs = GameState()
    agent: Agent = RandomAgent() if agent_type == "random" else LLMAgent(model)

    log.info("STS2 client starting. Waiting for game...")

    while True:
        # Wait for a decision point
        try:
            raw = api.wait_for_state(timeout_ms=30000)
        except Exception as e:
            log.warning(f"Connection error: {e}. Retrying in 3s...")
            time.sleep(3)
            continue

        if raw.get("timeout"):
            continue

        # Update internal state
        gs.update(raw)

        # Log events
        for msg in gs.events_log:
            log.info(f"Event: {msg}")

        # Render briefing
        briefing = render(gs)
        log.info(f"--- {gs.overlay_type or gs.context} ---")
        log.info(f"\n{briefing}")

        # Reset conversation on new run
        if gs.context in ("main_menu", "character_select"):
            agent.reset()

        # Pause at game over for human review
        if gs.context == "game_over":
            input("Press Enter to continue...")

        # Get agent decision
        try:
            decision = agent.decide(gs, briefing)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.error(f"Agent returned invalid response: {e}. Retrying...")
            continue

        action_idx = decision["action"]
        cmd = gs.commands[action_idx].copy()
        if "target" in decision:
            cmd["targetIndex"] = decision["target"]

        log.info(f"Action: {cmd}")

        if delay > 0:
            time.sleep(delay)

        # Execute
        try:
            result = api.send_action(cmd)
        except Exception as e:
            log.error(f"Action failed: {e}")
            continue

        if "error" in result:
            log.error(f"Server error: {result['error']}")
            continue

        # Update state from action response
        if "state" in result:
            gs.update(result["state"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STS2 Agent Client")
    parser.add_argument("--url", default="http://localhost:57541", help="Server base URL")
    parser.add_argument("--agent", default="random", choices=["random", "llm"], help="Agent type")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model ID (for llm agent)")
    parser.add_argument("--speed", default="normal", choices=SPEED_DELAYS.keys(), help="Decision speed (fast/normal/slow)")
    args = parser.parse_args()

    run(base_url=args.url, agent_type=args.agent, model=args.model, delay=SPEED_DELAYS[args.speed])
