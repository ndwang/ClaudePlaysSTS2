"""Main loop: wait for decision points, render state, get LLM decision, act."""

from __future__ import annotations
import time
import json
from api import STS2API
from state import GameState
from renderer import render
from llm import Agent, RandomAgent, LLMAgent

# ANSI color codes
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_BLUE = "\033[44m"

CONTEXT_COLORS = {
    "combat":           C.RED,
    "map":              C.BLUE,
    "event":            C.MAGENTA,
    "rest":             C.GREEN,
    "shop":             C.YELLOW,
    "treasure":         C.YELLOW,
    "rewards":          C.YELLOW,
    "card_selection":   C.CYAN,
    "hand_select":      C.CYAN,
    "game_over":        C.RED,
    "character_select":  C.BLUE,
    "main_menu":        C.WHITE,
}


SPEED_DELAYS = {
    "fast": 0,
    "normal": 1.0,
    "slow": 3.0,
}


def run(base_url: str = "http://localhost:57541", agent_type: str = "random", model: str = "claude-sonnet-4-20250514", delay: float = 0):
    api = STS2API(base_url)
    gs = GameState()
    agent: Agent = RandomAgent() if agent_type == "random" else LLMAgent(model)

    game_over_seen = False
    round_num = 0
    print(f"{C.CYAN}{C.BOLD}STS2 client starting. Waiting for game...{C.RESET}")

    while True:
        # Wait for a decision point
        try:
            raw = api.wait_for_state(timeout_ms=30000)
        except Exception as e:
            print(f"{C.YELLOW}WARNING: Connection error: {e}. Retrying in 3s...{C.RESET}")
            time.sleep(3)
            continue

        if raw.get("timeout"):
            continue

        # Update internal state
        gs.update(raw)

        # Artificial delay before processing
        if delay > 0:
            time.sleep(delay)

        # Round header
        round_num += 1
        ctx_label = gs.overlay_type or gs.context
        color = CONTEXT_COLORS.get(ctx_label, C.WHITE)
        header = f" Round {round_num} | {ctx_label} "
        separator = "=" * 60
        print(f"\n{color}{C.BOLD}{separator}")
        print(f"{header:=^60}")
        print(f"{separator}{C.RESET}")

        # Log events
        if gs.events_log:
            for msg in gs.events_log:
                print(f"  {C.DIM}>>{C.RESET} {C.YELLOW}{msg}{C.RESET}")
            print()

        # Render briefing
        briefing = render(gs)
        print(briefing)

        # Reset conversation on new run
        if gs.context in ("main_menu", "character_select"):
            agent.reset()
            round_num = 0

        # Pause at game over summary (second screen) for human review
        if gs.context == "game_over":
            if game_over_seen:
                input("Press Enter to continue...")
            game_over_seen = True
        else:
            game_over_seen = False

        # No commands available — server returned transient state
        if not gs.commands:
            print(f"{C.YELLOW}WARNING: No commands available. Retrying in 3s...{C.RESET}")
            time.sleep(3)
            continue

        # Get agent decision
        try:
            decision = agent.decide(gs, briefing)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"{C.RED}ERROR: Agent returned invalid response: {e}. Retrying...{C.RESET}")
            continue

        action_idx = decision["action"]
        cmd = gs.commands[action_idx].copy()
        if "target" in decision:
            cmd["targetIndex"] = decision["target"]

        cmd_type = cmd.get("type", "?")
        detail_parts = [f"{k}={v}" for k, v in cmd.items() if k != "type"]
        detail = " ".join(detail_parts)
        print(f"\n  {C.GREEN}{C.BOLD}-->{C.RESET} Action: {C.GREEN}{cmd_type}{C.RESET} {C.DIM}{detail}{C.RESET}")
        print(f"{C.DIM}{'-' * 60}{C.RESET}")

        # Execute
        try:
            result = api.send_action(cmd)
        except Exception as e:
            print(f"{C.RED}ERROR: Action failed: {e}{C.RESET}")
            continue

        if "error" in result:
            print(f"{C.RED}ERROR: Server error: {result['error']}{C.RESET}")
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
