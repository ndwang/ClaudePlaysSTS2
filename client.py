"""Main loop: wait for decision points, render state, get LLM decision, act."""

from __future__ import annotations
import os
import re
import sys
import time
import json
from api import STS2API
from state import GameState
from renderer import render
from llm import Agent, RandomAgent, LLMAgent
from obs import OBSOverlay

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class TeeWriter:
    """Duplicates writes to both the original stream and a log file (ANSI-stripped)."""

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file

    def write(self, text):
        self._original.write(text)
        self._log.write(ANSI_RE.sub("", text))
        self._log.flush()

    def flush(self):
        self._original.flush()
        self._log.flush()

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


def run(base_url: str = "http://localhost:57541", agent_type: str = "random", model: str = "claude-sonnet-4-20250514", delay: float = 0,
        obs_host: str = "localhost", obs_port: int = 4455, obs_password: str = "", obs_reset: bool = False, confirm: bool = False, log: str = ""):
    if log:
        log_file = open(log, "a", encoding="utf-8")
        sys.stdout = TeeWriter(sys.__stdout__, log_file)

    api = STS2API(base_url)
    gs = GameState()
    agent: Agent = RandomAgent() if agent_type == "random" else LLMAgent(model)
    obs = OBSOverlay(obs_host=obs_host, obs_port=obs_port, obs_password=obs_password)
    if obs_reset:
        obs.reset()

    game_over_seen = False
    round_num = 0
    print(f"{C.CYAN}{C.BOLD}STS2 client starting. Waiting for game...{C.RESET}")

    try:
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

            # Round header
            ctx_label = gs.overlay_type or gs.context
            if gs.context not in ("main_menu", "character_select"):
                round_num += 1
                obs.on_round()
            color = CONTEXT_COLORS.get(ctx_label, C.WHITE)
            header = f" Round {round_num} | {ctx_label} " if round_num > 0 else f" {ctx_label} "
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
                if gs.game_over and not game_over_seen:
                    obs.on_game_over(gs.game_over.score)
                if game_over_seen and confirm:
                    input("Press Enter to continue...")
                game_over_seen = True
            else:
                game_over_seen = False

            # No commands available — server returned transient state
            if not gs.commands:
                print(f"{C.YELLOW}WARNING: No commands available. Retrying in 3s...{C.RESET}")
                time.sleep(3)
                continue

            # Artificial delay before acting (state already printed above)
            if delay > 0:
                time.sleep(delay)

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
    finally:
        obs.on_exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STS2 Agent Client")
    parser.add_argument("--url", default="http://localhost:57541", help="Server base URL")
    parser.add_argument("--agent", default="random", choices=["random", "llm"], help="Agent type")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model ID (for llm agent)")
    parser.add_argument("--speed", default="normal", choices=SPEED_DELAYS.keys(), help="Decision speed (fast/normal/slow)")
    parser.add_argument("--obs-host", default="localhost", help="OBS WebSocket host")
    parser.add_argument("--obs-port", type=int, default=4455, help="OBS WebSocket port")
    parser.add_argument("--obs-password", default=os.environ.get("OBS_WEBSOCKET_PASSWORD", ""), help="OBS WebSocket password (or set OBS_WEBSOCKET_PASSWORD env var)")
    parser.add_argument("--obs-reset", action="store_true", help="Reset OBS overlay (timer, rounds, high score) before starting")
    parser.add_argument("--confirm", action="store_true", help="Pause for human confirmation at end of each run")
    parser.add_argument("--log", default="", help="Path to log file (appends clean text, no ANSI codes)")
    args = parser.parse_args()

    run(base_url=args.url, agent_type=args.agent, model=args.model, delay=SPEED_DELAYS[args.speed],
        obs_host=args.obs_host, obs_port=args.obs_port, obs_password=args.obs_password, obs_reset=args.obs_reset,
        confirm=args.confirm, log=args.log)
