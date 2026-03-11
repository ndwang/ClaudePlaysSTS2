"""Main loop: wait for decision points, render state, get LLM decision, act."""

from __future__ import annotations
import os
import re
import signal
import sys
import time
from api import STS2API
from state import GameState
from renderer import render, format_command
from llm import Agent, RandomAgent, LLMAgent
from obs import OBSOverlay
from i18n import t, set_lang

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
    "step": -1,
}


def auto_resolve(gs: GameState) -> dict | None:
    """Return action dict if the state can be auto-resolved without the agent."""
    commands = gs.commands
    if not commands:
        return None

    cmd_types = {cmd.get("type") for cmd in commands}

    # Only proceed/continue available -> just do it
    if cmd_types <= {"proceed", "continue"}:
        for i, cmd in enumerate(commands):
            if cmd.get("type") in ("proceed", "continue"):
                return {"action": i}

    # Main menu -> start or continue run
    if gs.context == "main_menu":
        for i, cmd in enumerate(commands):
            if cmd.get("type") in ("start_run", "continue_run"):
                return {"action": i}

    # Shop not open -> open first
    if gs.shop and not gs.shop.is_open:
        for i, cmd in enumerate(commands):
            if cmd.get("type") == "shop_open":
                return {"action": i}

    # Rewards -> auto-claim gold and relics
    if gs.overlay_type == "rewards" and gs.overlay:
        for i, cmd in enumerate(commands):
            if cmd.get("type") == "select_reward":
                reward_idx = cmd.get("rewardIndex")
                for r in gs.overlay.rewards:
                    if r.index == reward_idx and r.reward_type in ("gold", "relic"):
                        return {"action": i}

    return None


def run(base_url: str = "http://localhost:57541", agent_type: str = "random", model: str = "claude-sonnet-4-20250514", delay: float = 0,
        thinking_budget: int = 0, obs_host: str = "localhost", obs_port: int = 4455, obs_password: str = "", obs_reset: bool = False,
        run_reset: bool = False, knowledge_reset: bool = False,
        confirm: bool = False, log: str = ""):
    log_file = None
    if log:
        log_file = open(log, "a", encoding="utf-8")
        sys.stdout = TeeWriter(sys.__stdout__, log_file)

    api = STS2API(base_url)
    gs = GameState()
    agent: Agent = RandomAgent() if agent_type == "random" else LLMAgent(model, thinking_budget=thinking_budget)

    obs = OBSOverlay(obs_host=obs_host, obs_port=obs_port, obs_password=obs_password)

    # Wire up real-time reasoning streaming for LLM agent
    if isinstance(agent, LLMAgent):
        def _on_reasoning_delta(text):
            print(f"{C.CYAN}{text}{C.RESET}", end="", flush=True)
            obs.on_reasoning_delta(text)
        agent.on_reasoning_delta = _on_reasoning_delta

    # Apply reset flags
    if obs_reset:
        obs.reset()
    if isinstance(agent, LLMAgent):
        if knowledge_reset:
            LLMAgent.clear_cross_run_kb()
            agent.cross_run_kb.clear()
        if run_reset:
            LLMAgent.clear_run_state()

    # Push initial state to overlay
    if isinstance(agent, LLMAgent):
        obs.on_kb_update(agent.in_run_kb, agent.cross_run_kb)
        obs.on_cost_update(agent.token_tracker.cost_usd)

    # Attempt crash recovery for LLM agent
    recovered = not run_reset and isinstance(agent, LLMAgent) and agent.load_run_state()
    if recovered:
        print(f"{C.YELLOW}{C.BOLD}{t('client.recovered')}{C.RESET}")
        obs.on_kb_update(agent.in_run_kb, agent.cross_run_kb)

    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print(f"\n{C.RED}{C.BOLD}{t('client.forced_shutdown')}{C.RESET}")
            sys.exit(1)
        shutdown_requested = True
        print(f"\n{C.YELLOW}{C.BOLD}{t('client.shutdown')}{C.RESET}")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    game_over_seen = False
    round_num = 0
    round_counted = False  # track to avoid double-counting on retries
    print(f"{C.CYAN}{C.BOLD}{t('client.starting')}{C.RESET}")

    try:
        while not shutdown_requested:
            # Wait for a decision point
            try:
                raw = api.wait_for_state(timeout_ms=30000)
            except Exception as e:
                print(f"{C.YELLOW}{t('client.conn_error', error=e)}{C.RESET}")
                time.sleep(3)
                continue

            if raw.get("timeout"):
                continue

            # Update internal state
            gs.update(raw)

            # No commands available -- server returned transient state
            if not gs.commands:
                print(f"{C.YELLOW}{t('client.no_commands')}{C.RESET}")
                time.sleep(3)
                continue

            # Game over: display, reflect, and track before auto-resolve
            if gs.context == "game_over":
                if gs.game_over and not game_over_seen:
                    obs.on_game_over(gs.game_over.score)
                    print(f"\n{C.RED}{C.BOLD}{'=' * 60}")
                    print(f"{t('client.gameover_banner'):=^60}")
                    print(f"{'=' * 60}{C.RESET}")
                    briefing = render(gs)
                    print(briefing)

                    # Post-run reflection
                    print(f"\n{C.CYAN}{C.BOLD}{t('client.reflecting')}{C.RESET}")
                    obs.on_reasoning_clear()
                    try:
                        agent.reflect(briefing)
                        if agent.last_reasoning:
                            print()  # newline after streamed reasoning
                        if isinstance(agent, LLMAgent):
                            obs.on_kb_update(agent.in_run_kb, agent.cross_run_kb)
                    except Exception as e:
                        print(f"{C.YELLOW}{t('client.reflect_failed', error=e)}{C.RESET}")

                if game_over_seen and confirm:
                    input(t("client.press_enter"))
                game_over_seen = True
            else:
                game_over_seen = False

            # Auto-resolve trivial decisions
            auto = auto_resolve(gs)
            if auto is not None:
                cmd = gs.commands[auto["action"]].copy()
                cmd_type = cmd.get("type", "?")
                print(f"  {C.DIM}{t('client.auto', cmd_type=cmd_type)}{C.RESET}")
                try:
                    api.send_action(cmd)
                except Exception as e:
                    print(f"{C.RED}{t('client.auto_failed', error=e)}{C.RESET}")
                continue

            # --- Agent decision needed ---

            # Reset agent on new run (skip if resuming from crash recovery)
            if gs.context in ("main_menu", "character_select"):
                if recovered:
                    recovered = False
                else:
                    agent.reset()
                    round_num = 0

            # Round header
            ctx_label = gs.overlay_type or gs.context
            if gs.context not in ("main_menu", "character_select") and not round_counted:
                round_num += 1
                obs.on_round()
                round_counted = True
            color = CONTEXT_COLORS.get(ctx_label, C.WHITE)
            header = t("client.round_header", round=round_num, ctx=ctx_label) if round_num > 0 else f" {ctx_label} "
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

            # Artificial delay or manual step before deciding
            if delay > 0:
                time.sleep(delay)
            elif delay < 0:
                input(f"  {C.DIM}{t('client.press_execute')}{C.RESET}")

            # Get agent decision
            obs.on_reasoning_clear()
            cost_before = agent.token_tracker.cost_usd if isinstance(agent, LLMAgent) else 0
            decision = agent.decide(gs, briefing)

            action_idx = decision["action"]
            cmd = gs.commands[action_idx].copy()
            if "target" in decision:
                cmd["targetIndex"] = decision["target"]

            enemies = gs.combat.enemies if gs.combat else []
            formatted = format_command(cmd, enemies=enemies)
            # Reasoning was already streamed to console and OBS
            if agent.last_reasoning:
                print()  # newline after streamed reasoning

            # Push KB state to overlay
            if isinstance(agent, LLMAgent):
                obs.on_kb_update(agent.in_run_kb, agent.cross_run_kb)

            obs.on_reasoning_action(f"--> {formatted}")

            print(f"\n  {C.GREEN}{C.BOLD}-->{C.RESET} {formatted}")
            if isinstance(agent, LLMAgent):
                tt = agent.token_tracker
                turn_cost = tt.cost_usd - cost_before
                print(f"  {C.DIM}${turn_cost:.4f} this turn | ${tt.cost_usd:.2f} lifetime{C.RESET}")
                obs.on_cost_update(tt.cost_usd)
            print(f"{C.DIM}{'-' * 60}{C.RESET}")

            # Execute
            try:
                result = api.send_action(cmd)
            except Exception as e:
                print(f"{C.RED}{t('client.action_failed', error=e)}{C.RESET}")
                continue

            if "error" in result:
                print(f"{C.RED}{t('client.server_error', error=result['error'])}{C.RESET}")
                continue

            # Update state from action response
            round_counted = False
            if "state" in result:
                gs.update(result["state"])
    finally:
        obs.on_exit()
        if log_file:
            sys.stdout = sys.__stdout__
            log_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STS2 Agent Client")
    parser.add_argument("--url", default="http://localhost:57541", help="Server base URL")
    parser.add_argument("--agent", default="random", choices=["random", "llm"], help="Agent type")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model ID (for llm agent)")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Extended thinking token budget (0=disabled, e.g. 4000)")
    parser.add_argument("--speed", default="normal", choices=SPEED_DELAYS.keys(), help="Decision speed (fast/normal/slow)")
    parser.add_argument("--lang", default="en", choices=["en", "zh"], help="Display language (en/zh)")
    parser.add_argument("--obs-host", default="localhost", help="OBS WebSocket host")
    parser.add_argument("--obs-port", type=int, default=4455, help="OBS WebSocket port")
    parser.add_argument("--obs-password", default=os.environ.get("OBS_WEBSOCKET_PASSWORD", ""), help="OBS WebSocket password (or set OBS_WEBSOCKET_PASSWORD env var)")
    parser.add_argument("--obs-reset", action="store_true", help="Reset OBS overlay (timer, rounds, high score)")
    parser.add_argument("--run-reset", action="store_true", help="Clear agent run state (conversation + in-run KB)")
    parser.add_argument("--knowledge-reset", action="store_true", help="Clear cross-run knowledge base")
    parser.add_argument("--reset", action="store_true", help="Reset all state (OBS + run + knowledge)")
    parser.add_argument("--confirm", action="store_true", help="Pause for human confirmation at end of each run")
    parser.add_argument("--log", default="", help="Path to log file (appends clean text, no ANSI codes)")
    args = parser.parse_args()

    if args.reset:
        args.obs_reset = args.run_reset = args.knowledge_reset = True

    set_lang(args.lang)

    run(base_url=args.url, agent_type=args.agent, model=args.model, delay=SPEED_DELAYS[args.speed],
        thinking_budget=args.thinking_budget,
        obs_host=args.obs_host, obs_port=args.obs_port, obs_password=args.obs_password, obs_reset=args.obs_reset,
        run_reset=args.run_reset, knowledge_reset=args.knowledge_reset,
        confirm=args.confirm, log=args.log)
