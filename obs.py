"""OBS overlay integration via obs-websocket browser source events."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

OBS_DIR = Path(__file__).parent / "obs"
STATE_FILE = OBS_DIR / "state.json"


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


class OBSOverlay:
    """Controls OBS browser sources via websocket events."""

    def __init__(self, obs_host: str = "localhost", obs_port: int = 4455, obs_password: str = "") -> None:
        OBS_DIR.mkdir(exist_ok=True)
        # Write overlay locale strings for HTML browser sources
        from i18n import write_overlay_locale
        write_overlay_locale(OBS_DIR)
        state = _load_state()
        self.high_score: int = int(state.get("high_score", 0))
        self.total_rounds: int = int(state.get("total_rounds", 0))
        self.timer_elapsed: float = state.get("timer_elapsed", 0)  # seconds accumulated before this session
        self.timer_start: float = time.time()  # epoch when this session started
        self._last_kb: dict = state.get("kb", {"in_run": {}, "cross_run": {}})
        self._reasoning_blocks: list[dict] = state.get("reasoning_blocks", [])
        self._current_reasoning: str = ""
        self._ws = None

        try:
            import obsws_python as obs
            self._ws = obs.ReqClient(host=obs_host, port=obs_port, password=obs_password, timeout=3)
            print(f"  OBS WebSocket connected ({obs_host}:{obs_port})")
        except Exception as e:
            print(f"  OBS WebSocket not available ({e}). Overlays won't update.")

        self._emit("high-score-update", {"value": self.high_score})
        self._emit("round-update", {"value": self.total_rounds})
        self._emit("timer-sync", {"elapsed": self.timer_elapsed, "epoch": self.timer_start})

    def _emit(self, event_name: str, event_data: dict) -> None:
        """Send a custom event to all OBS browser sources."""
        if not self._ws:
            return
        try:
            self._ws.call_vendor_request(
                vendor_name="obs-browser",
                request_type="emit_event",
                request_data={
                    "event_name": event_name,
                    "event_data": event_data,
                },
            )
        except Exception as e:
            print(f"  OBS emit '{event_name}' failed: {e}")

    def _save(self) -> None:
        _save_state({
            "high_score": self.high_score,
            "total_rounds": self.total_rounds,
            "timer_elapsed": self.timer_elapsed + (time.time() - self.timer_start),
            "kb": self._last_kb,
            "reasoning_blocks": self._reasoning_blocks,
        })
        self._emit("timer-sync", {"elapsed": self.timer_elapsed, "epoch": self.timer_start})

    def reset(self) -> None:
        """Reset all overlay values."""
        self.high_score = 0
        self.total_rounds = 0
        self.timer_elapsed = 0
        self.timer_start = time.time()
        self._save()
        self._emit("timer-sync", {"elapsed": 0, "epoch": self.timer_start})
        self._emit("round-update", {"value": 0})
        self._emit("high-score-update", {"value": 0})

    def on_round(self) -> None:
        """Call each decision round."""
        self.total_rounds += 1
        self._save()
        self._emit("round-update", {"value": self.total_rounds})

    def on_exit(self) -> None:
        """Call when the agent is shutting down."""
        if self._ws:
            try:
                self._ws.disconnect()
            except Exception:
                pass
            self._ws = None

    def on_reasoning_clear(self) -> None:
        """Clear the reasoning overlay for a new decision."""
        self._current_reasoning = ""
        self._emit("reasoning-clear", {})

    def on_reasoning_delta(self, text: str) -> None:
        """Stream a reasoning text chunk to the OBS overlay."""
        self._current_reasoning += text
        self._emit("reasoning-delta", {"text": text})

    def on_reasoning(self, text: str) -> None:
        """Push full agent reasoning text to the OBS overlay (fallback)."""
        self._current_reasoning = text
        self._emit("reasoning-update", {"text": text})

    def on_reasoning_action(self, text: str) -> None:
        """Show the chosen action in the reasoning overlay and close the block."""
        self._reasoning_blocks.append({"text": self._current_reasoning, "action": text})
        # Keep only the last 20 blocks
        self._reasoning_blocks = self._reasoning_blocks[-20:]
        self._current_reasoning = ""
        self._save()
        self._emit("reasoning-action", {"text": text})

    def on_kb_update(self, in_run_kb: dict[str, str], cross_run_kb: dict[str, str]) -> None:
        """Push knowledge base contents to the OBS overlay."""
        self._last_kb = {"in_run": dict(in_run_kb), "cross_run": dict(cross_run_kb)}
        self._save()
        self._emit("kb-update", self._last_kb)

    def on_status_update(self, status: str) -> None:
        """Push agent status (e.g. 'thinking', 'reflecting', 'waiting') to the OBS overlay."""
        self._emit("status-update", {"value": status})

    def on_model_update(self, model: str) -> None:
        """Push model name to the OBS overlay."""
        self._emit("model-update", {"value": model})

    def on_cost_update(self, cost_usd: float) -> None:
        """Push lifetime cost to the OBS overlay."""
        self._emit("cost-update", {"value": cost_usd})

    def on_game_over(self, score: int) -> None:
        """Call when game over is reached."""
        if score > self.high_score:
            self.high_score = score
            self._save()
            self._emit("high-score-update", {"value": self.high_score})
