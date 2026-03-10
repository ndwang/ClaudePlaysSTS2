"""OBS overlay integration via obs-websocket browser source events."""

from __future__ import annotations

import json
import logging
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
        state = _load_state()
        self.high_score: int = int(state.get("high_score", 0))
        self.total_rounds: int = int(state.get("total_rounds", 0))
        self._ws = None

        try:
            import obsws_python as obs
            self._ws = obs.ReqClient(host=obs_host, port=obs_port, password=obs_password, timeout=3)
            log.info("Connected to OBS WebSocket at %s:%d", obs_host, obs_port)
        except Exception as e:
            log.warning("OBS WebSocket not available (%s). Overlays won't update.", e)

        self._emit("high-score-update", {"value": self.high_score})
        self._emit("round-update", {"value": self.total_rounds})
        self._emit("timer-control", {"action": "start"})

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
            log.warning("Failed to emit '%s': %s", event_name, e)

    def _save(self) -> None:
        _save_state({"high_score": self.high_score, "total_rounds": self.total_rounds})

    def reset(self) -> None:
        """Reset all overlay values."""
        self.high_score = 0
        self.total_rounds = 0
        self._save()
        self._emit("timer-control", {"action": "reset"})
        self._emit("round-update", {"value": 0})
        self._emit("high-score-update", {"value": 0})

    def on_round(self) -> None:
        """Call each decision round."""
        self.total_rounds += 1
        self._save()
        self._emit("round-update", {"value": self.total_rounds})

    def on_exit(self) -> None:
        """Call when the agent is shutting down."""
        self._emit("timer-control", {"action": "pause"})

    def on_game_over(self, score: int) -> None:
        """Call when game over is reached."""
        if score > self.high_score:
            self.high_score = score
            self._save()
            self._emit("high-score-update", {"value": self.high_score})
