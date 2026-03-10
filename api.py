"""HTTP client for the STS2 agent mod server."""

import requests
from typing import Any


class STS2API:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()

    def get_state(self) -> dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def wait_for_state(self, timeout_ms: int = 30000) -> dict[str, Any]:
        resp = self.session.get(
            f"{self.base_url}/state/wait",
            params={"timeout": timeout_ms},
            timeout=timeout_ms / 1000 + 5,
        )
        resp.raise_for_status()
        return resp.json()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/action", json=action)
        resp.raise_for_status()
        return resp.json()

    def get_map(self) -> dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/map")
        resp.raise_for_status()
        return resp.json()
