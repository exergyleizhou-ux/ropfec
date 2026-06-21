"""Minimal external fixture — proves BOS_PLATFORM_PATH loader path."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

EXTERNAL_FIXTURE = True


class SignalAPI:
    def __init__(self):
        self.published: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def publish(self, key: str, value: Any, meta: Optional[Dict[str, Any]] = None) -> None:
        self.published[key] = value

    def get(self, key: str) -> Any:
        return self.published.get(key)

    def apply_control(self, alpha, meta: Optional[Dict[str, Any]] = None) -> None:
        self.published["last_control"] = alpha

    def publish_quorum(self, agent_id: int, value: float) -> None:
        self.published[f"quorum/{agent_id}"] = value
