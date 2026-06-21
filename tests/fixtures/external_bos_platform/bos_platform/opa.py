from __future__ import annotations

from typing import Any, Optional

EXTERNAL_FIXTURE = True


class OPA:
    def __init__(self, capacity: float = 100.0):
        self.capacity = capacity
        self.violations = 0

    def enforce_policy(self, policy: str, value: Any, **kwargs) -> Any:
        return value

    def check_policy(self, policy: str, value: Any, **kwargs) -> bool:
        return True

    def get_violation_count(self) -> int:
        return self.violations

    def evolve_policy_from_dt(self, risk: float, L=None, bos_api=None, **kwargs):
        return None
