from __future__ import annotations

from typing import Any, List, Optional

EXTERNAL_FIXTURE = True


class Kalman:
    def __init__(self):
        self.state: List[float] = [0.0, 0.0, 0.0]

    def update(self, x_meas: List[float], L: Optional[Any] = None) -> List[float]:
        self.state = list(x_meas)
        return self.state

    def get_covariance(self):
        return [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]

    def get_state(self):
        return list(self.state)
