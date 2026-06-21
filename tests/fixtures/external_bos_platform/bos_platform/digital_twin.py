from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EXTERNAL_FIXTURE = True


class DigitalTwin:
    def __init__(self, x0: List[float], noise: float = 0.0):
        self.x0 = list(x0)
        self.noise = noise

    def sample_robust_parameters(self, n: int = 10) -> List[Dict[str, Any]]:
        return [{"k": [1.0] * 5, "alpha_nominal": [0.6, 0.85, 1.7], "x0_pert": self.x0} for _ in range(n)]

    def simulate_trajectory(
        self,
        x0,
        alpha,
        T: float = 1.0,
        alpha_seq=None,
        dt=None,
        L=None,
    ) -> Tuple[np.ndarray, list]:
        x = np.asarray(x0, dtype=float)
        a = np.asarray(alpha, dtype=float)
        return x * (1.0 + 0.01 * float(np.sum(a)) * T), []

    def refine_from_observations(self, observations, L=None, **kwargs):
        return {"improvement_pct": 0.0, "fidelity": 1.0}
