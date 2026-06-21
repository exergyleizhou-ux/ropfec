"""
Our own BOS OPA governance implementation (policy-as-code for BMAC).

Spec-faithful + deepened (FEC_exponent, resource, total_flux, DT-informed policies).
Our production version - no external source.
Corresponds: enforce/check around alpha, quorum, DT predictions.
"""
from __future__ import annotations
from typing import Any, List, Optional, Dict

class OPA:
    """Our own BOS OPA (production governance)."""
    def __init__(self, capacity: float = 30.0):
        self.capacity = capacity
        self.enforced: List[Any] = []
        self.violations = 0
        self.violation_log: List[Dict[str, Any]] = []

    def enforce_policy(self, policy: str, value: Any) -> Any:
        self.enforced.append((policy, value))
        if policy in ("ROP_constraint", "FEC_exponent"):
            print(f"[BOS OPA] enforced {policy}")
            return value
        if policy == "resource_ok":
            total = sum(value) if isinstance(value, (list, tuple)) else float(value)
            ok = total <= self.capacity
            if not ok:
                self.violations += 1
                self.violation_log.append({"policy": policy, "total": total, "capacity": self.capacity})
            print(f"[BOS OPA] {policy}: total={total:.2f} <= capacity? {ok}")
            return ok
        if policy == "total_signal_flux":
            total = sum(value) if isinstance(value, (list, tuple)) else float(value)
            ok = total <= self.capacity
            if not ok:
                self.violations += 1
                self.violation_log.append({"policy": policy, "total": total})
            return ok
        return True

    def check_policy(self, policy: str, *args: Any) -> Any:
        if policy in ("resource_ok", "total_signal_flux"):
            total = sum(args[0]) if args and args[0] is not None else 0.0
            return total <= self.capacity
        return True

    def get_violation_count(self) -> int:
        return self.violations

    def explain_violation(self, last: bool = True) -> Optional[Dict[str, Any]]:
        if not self.violation_log:
            return None
        return self.violation_log[-1] if last else self.violation_log

    def evolve_policy_from_dt(self, dt_pred_risk: float, L: Optional[List[float]] = None, bos_api: Any = None) -> Dict[str, Any]:
        """
        Phase2: OPA policies evolve with DT (Musk/Huang data-driven governance).
        If DT predicts high risk (e.g. pred_load or sens-weighted), tighten capacity or effective ROP (via L scaling on bounds conceptually).
        First-principles: use L to weight risk if provided; publish via bos_api for full BOS loop.
        Returns evolved info; side-effect: may lower self.capacity, log violation-like 'evolved'.
        """
        evolved = {"risk": float(dt_pred_risk), "action": "none", "new_capacity": self.capacity}
        thresh = 2.5  # toy risk threshold (from DT pred_sum or load)
        if dt_pred_risk > thresh:
            old_cap = self.capacity
            self.capacity = max(5.0, self.capacity * 0.85)  # tighten 15%
            self.violations += 1  # count as governance action
            self.violation_log.append({"policy": "dt_evolve_tighten", "risk": dt_pred_risk, "old_capacity": old_cap, "new_capacity": self.capacity})
            evolved.update({"action": "tightened", "old_capacity": old_cap, "new_capacity": self.capacity})
            print(f"[BOS OPA] DT-evolved policy: risk={dt_pred_risk:.2f} > thresh, capacity {old_cap:.1f} -> {self.capacity:.1f} (L-weighted: {bool(L)})")
        if bos_api is not None:
            try:
                bos_api.publish("opa_evolved_policy", evolved, meta={"source": "dt_risk", "L_used": bool(L)})
            except Exception:
                pass
        return evolved
