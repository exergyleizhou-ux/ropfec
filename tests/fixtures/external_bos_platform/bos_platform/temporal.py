from __future__ import annotations

from typing import Any, Dict, List, Optional

EXTERNAL_FIXTURE = True


class TemporalWorkflow:
    def __init__(self):
        self.workflows: List[Dict[str, Any]] = []

    def start_workflow(self, name: str) -> Dict[str, Any]:
        wf = {"name": name, "checkpoints": [], "step": 0}
        self.workflows.append(wf)
        return wf

    def advance(self, wf: Dict[str, Any], n: int = 1) -> None:
        wf["step"] = wf.get("step", 0) + n

    def checkpoint(self, wf: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> None:
        wf["checkpoints"].append({"state": dict(state or {})})
