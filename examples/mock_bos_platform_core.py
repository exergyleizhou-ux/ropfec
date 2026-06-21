"""
Minimal BOS Platform Core (mock) — HTTP contract harness.

Purpose:
- Provide a local FastAPI server that implements the *shape* of bos-platform's
  Phase A V5 endpoint `POST /api/v1/twin/run` and `GET /api/v1/health/live`.
- Enables end-to-end adapter HTTP verification without Docker / real backend.

This is NOT the real bos-platform implementation. It is a contract harness so
`bmac_adapter` can be exercised in HTTP mode deterministically.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException


app = FastAPI(title="BOS Mock Core", version="0.0.0")


@dataclass
class _TwinState:
    biomass_kg: float
    substrate_kg: float
    temperature_c: float
    moisture_pct: float
    nitrogen_kg: float


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@app.get("/api/v1/health/live")
def health_live() -> Dict[str, Any]:
    return {"status": "ok", "ts": _now_iso()}


@app.post("/api/v1/twin/run")
def twin_run(
    body: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    # Optional auth: accept any Bearer token, reject clearly if malformed when present.
    if authorization is not None and not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="invalid Authorization header")

    initial = body.get("initial_state") or {}
    inputs = body.get("inputs") or []
    if not isinstance(inputs, list) or len(inputs) < 1:
        raise HTTPException(status_code=422, detail="inputs must be a non-empty list")

    # Minimal validation + defaulting (contract harness, not strict pydantic)
    s = _TwinState(
        biomass_kg=float(initial.get("biomass_kg", 0.5)),
        substrate_kg=float(initial.get("substrate_kg", 10.0)),
        temperature_c=float(initial.get("temperature_c", 28.0)),
        moisture_pct=float(initial.get("moisture_pct", 70.0)),
        nitrogen_kg=float(initial.get("nitrogen_kg", 0.05)),
    )

    traj: List[Dict[str, Any]] = []
    est: List[Dict[str, Any]] = []

    cumulative = 0.0
    for i, step in enumerate(inputs):
        dt_h = float(step.get("dt_hours", 1.0))
        feed = float(step.get("feed_rate_kg_h", 0.0))
        cumulative += dt_h
        # Toy evolution: substrate decreases, biomass increases with feed.
        s.substrate_kg = max(0.0, s.substrate_kg + feed * dt_h - 0.1 * s.biomass_kg * dt_h)
        s.biomass_kg = max(0.0, s.biomass_kg + 0.05 * feed * dt_h)

        snapshot = {
            "step_index": i,
            "cumulative_time_h": round(cumulative, 4),
            "state": asdict(s),
            "growth_rate_kg_h": round(0.01 * s.biomass_kg, 6),
            "ser_instantaneous": round(0.2 + 0.001 * s.biomass_kg, 6),
        }
        traj.append(snapshot)
        est.append(asdict(s))

    resp = {
        "trajectory": traj,
        "estimated_states": est,
        "final_state": est[-1],
        "innovation_stats": {
            "n_updates": 0,
            "mean_abs_innovation": {},
            "rms_innovation": {},
        },
        "evidence_level": "planned",
        "engine_version": "9.0.0",
    }
    return resp

