"""
Adapter HTTP demo for BOS_PLATFORM_PATH=~/Projects/bos-platform/bmac_adapter.

- If BOS_API_BASE_URL is NOT set: verifies adapter loads (delegate mode) and runs a toy DT sim.
- If BOS_API_BASE_URL IS set: requires that the server responds to POST /api/v1/twin/run
  (the adapter stores the last HTTP response on dt.last_http_result).
"""

from __future__ import annotations

import os
import sys


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> None:
    # Ensure project import
    sys.path.insert(0, ".")

    from bos_platform.loader import load_bos_platform

    bundle = load_bos_platform(prefer_external=True)
    print("bos_platform source:", bundle.source)

    dt = bundle.DigitalTwin([1.0, 1.0, 1.0], noise=0.05)
    x_final, _ = dt.simulate_trajectory([1.0, 1.0, 1.0], [0.6, 0.85, 1.7], T=1.0, dt=0.1)
    require(x_final is not None, "DT simulate failed unexpectedly")

    base = os.environ.get("BOS_API_BASE_URL", "").strip()
    if base:
        # In HTTP mode, we require at least one successful response.
        got = getattr(dt, "last_http_result", None)
        require(got is not None, f"BOS_API_BASE_URL set ({base}) but /api/v1/twin/run call failed")
        require("final_state" in got, "twin/run response missing final_state")
        print("HTTP twin/run ok: evidence_level=", got.get("evidence_level"))
    else:
        print("BOS_API_BASE_URL not set; delegate-only demo OK.")


if __name__ == "__main__":
    main()

