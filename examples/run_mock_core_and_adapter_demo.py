"""
One-shot E2E: start mock BOS core → run adapter_http_demo → stop.

This makes the adapter HTTP path verifiable without Docker / real bos-platform.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request


def _wait_ok(url: str, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def main() -> int:
    py = os.environ.get("PY", sys.executable)
    base = "http://127.0.0.1:8765"

    env = os.environ.copy()
    env.setdefault("BOS_API_BASE_URL", base)
    env.setdefault("BOS_API_TOKEN", "mock-token")
    env.setdefault("BMAC_HOME", os.path.abspath("."))
    env.setdefault("BOS_PLATFORM_PATH", os.path.expanduser("~/Projects/bos-platform/bmac_adapter"))
    env.setdefault("PYTHONPATH", ".")

    core = subprocess.Popen(
        [
            py,
            "-m",
            "uvicorn",
            "examples.mock_bos_platform_core:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--log-level",
            "warning",
        ],
        env=env,
    )
    try:
        if not _wait_ok(f"{base}/api/v1/health/live", 15.0):
            print("FAIL: mock core did not become healthy")
            return 1
        p = subprocess.run([py, "examples/adapter_http_demo.py"], env=env)
        return int(p.returncode)
    finally:
        try:
            core.send_signal(signal.SIGTERM)
        except Exception:
            pass
        try:
            core.wait(timeout=5)
        except Exception:
            try:
                core.kill()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

