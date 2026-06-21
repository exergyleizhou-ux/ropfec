#!/usr/bin/env python3
"""
bmac_demo.py
Top-level demo for BOS-BMAC Phase 0.

Runs a full chain using the toy from spec, showing the mapping.

Usage:
    python bmac_demo.py
"""
import sys
sys.path.insert(0, ".")
from examples.end_to_end_toy import main as run_e2e

if __name__ == "__main__":
    print("BOS-BMAC Phase 0 Demo (see spec for details)")
    run_e2e()
    print("\nSee examples/ for more (numerical, glue, etc.).")
    print("Run 'python examples/run_all.py' for full verification.")