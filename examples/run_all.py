#!/usr/bin/env python3
"""
run_all.py
Convenience script to run all Phase 0 tests and examples.

Usage:
    cd /Users/lei/Desktop/bos-bmac
    /Library/Developer/CommandLineTools/usr/bin/python3 examples/run_all.py

(editable install makes bmac_engine/bos_platform importable; PYTHONPATH=. still recommended for dev changes)
"""
import subprocess
import sys
import os
import shutil

def find_good_python():
    # Hardcode the recommended for this machine/project (CLT python has the deps)
    return '/Library/Developer/CommandLineTools/usr/bin/python3'

def run(cmd):
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)

def ensure_pytest(py, base):
    import_check = subprocess.run(
        f"{base} {py} -c 'import pytest'",
        shell=True,
        capture_output=True,
        text=True,
    )
    if import_check.returncode == 0:
        return
    print("pytest not found; installing dev dependency...")
    install = subprocess.run(
        f"{base} {py} -m pip install --user 'pytest>=7.0' -q",
        shell=True,
    )
    if install.returncode != 0:
        print("FAILED: could not install pytest")
        sys.exit(1)


def run_tests(py, base):
    """Run pytest; fail on any test failure."""
    ensure_pytest(py, base)
    run(f"{base} {py} -m pytest tests/ -q --tb=short")

def main():
    print("=== BOS-BMAC Phase 0: Run All ===")
    py = find_good_python()
    print(f"Using python: {py}")
    base = "PYTHONPATH=."
    run_tests(py, base)
    run(f"{base} {py} examples/end_to_end_toy.py")
    run(f"{base} {py} examples/numerical_toy_validation.py")
    run(f"{base} {py} examples/bos_glue_example.py")
    run(f"{base} {py} examples/phase1_digital_twin_robust_demo.py")
    run(f"{base} {py} examples/correspondence_verification.py")
    print("\n=== ALL Phase 0 tests and demos PASSED ===")

if __name__ == "__main__":
    main()
