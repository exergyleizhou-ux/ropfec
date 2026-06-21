#!/bin/bash
# Setup script for BOS-BMAC external env
# Run with: bash setup_env.sh

set -e

PY="/Library/Developer/CommandLineTools/usr/bin/python3"

echo "Using Python: $PY"
$PY --version

echo "Installing matplotlib..."
$PY -m pip install --user matplotlib

echo "Installing casadi (may take time and bandwidth)..."
$PY -m pip install --user casadi || echo "casadi install failed or partial, see below for manual wheel"

echo "Installing project editable..."
$PY -m pip install --user -e .

echo "Adding python bin to PATH (for this session)..."
export PATH="/Users/lei/Library/Python/3.9/bin:$PATH"

echo "Verifying..."
$PY -c "
import numpy, scipy, matplotlib
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('matplotlib:', matplotlib.__version__)
try:
    import casadi
    print('casadi:', casadi.__version__)
except:
    print('casadi: not installed (fallback to scipy will be used)')
"

echo "Done. If casadi failed, manually download the wheel and install:"
echo "$PY -m pip install --user /path/to/casadi-*.whl --no-deps"
