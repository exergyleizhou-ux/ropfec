#!/usr/bin/env bash
# Resolve BOS_PLATFORM_PATH for bos-bmac integration.
# Writes .bos_platform_path in project root (source it: eval "$(./scripts/setup_bos_platform_path.sh --export)")
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$ROOT/tests/fixtures/external_bos_platform"
PROJECTS_ADAPTER="$HOME/Projects/bos-platform/bmac_adapter"
LOCAL_FILE="$ROOT/.bos_platform_path"

export BMAC_HOME="${BMAC_HOME:-$ROOT}"

choose_path() {
  if [[ -n "${BOS_PLATFORM_PATH:-}" ]] && [[ -d "${BOS_PLATFORM_PATH}/bos_platform" || -f "${BOS_PLATFORM_PATH}/signal_control.py" ]]; then
    echo "$BOS_PLATFORM_PATH"
    return 0
  fi
  if [[ -d "$PROJECTS_ADAPTER/bos_platform" ]]; then
    echo "$PROJECTS_ADAPTER"
    return 0
  fi
  if [[ -f "$LOCAL_FILE" ]]; then
    local saved
    saved="$(cat "$LOCAL_FILE")"
    if [[ -d "$saved" ]]; then
      echo "$saved"
      return 0
    fi
  fi
  if [[ -d "$FIXTURE/bos_platform" ]]; then
    echo "$FIXTURE"
    return 0
  fi
  return 1
}

if [[ "${1:-}" == "--export" ]]; then
  if path="$(choose_path)"; then
    echo "export BMAC_HOME='$BMAC_HOME'"
    echo "export BOS_PLATFORM_PATH='$path'"
  else
    echo "echo 'No BOS_PLATFORM_PATH candidate found'" >&2
    exit 1
  fi
  exit 0
fi

if path="$(choose_path)"; then
  echo "$path" > "$LOCAL_FILE"
  echo "BMAC_HOME=$BMAC_HOME"
  echo "BOS_PLATFORM_PATH=$path"
  echo "Saved to $LOCAL_FILE"
  echo ""
  echo "Usage:"
  echo "  eval \"\$(./scripts/setup_bos_platform_path.sh --export)\""
  echo "  export BMAC_HOME=\"$BMAC_HOME\""
  echo "  PYTHONPATH=. python3 -c \"from bos_platform.loader import load_bos_platform; print(load_bos_platform().source)\""
else
  echo "No compatible bos_platform package found." >&2
  echo "Clone adapter or set BOS_PLATFORM_PATH manually." >&2
  echo "Real repo: $HOME/Projects/bos-platform (HTTP API — needs bmac_adapter; see bos_platform/adapters/README.md)" >&2
  exit 1
fi
