#!/usr/bin/env bash
# Archive duplicate BOS-BMAC snapshots on Desktop; keep only ~/Desktop/bos-bmac as mainline.
set -euo pipefail

DESKTOP="${HOME}/Desktop"
ARCHIVE="${DESKTOP}/_bos_bmac_archive"
MAIN="${DESKTOP}/bos-bmac"
STAMP="$(date +%Y%m%d)"

mkdir -p "${ARCHIVE}"

move_if_exists() {
  local src="$1"
  if [[ -e "${src}" ]]; then
    local base
    base="$(basename "${src}")"
    local dest="${ARCHIVE}/${base}.${STAMP}"
    if [[ -e "${dest}" ]]; then
      dest="${ARCHIVE}/${base}.${STAMP}.$$"
    fi
    mv "${src}" "${dest}"
    echo "archived: ${src} -> ${dest}"
  fi
}

if [[ ! -d "${MAIN}" ]]; then
  echo "ERROR: main project missing at ${MAIN}" >&2
  exit 1
fi

for item in \
  bos-bmac-full-latest \
  bmac_engine-latest \
  bos-bmac-examples-latest \
  bos-bmac-README-latest.md \
  bos_glue_example.py \
  bmac_demo.py \
  end_to_end_toy.py \
  numerical_toy_validation.py \
  run_all.py \
  setup.py \
  pyproject.toml \
  BOS-BMAC_Phase0_Impl_Status.txt \
  BOS-BMAC_Phase0_Spec_v1.0.md \
  BOS-BMAC_Phase0_Spec_v1.0.tex
do
  move_if_exists "${DESKTOP}/${item}"
done

cat > "${ARCHIVE}/README.txt" <<EOF
Archived ${STAMP}. Authoritative project: ${MAIN}
Do not edit archived copies. Use: cd ${MAIN} && PYTHONPATH=. python3 examples/run_all.py
To use external bos-platform: export BOS_PLATFORM_PATH=/path/to/checkout
EOF

echo "Done. Mainline: ${MAIN}"
