#!/usr/bin/env bash
set -euo pipefail

NUSCENES_ROOT="${NUSCENES_ROOT:-data/nuscenes}"
mkdir -p "${NUSCENES_ROOT}"

cat <<EOF
nuScenes root prepared at: ${NUSCENES_ROOT}

Manual download is required because nuScenes access depends on an account and license.

Expected minimum layout:
  ${NUSCENES_ROOT}/samples/
  ${NUSCENES_ROOT}/sweeps/
  ${NUSCENES_ROOT}/v1.0-trainval/ or ${NUSCENES_ROOT}/v1.0-mini/
  ${NUSCENES_ROOT}/maps/

Recommended next steps:
  1. Download nuScenes from https://www.nuscenes.org/download
  2. Download the map expansion and extract basemap/expansion/prediction into maps/
  3. Run: python -c "from nuscenes.nuscenes import NuScenes; NuScenes(version='v1.0-mini', dataroot='${NUSCENES_ROOT}', verbose=True)"
EOF

