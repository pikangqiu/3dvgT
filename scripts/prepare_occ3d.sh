#!/usr/bin/env bash
set -euo pipefail

OCC3D_ROOT="${OCC3D_ROOT:-data/occ3d}"
OCC3D_ARCHIVE_URL="${OCC3D_ARCHIVE_URL:-}"
OCC3D_ARCHIVE_PATH="${OCC3D_ARCHIVE_PATH:-${OCC3D_ROOT}/occ3d.zip}"
DOWNLOAD="false"

usage() {
  cat <<EOF
Usage: bash scripts/prepare_occ3d.sh [--download]

Environment:
  OCC3D_ROOT          Target root for Occ3D/OpenOccupancy labels. Default: data/occ3d
  OCC3D_ARCHIVE_URL   Optional user-provided archive URL after license/access approval.
  OCC3D_ARCHIVE_PATH  Optional output archive path. Default: \${OCC3D_ROOT}/occ3d.zip

The default mode creates the target root and prints manual setup steps. Use --download only
when OCC3D_ARCHIVE_URL points to an archive you are allowed to download.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download)
      DOWNLOAD="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OCC3D_ROOT}"

if [[ "${DOWNLOAD}" == "true" ]]; then
  if [[ -z "${OCC3D_ARCHIVE_URL}" ]]; then
    echo "OCC3D_ARCHIVE_URL is required when using --download" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${OCC3D_ARCHIVE_PATH}")"
  curl -L "${OCC3D_ARCHIVE_URL}" -o "${OCC3D_ARCHIVE_PATH}"
fi

cat <<EOF
Occ3D root prepared at: ${OCC3D_ROOT}

Occ3D/OpenOccupancy access may require following the dataset author's download
instructions and license terms. This script avoids silent downloads unless you pass
--download with OCC3D_ARCHIVE_URL.

Expected benchmark layout after extraction:
  ${OCC3D_ROOT}/occ3d-nuscenes/
  ${OCC3D_ROOT}/occ3d-nuscenes/trainval/gts/<scene_name>/<sample_token>/labels.npz
  ${OCC3D_ROOT}/occ3d-nuscenes/trainval/annotations.json

The label attachment script also accepts locally reorganized labels at:
  ${OCC3D_ROOT}/occ3d-nuscenes/gts/<scene_name>/<sample_token>/labels.npz

Recommended next steps:
  1. Follow Occ3D: https://tsinghua-mars-lab.github.io/Occ3D/
  2. Follow OpenOccupancy reference code: https://github.com/JeffWang987/OpenOccupancy
  3. Extract labels under ${OCC3D_ROOT}/occ3d-nuscenes/
  4. Keep nuScenes data under data/nuscenes and run:
     PYTHONPATH=src python scripts/check_nuscenes.py --root data/nuscenes --version v1.0-trainval
  5. Attach public labels to the eval manifest:
     PYTHONPATH=src python scripts/attach_occ3d_labels.py --manifest data/manifests/nuscenes-mini.val.jsonl --occ3d-root ${OCC3D_ROOT} --output data/manifests/nuscenes-mini.val.occ3d.jsonl --nuscenes-root data/nuscenes --nuscenes-version v1.0-trainval
  6. Keep local LiDAR proxy occupancy and public semantic occupancy reports separate:
     local proxy metric: bev_occupancy_iou
     public benchmark metric: occupancy_miou / class IoU

Optional explicit download:
  OCC3D_ARCHIVE_URL=<approved_url> bash scripts/prepare_occ3d.sh --download
EOF
