#!/usr/bin/env bash
set -euo pipefail

SATELLITE_RASTER_ROOT="${SATELLITE_RASTER_ROOT:-data/satellite_rasters}"
SATELLITE_RASTER_CONFIG="${SATELLITE_RASTER_CONFIG:-${SATELLITE_RASTER_ROOT}/config.json}"
SATELLITE_RASTER_TEMPLATE="${SATELLITE_RASTER_TEMPLATE:-configs/satellite_rasters.example.json}"
OVERWRITE="false"

usage() {
  cat <<EOF
Usage: bash scripts/prepare_satellite_rasters.sh [--overwrite]

Environment:
  SATELLITE_RASTER_ROOT      Directory for local satellite rasters. Default: data/satellite_rasters
  SATELLITE_RASTER_CONFIG    Config path. Default: \${SATELLITE_RASTER_ROOT}/config.json
  SATELLITE_RASTER_TEMPLATE  Template config. Default: configs/satellite_rasters.example.json

The script creates the raster root and copies the template config when no config exists.
Edit config.json to point each nuScenes map_location to a local georeferenced raster.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE="true"
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

mkdir -p "${SATELLITE_RASTER_ROOT}"

if [[ ! -f "${SATELLITE_RASTER_TEMPLATE}" ]]; then
  echo "missing template: ${SATELLITE_RASTER_TEMPLATE}" >&2
  exit 2
fi

if [[ "${OVERWRITE}" == "true" || ! -f "${SATELLITE_RASTER_CONFIG}" ]]; then
  mkdir -p "$(dirname "${SATELLITE_RASTER_CONFIG}")"
  cp "${SATELLITE_RASTER_TEMPLATE}" "${SATELLITE_RASTER_CONFIG}"
fi

cat <<EOF
Satellite raster root prepared at: ${SATELLITE_RASTER_ROOT}
Satellite raster config: ${SATELLITE_RASTER_CONFIG}

Manual raster setup is still required for real training:
  1. Put local satellite rasters under ${SATELLITE_RASTER_ROOT}/
  2. Edit ${SATELLITE_RASTER_CONFIG}
  3. Set each map_location raster_path, origin_ego_xy_m, origin_pixel_xy, and meters_per_pixel
  4. Validate against a generated manifest:
     PYTHONPATH=src python scripts/check_satellite_rasters.py --config ${SATELLITE_RASTER_CONFIG} --manifest data/manifests/nuscenes-mini.jsonl
  5. Materialize aligned satellite crops:
     PYTHONPATH=src python scripts/materialize_satellite_crops.py data/manifests/nuscenes-mini.jsonl --config ${SATELLITE_RASTER_CONFIG} --output data/manifests/nuscenes-mini.satellite.jsonl

Use --overwrite only if you intentionally want to reset the config from the template.
EOF
