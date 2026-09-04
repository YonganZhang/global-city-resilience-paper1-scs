#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${1:-${PACKAGE_ROOT}/raw}"
mkdir -p "${RAW_DIR}"

download() {
  local filename="$1"
  local url="$2"
  local sha256="$3"
  local target="${RAW_DIR}/${filename}"
  local partial="${target}.part"

  if [[ -f "${target}" ]] && printf '%s  %s\n' "${sha256}" "${target}" | sha256sum --check --status; then
    echo "[skip] ${filename} already exists and passed checksum verification"
    return
  fi

  echo "[download] ${filename}"
  curl --fail --location --continue-at - \
    --connect-timeout 30 --max-time 21600 --retry 3 --retry-delay 5 \
    --output "${partial}" "${url}"
  mv "${partial}" "${target}"
  printf '%s  %s\n' "${sha256}" "${target}" | sha256sum --check
}

download "PGA_250y.tif" \
  "https://hazards-data.unepgrid.ch/PGA_250y.tif" \
  "3318d7d6610504e0bf85019328f8d9f1f9ae5d8d3515ede1f6c97ff1b57cc030"

download "global_pc_h100glob.tif" \
  "https://hazards-data.unepgrid.ch/global_pc_h100glob.tif" \
  "83e586bc2eb0245115c5ea78bc6ddc327a35077f0debd9928d53cd77b57e7b64"

download "Wind_CC_T100.tif" \
  "https://hazards-data.unepgrid.ch/Wind_CC_T100.tif" \
  "487ed782ac4e2a84e544cdc7bcfb310e3329b15c3580f41d35f4184bb4658a30"

download "drought_spi6.tif" \
  "https://hazards-data.unepgrid.ch/spi-06_past_dur_GF.tif" \
  "040a891db2ab38171116e41a7ba513057818a2bea949a626d9d0adfa35c1d2c7"

download "landslide_eq.tif" \
  "https://hazards-data.unepgrid.ch/n1_mosaic_wgs84_opt.tif" \
  "26bd4601219537a1a15b35d2872dd98f8b5bd4f19a599c6290262a57515a6973"

download "landslide_rain.tif" \
  "https://hazards-data.unepgrid.ch/n2_mosaic_wgs84_opt.tif" \
  "14d9de3af2d6798f66ed507f2f1c801f3c87cf2205928b349731e2e2747aa5fe"

download "Tsunami_hazard.gpkg" \
  "https://hazards-data.unepgrid.ch/Tsunami_hazard_results.gpkg" \
  "5c4382f2a160d03fecfd1fb6fd0d939649310c26a3fb1858b02d3f5c44bf14e3"

echo "All raw hazard files were downloaded and verified: ${RAW_DIR}"
