#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PYTHON="/home/kewei/anaconda3/envs/test/bin/python"
DINO_PYTHON="/home/kewei/anaconda3/envs/dino/bin/python"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

"${TEST_PYTHON}" "${ROOT_DIR}/scripts/encode_and_pixel_eval.py" "$@"
"${DINO_PYTHON}" "${ROOT_DIR}/scripts/dinov3_feature_eval.py"
"${DINO_PYTHON}" "${ROOT_DIR}/scripts/make_report.py"
