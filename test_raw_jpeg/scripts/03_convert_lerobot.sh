#!/usr/bin/env bash
# Stage 3: run the real converter over both sources at each CRF.
#
# Nine datasets: {raw, jpeg100, jpeg80} x {crf0, crf20, crf30}.  Produced by
# rosbag2_to_lerobotv3.py itself rather than by a standalone ffmpeg call, so the
# sizes quoted in the report are the sizes an actual training dataset would have.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
PYTHON="/home/kewei/anaconda3/envs/lerobot/bin/python"
CONVERTER="${REPO_DIR}/tool/rosbag2_to_lerobotv3.py"

RAW_INPUT="$(${PYTHON} -c "import json,pathlib;print(pathlib.Path(json.loads(open('${ROOT_DIR}/config.json').read())['source_raw_bag']))")"
JPEG80_INPUT="${ROOT_DIR}/bags/express_jpeg80"
JPEG100_INPUT="${ROOT_DIR}/bags/express_jpeg100"
RAW_PROFILE="${REPO_DIR}/tool/profiles/marvin-gripper-quadtile.json"
JPEG_PROFILE="${ROOT_DIR}/profiles/marvin-gripper-quadtile-compressed.json"

run_one() {
  local source_name="$1" input="$2" profile="$3" crf="$4"
  local output="${ROOT_DIR}/lerobot/${source_name}_crf${crf}"
  if [[ -d "${output}" ]]; then
    echo "== skip ${source_name} crf=${crf} (exists)"
    return
  fi
  echo "== convert ${source_name} crf=${crf} -> ${output}"
  local start
  start=$(date +%s.%N)
  "${PYTHON}" "${CONVERTER}" \
    --input "${input}" \
    --output "${output}" \
    --profile "${profile}" \
    --repo-id local/express \
    --task express \
    --fps 30 \
    --state-tolerance-ms 20 \
    --video-codec h264 \
    --crf "${crf}" \
    --progress none \
    > "${ROOT_DIR}/results/convert_${source_name}_crf${crf}.log" 2>&1
  local end
  end=$(date +%s.%N)
  echo "   done in $(echo "${end} - ${start}" | bc)s"
  echo "${source_name},${crf},$(echo "${end} - ${start}" | bc)" >> "${ROOT_DIR}/results/convert_timings.csv"
}

mkdir -p "${ROOT_DIR}/results"
if [[ ! -f "${ROOT_DIR}/results/convert_timings.csv" ]]; then
  echo "source,crf,seconds" > "${ROOT_DIR}/results/convert_timings.csv"
fi

for crf in 0 20 30; do
  run_one raw "${RAW_INPUT}" "${RAW_PROFILE}" "${crf}"
done
for crf in 0 20 30; do
  run_one jpeg100 "${JPEG100_INPUT}" "${JPEG_PROFILE}" "${crf}"
done
for crf in 0 20 30; do
  run_one jpeg80 "${JPEG80_INPUT}" "${JPEG_PROFILE}" "${crf}"
done

echo "stage 3 complete"
