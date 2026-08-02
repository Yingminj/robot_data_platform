#!/usr/bin/env bash
# Run the ROSbag quality checker in the local lerobot conda environment.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BAG_ROOT="${RECORD_ROOT:-$SCRIPT_DIR/recorded_bags}"

if [[ $# -gt 0 && "$1" != -* ]]; then
    BAG_PATH="$1"
    shift
else
    latest_metadata="$(
        find "$DEFAULT_BAG_ROOT" -mindepth 2 -maxdepth 2 -name metadata.yaml \
            -type f -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 || true
    )"
    BAG_PATH="${latest_metadata#* }"
    if [[ -z "$BAG_PATH" || "$BAG_PATH" == "$latest_metadata" ]]; then
        echo "错误：$DEFAULT_BAG_ROOT 下没有可检查的 rosbag。"
        echo "用法：$0 /path/to/rosbag [检查选项]"
        exit 2
    fi
    echo "自动选择最新 rosbag：$BAG_PATH"
fi

LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/kewei/anaconda3/envs/lerobot/bin/python}"
if [[ ! -x "$LEROBOT_PYTHON" ]]; then
    echo "错误：找不到 lerobot 环境的 Python：$LEROBOT_PYTHON"
    exit 2
fi

exec "$LEROBOT_PYTHON" "$SCRIPT_DIR/check_rosbag_quality.py" "$BAG_PATH" "$@"
