#!/usr/bin/env bash
# Run the ROSbag quality checker in the local lerobot conda environment.
#
# Usage:
#   ./check_rosbag_quality.sh /path/to/rosbag [--profile NAME] [其他选项]
#   RECORD_ROOT=/path/to/bags ./check_rosbag_quality.sh          # 自动选最新
#
# Without a bag argument the newest rosbag under $RECORD_ROOT is checked.
# RECORD_ROOT has no default: there is no canonical recording directory in this
# repository, so it must be pointed at wherever the bags actually live.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 && "$1" != -* ]]; then
    BAG_PATH="$1"
    shift
else
    if [[ -z "${RECORD_ROOT:-}" ]]; then
        echo "用法：$0 /path/to/rosbag [检查选项]"
        echo "或先设置录制根目录：RECORD_ROOT=/path/to/bags $0"
        exit 2
    fi
    if [[ ! -d "$RECORD_ROOT" ]]; then
        echo "错误：RECORD_ROOT 不是目录：$RECORD_ROOT"
        exit 2
    fi
    # rosbag2 directories are identified by their metadata.yaml, at any depth.
    latest_metadata="$(
        find "$RECORD_ROOT" -name metadata.yaml -type f -printf '%T@ %h\n' 2>/dev/null \
            | sort -nr | head -n 1 || true
    )"
    BAG_PATH="${latest_metadata#* }"
    if [[ -z "$latest_metadata" || -z "$BAG_PATH" || "$BAG_PATH" == "$latest_metadata" ]]; then
        echo "错误：$RECORD_ROOT 下没有可检查的 rosbag（未找到 metadata.yaml）。"
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
