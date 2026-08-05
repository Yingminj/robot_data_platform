#!/usr/bin/env bash
# 把一个录制根目录下的所有 MCAP 采集批次转换成 LeRobot v3 数据集。
#
# 目录结构假设（dataset/new_mcap_recorded_data 就是这个形状）：
#
#   <录制根目录>/
#     express/                     <- 一个采集批次 = 一个数据集 = 一个 task
#       my_bag-26-08-04-14-07-46/
#         data/                    <- 真正的 rosbag（metadata.yaml + *.mcap）
#         video/
#       my_bag-.../
#     另一个批次/
#       ...
#
# 每个批次转成一个独立的数据集，因为 LeRobot 要求同一个数据集里所有 episode 暴露
# 相同的特征集，而不同批次可能相机数量、末端执行器都不一样。转换器自己会递归找
# metadata.yaml，所以中间多一层 data/ 不用管。
#
# 用法：
#   RECIPE=<配方> ./scripts/convert_batches.sh <录制根目录> [-- 额外参数...]
#
#   RECIPE=mcap-gripper-quadtile ./scripts/convert_batches.sh ../dataset/new_mcap_recorded_data
#   RECIPE=mcap-gripper-quadtile-raw ./scripts/convert_batches.sh ../dataset/.../express_raw
#   RECIPE=db3-gripper ./scripts/convert_batches.sh ../dataset/old_db3_data
#   CRF=28 RECIPE=mcap-dexhand ./scripts/convert_batches.sh ../dataset/dexhand
#   DRY_RUN=1 RECIPE=mcap-gripper-quadtile ./scripts/convert_batches.sh ../dataset/...
#
# 一个批次一个配方：配方决定 profile、相机话题与对齐宽严，混批会因为特征集不同而失败。
# 用 `rdp recipes` 查看有哪些配方；用 CHECK=1 在转换前先核对话题是否对得上。
#
# 环境变量：
#   RECIPE         转换配方（必填），例如 mcap-gripper-quadtile / db3-gripper
#   OUTPUT_ROOT    输出根目录（默认 <录制根目录的同级>/lerobot）
#   PROFILE        覆盖配方的 profile（少用；会同时丢弃配方里的相机话题覆盖）
#   FPS            目标帧率（默认取配方设置）
#   CRF            视频质量，最常调整（默认取配方设置）
#   CODEC          视频编码器（默认取配方设置）
#   TASK_MAP       "批次名=任务描述" 的文件，每行一条；未列出的批次回退到批次名
#   OVERWRITE      1 = 覆盖已存在的数据集（默认跳过已转换的批次）
#   CHECK          1 = 转换前先跑 rdp check-rosbag，不通过就跳过该批次
#   DRY_RUN        1 = 只打印将要执行的命令
#   LEROBOT_PYTHON lerobot 环境的 python（默认 ~/anaconda3/envs/lerobot/bin/python）

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(dirname -- "$SCRIPT_DIR")"
RDP="$TOOL_DIR/rdp"

if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
    sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
fi

SOURCE_ROOT="$1"
shift
# 剩下的参数原样传给转换器；-- 只是为了可读，可有可无。
[[ ${1:-} == "--" ]] && shift
EXTRA_ARGS=("$@")

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "错误：不是目录：$SOURCE_ROOT" >&2
    exit 2
fi
SOURCE_ROOT="$(cd -- "$SOURCE_ROOT" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname -- "$SOURCE_ROOT")/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$HOME/anaconda3/envs/lerobot/bin/python}"

if [[ -z "${RECIPE:-}" && -z "${PROFILE:-}" ]]; then
    echo "错误：必须指定 RECIPE（或 PROFILE）。可用配方：" >&2
    "$LEROBOT_PYTHON" "$RDP" recipes 2>/dev/null | head -10 >&2
    exit 2
fi

# 配方提供的默认值不在这里重复；只有显式设置的环境变量才会传下去，
# 从而保持「命令行 > 配方 > 内置默认」这一个优先级顺序。
COMMON_ARGS=()
[[ -n "${RECIPE:-}" ]]  && COMMON_ARGS+=(--recipe "$RECIPE")
[[ -n "${PROFILE:-}" ]] && COMMON_ARGS+=(--profile "$PROFILE")
CONVERT_ARGS=()
[[ -n "${FPS:-}" ]]     && CONVERT_ARGS+=(--fps "$FPS")
[[ -n "${CRF:-}" ]]     && CONVERT_ARGS+=(--crf "$CRF")
[[ -n "${CODEC:-}" ]]   && CONVERT_ARGS+=(--video-codec "$CODEC")

if [[ ! -x "$LEROBOT_PYTHON" ]]; then
    echo "错误：找不到 lerobot 环境的 Python：$LEROBOT_PYTHON" >&2
    echo "      用 LEROBOT_PYTHON=/path/to/python 指定。" >&2
    exit 2
fi

# 一个 rosbag 目录由 metadata.yaml 标识，深度不限。
count_bags() {
    find "$1" -name metadata.yaml -type f 2>/dev/null | wc -l
}

total_bags="$(count_bags "$SOURCE_ROOT")"
if [[ "$total_bags" -eq 0 ]]; then
    echo "错误：$SOURCE_ROOT 下没有找到任何 rosbag（没有 metadata.yaml）。" >&2
    exit 2
fi

# 判断传进来的是「批次的根目录」还是「单个批次」：
# 批次目录含有多个 bag（express/ 下有 51 个），而单个批次的子目录是 bag 本身，
# 每个只含 1 个。所以只要有任何一个子目录含 >1 个 bag，就按批次根目录处理。
COLLECTIONS=()
for entry in "$SOURCE_ROOT"/*/; do
    [[ -d "$entry" ]] || continue
    [[ "$(count_bags "$entry")" -gt 1 ]] || continue
    COLLECTIONS+=("${entry%/}")
done

if [[ ${#COLLECTIONS[@]} -eq 0 ]]; then
    # 子目录都只含 0~1 个 bag：根目录自己就是一个批次。
    COLLECTIONS=("$SOURCE_ROOT")
else
    # 混合结构下，别把只含 1 个 bag 的子目录漏掉——但它们属于哪个批次是无法猜的，
    # 所以只在总数对不上时报出来，让人自己决定，而不是悄悄少转。
    grouped=0
    for collection in "${COLLECTIONS[@]}"; do
        grouped=$((grouped + $(count_bags "$collection")))
    done
    if [[ "$grouped" -ne "$total_bags" ]]; then
        echo "警告：$SOURCE_ROOT 下共 $total_bags 个 bag，但按批次只归入了 $grouped 个。" >&2
        echo "      有 bag 不在任何批次子目录里，这部分不会被转换。" >&2
        echo >&2
    fi
fi

task_for() {
    local name="$1"
    if [[ -n "${TASK_MAP:-}" && -f "$TASK_MAP" ]]; then
        local line
        line="$(grep -m1 -- "^${name}=" "$TASK_MAP" 2>/dev/null || true)"
        if [[ -n "$line" ]]; then
            printf '%s\n' "${line#*=}"
            return
        fi
    fi
    printf '%s\n' "$name"
}

echo "录制根目录：$SOURCE_ROOT"
echo "输出根目录：$OUTPUT_ROOT"
echo "配方      ：${RECIPE:-（未指定，使用 PROFILE=$PROFILE）}"
echo "批次数量  ：${#COLLECTIONS[@]}"
echo

converted=(); skipped=(); failed=()

for collection in "${COLLECTIONS[@]}"; do
    name="$(basename -- "$collection")"
    bags="$(count_bags "$collection")"
    output="$OUTPUT_ROOT/$name"
    task="$(task_for "$name")"

    echo "── $name（$bags 个 bag）→ $output"

    if [[ -d "$output" && "${OVERWRITE:-0}" != "1" ]]; then
        echo "   已存在，跳过（OVERWRITE=1 可覆盖）"
        skipped+=("$name")
        continue
    fi

    if [[ "${CHECK:-0}" == "1" ]]; then
        echo "   质量检查中…"
        check_failed=0
        while IFS= read -r metadata; do
            bag_dir="$(dirname -- "$metadata")"
            if ! "$LEROBOT_PYTHON" "$RDP" check-rosbag \
                    "$bag_dir" "${COMMON_ARGS[@]}" >/dev/null 2>&1; then
                echo "   质量检查未通过：$bag_dir"
                check_failed=1
            fi
        done < <(find "$collection" -name metadata.yaml -type f | sort)
        if [[ "$check_failed" -eq 1 ]]; then
            echo "   跳过该批次（用 rdp check-rosbag 看具体原因）"
            failed+=("$name（质量检查）")
            continue
        fi
    fi

    cmd=(
        "$LEROBOT_PYTHON" "$RDP" convert
        --input "$collection"
        --output "$output"
        --repo-id "local/$name"
        --task "$task"
        "${COMMON_ARGS[@]}"
        "${CONVERT_ARGS[@]}"
    )
    [[ "${OVERWRITE:-0}" == "1" ]] && cmd+=(--overwrite)
    cmd+=("${EXTRA_ARGS[@]}")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '   [dry-run]'; printf ' %q' "${cmd[@]}"; printf '\n'
        continue
    fi

    # 单个批次失败不该中断其余批次；set -e 在这里要临时让路。
    if "${cmd[@]}"; then
        echo "   完成：$output"
        converted+=("$name")
    else
        echo "   失败：$name（上面是转换器的输出）" >&2
        failed+=("$name")
    fi
    echo
done

echo "════════════════════════════════════════"
echo "转换完成 ${#converted[@]}：${converted[*]:-无}"
echo "跳过     ${#skipped[@]}：${skipped[*]:-无}"
echo "失败     ${#failed[@]}：${failed[*]:-无}"

[[ ${#failed[@]} -eq 0 ]]
