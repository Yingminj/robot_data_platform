#!/usr/bin/env python3
"""Stage 4: measure the MP4s the converter actually wrote.

For every {source, CRF} dataset each camera video is decoded and compared
against two references:

* ``vs_raw`` -- the uncompressed ground-truth tile.  This is the number that
  matters for "how far is the training frame from what the sensor saw", and for
  the JPEG datasets it includes the recording-side JPEG loss.
* ``vs_input`` -- the frames that were handed to the video encoder.  For the
  JPEG datasets this isolates what H.264 alone added on top of the JPEG, which
  is the only way to tell whether a lower CRF is still buying anything once the
  frames arrive pre-damaged.

For the raw datasets the two references are the same array, so ``vs_input`` is
skipped rather than recomputed.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CAMERAS,
    CONFIG,
    METRIC_KEYS,
    ROOT,
    frame_metrics,
    open_memmap,
    summarize_rows,
    write_json,
)

CRF_LEVELS: list[int] = CONFIG["video"]["crf_levels"]
SOURCES = ["raw", "jpeg100", "jpeg80"]


def video_path(dataset: Path, camera: str) -> Path:
    matches = sorted((dataset / "videos" / f"observation.images.{camera}").rglob("*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one mp4 for {camera} in {dataset}, found {matches}")
    return matches[0]


def decode_frames(path: Path):
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def probe(path: Path) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,pix_fmt,width,height,nb_frames,bit_rate,r_frame_rate",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        return json.loads(out.stdout).get("streams", [{}])[0]
    except Exception as exc:  # ffprobe is a convenience, not a dependency
        return {"error": str(exc)}


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def save_worst_frame_figure(reference, candidate, output: Path, title: str) -> None:
    abs_diff = np.abs(candidate.astype(np.int16) - reference.astype(np.int16)).astype(np.uint8)
    amplified = np.clip(abs_diff.astype(np.uint16) * 4, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(abs_diff.max(axis=2), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    panel = np.concatenate([reference, candidate, amplified, heat], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    width = reference.shape[1]
    for i, label in enumerate(["raw", "decoded", "|diff| x4", "max-channel heatmap"]):
        cv2.rectangle(panel_bgr, (i * width, 0), ((i + 1) * width, 40), (0, 0, 0), -1)
        cv2.putText(panel_bgr, label, (i * width + 10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(panel_bgr, (0, panel_bgr.shape[0] - 34), (panel_bgr.shape[1], panel_bgr.shape[0]),
                  (0, 0, 0), -1)
    cv2.putText(panel_bgr, title, (10, panel_bgr.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), panel_bgr)


def main() -> None:
    results = ROOT / "results"
    figures = ROOT / "figures"
    duration_s = json.loads((results / "episode_audit.json").read_text())["frame_count"] / CONFIG["fps"]
    frames_expected = json.loads((results / "episode_audit.json").read_text())["frame_count"]

    references = {
        source: {camera: open_memmap(source, camera, "r") for camera in CAMERAS}
        for source in SOURCES
    }

    rows: list[dict] = []
    entries: list[dict] = []

    for source in SOURCES:
        for crf in CRF_LEVELS:
            dataset = ROOT / "lerobot" / f"{source}_crf{crf}"
            if not dataset.is_dir():
                print(f"missing dataset {dataset}, skipping")
                continue
            entry = {
                "source": source,
                "crf": crf,
                "dataset": str(dataset),
                "dataset_bytes": directory_bytes(dataset),
                "video_bytes": directory_bytes(dataset / "videos"),
                "parquet_bytes": directory_bytes(dataset / "data"),
                "cameras": {},
            }
            for camera in CAMERAS:
                path = video_path(dataset, camera)
                size = path.stat().st_size
                truth = references["raw"][camera]
                encoder_input = references[source][camera]
                worst = None
                decoded_count = 0
                for index, frame in enumerate(decode_frames(path)):
                    if index >= frames_expected:
                        break
                    metrics_raw = frame_metrics(truth[index], frame)
                    row = {
                        "source": source,
                        "crf": crf,
                        "camera": camera,
                        "frame_index": index,
                        "reference": "raw",
                        **metrics_raw,
                    }
                    rows.append(row)
                    if source != "raw":
                        rows.append({
                            "source": source,
                            "crf": crf,
                            "camera": camera,
                            "frame_index": index,
                            "reference": "encoder_input",
                            **frame_metrics(encoder_input[index], frame),
                        })
                    if worst is None or metrics_raw["ssim"] < worst[0]:
                        worst = (metrics_raw["ssim"], index, frame.copy())
                    decoded_count += 1
                if decoded_count != frames_expected:
                    raise RuntimeError(
                        f"{dataset}/{camera}: decoded {decoded_count} frames, "
                        f"expected {frames_expected}"
                    )
                subset = [r for r in rows if r["source"] == source and r["crf"] == crf
                          and r["camera"] == camera and r["reference"] == "raw"]
                entry["cameras"][camera] = {
                    "video_path": str(path),
                    "video_bytes": size,
                    "bitrate_mbps": size * 8 / duration_s / 1e6,
                    "bytes_per_frame": size / frames_expected,
                    "probe": probe(path),
                    "vs_raw": summarize_rows(subset, METRIC_KEYS),
                }
                if source != "raw":
                    subset_input = [r for r in rows if r["source"] == source and r["crf"] == crf
                                    and r["camera"] == camera and r["reference"] == "encoder_input"]
                    entry["cameras"][camera]["vs_input"] = summarize_rows(subset_input, METRIC_KEYS)
                if camera == "top":
                    ssim, index, frame = worst
                    save_worst_frame_figure(
                        np.asarray(truth[index]),
                        frame,
                        figures / f"worst_{source}_crf{crf}_top.png",
                        f"{source} + h264 crf {crf} | camera top | worst SSIM frame {index} "
                        f"= {ssim:.6f} (reference: raw)",
                    )
                print(f"  {source} crf{crf} {camera}: {size/2**20:.2f} MiB, "
                      f"PSNR {entry['cameras'][camera]['vs_raw']['psnr_db']['mean']:.3f} dB, "
                      f"SSIM {entry['cameras'][camera]['vs_raw']['ssim']['mean']:.6f}", flush=True)
            total_video = sum(c["video_bytes"] for c in entry["cameras"].values())
            entry["video_bytes_sum_cameras"] = total_video
            entry["bitrate_mbps_total"] = total_video * 8 / duration_s / 1e6
            entries.append(entry)
            print(f"{source} crf{crf}: dataset {entry['dataset_bytes']/2**20:.2f} MiB, "
                  f"video {total_video/2**20:.2f} MiB", flush=True)

    fieldnames = ["source", "crf", "camera", "frame_index", "reference", *METRIC_KEYS]
    with (results / "per_frame_video_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_json(results / "video_summary.json", {
        "frames": frames_expected,
        "duration_s": duration_s,
        "encoder": CONFIG["video"],
        "entries": entries,
    })
    print("stage 4 complete")


if __name__ == "__main__":
    main()
