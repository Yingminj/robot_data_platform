#!/usr/bin/env python3
"""Stage 2: rewrite the raw bag as a JPEG recording of the same episode.

``express_mcap`` is JPEG-80 but it is a *different session*, so it cannot answer
"what did quality 80 cost us on this episode".  This script produces the missing
counterfactual: the same bag, same timestamps, same joint data, with
``/quad_tile`` (sensor_msgs/Image) replaced by ``/quad_tile/compressed``
(sensor_msgs/CompressedImage) carrying the JPEG the RealSense node would have
emitted.  Everything else is copied through as unmodified CDR bytes, so the
converter sees an identical control stream and picks identical rows.

Run it once per quality level under test.  ``--quality 80`` is the production
setting; ``--quality 100`` is the "compress at record time but barely" arm,
which separates "JPEG at all" from "JPEG at 80" in the downstream numbers.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/kewei/YING/robot_data_platform/tool")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_data.align.bag_io import open_bag_reader  # noqa: E402
from robot_data.profiles import load_profile  # noqa: E402
from robot_data.ros.media import decode_image  # noqa: E402

from common import CONFIG, ROOT, encode_jpeg, write_json  # noqa: E402

SOURCE_TOPIC = "/quad_tile"
TARGET_TOPIC = "/quad_tile/compressed"


def build_typestore(profile):
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    for typename, definition in (profile.message_definitions or {}).items():
        if typename not in typestore.types:
            typestore.register(get_types_from_msg(definition, typename))
    return typestore


def main() -> None:
    from rosbags.rosbag2 import StoragePlugin, Writer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality", type=int, default=CONFIG["jpeg"]["production_quality"],
        help="JPEG quality to re-encode the mosaic at (default: the production value)",
    )
    args = parser.parse_args()
    quality = args.quality

    source = Path(CONFIG["source_raw_bag"])
    destination = ROOT / "bags" / f"express_jpeg{quality}"
    if destination.exists():
        shutil.rmtree(destination)

    profile = load_profile(CONFIG["profile"])
    typestore = build_typestore(profile)
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]

    reader = open_bag_reader(source, profile)
    reader.open()
    started = time.perf_counter()
    counts: dict[str, int] = {}
    jpeg_bytes: list[int] = []
    raw_bytes: list[int] = []
    try:
        writer = Writer(destination, version=9, storage_plugin=StoragePlugin.MCAP)
        writer.open()
        try:
            outgoing = {}
            for connection in reader.connections:
                if connection.topic == SOURCE_TOPIC:
                    outgoing[connection.id] = writer.add_connection(
                        TARGET_TOPIC,
                        "sensor_msgs/msg/CompressedImage",
                        typestore=typestore,
                    )
                else:
                    outgoing[connection.id] = writer.add_connection(
                        connection.topic,
                        connection.msgtype,
                        typestore=typestore,
                    )

            for connection, timestamp, rawdata in reader.messages():
                counts[connection.topic] = counts.get(connection.topic, 0) + 1
                if connection.topic != SOURCE_TOPIC:
                    writer.write(outgoing[connection.id], timestamp, rawdata)
                    continue
                message = reader.deserialize(rawdata, connection.msgtype)
                raw_bytes.append(len(bytes(message.data)))
                mosaic = decode_image(message, connection.msgtype)
                buffer = encode_jpeg(mosaic, quality)
                jpeg_bytes.append(len(buffer))
                compressed = CompressedImage(
                    header=message.header,
                    format="jpeg",
                    data=np.frombuffer(buffer, dtype=np.uint8),
                )
                writer.write(
                    outgoing[connection.id],
                    timestamp,
                    typestore.serialize_cdr(compressed, "sensor_msgs/msg/CompressedImage"),
                )
                if len(jpeg_bytes) % 200 == 0:
                    rate = len(jpeg_bytes) / (time.perf_counter() - started)
                    print(f"  {len(jpeg_bytes)} frames re-encoded ({rate:.1f} fps)", flush=True)
        finally:
            writer.close()
    finally:
        reader.close()

    source_size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    destination_size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    summary = {
        "source": str(source),
        "destination": str(destination),
        "jpeg_quality": quality,
        "source_topic": SOURCE_TOPIC,
        "target_topic": TARGET_TOPIC,
        "message_counts": counts,
        "image_frames": len(jpeg_bytes),
        "raw_image_bytes_mean": float(np.mean(raw_bytes)),
        "jpeg_bytes_mean": float(np.mean(jpeg_bytes)),
        "jpeg_bytes_total": int(np.sum(jpeg_bytes)),
        "source_bag_bytes": source_size,
        "jpeg_bag_bytes": destination_size,
        "bag_shrink_factor": source_size / destination_size,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(ROOT / "results" / f"jpeg{quality}_bag.json", summary)
    print(
        f"wrote {destination} : {destination_size / 2**20:.1f} MiB "
        f"(source {source_size / 2**20:.1f} MiB, {summary['bag_shrink_factor']:.1f}x smaller)"
    )


if __name__ == "__main__":
    main()
