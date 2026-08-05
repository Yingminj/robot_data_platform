"""Second pass over a bag: decode only the frames the grid actually selected."""

from __future__ import annotations

import numpy as np

from robot_data.align.bag_io import RawBag, open_bag_reader
from robot_data.align.config import AlignmentConfig
from robot_data.errors import ConversionError
from robot_data.progress import connection_total, tracked
from robot_data.ros.media import decode_depth, decode_image, resize_letterbox


def decode_selected_media(
    raw: RawBag,
    cfg: AlignmentConfig,
    image_indices: dict[str, np.ndarray],
    depth_indices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    profile = raw.profile
    # Several cameras can be tiles of one stitched topic, so accumulate rather
    # than assigning: a plain dict comprehension keyed by topic would keep only
    # the last camera's frame selection and silently drop the others'.
    selected_by_topic: dict[str, set[int]] = {}
    for name, indices in image_indices.items():
        selected_by_topic.setdefault(profile.cameras[name], set()).update(
            np.unique(indices).tolist()
        )
    if cfg.include_depth:
        selected_by_topic.update(
            {
                profile.depths[name]: set(np.unique(indices).tolist())
                for name, indices in depth_indices.items()
            }
        )
    counters = {topic: 0 for topic in selected_by_topic}
    decoded: dict[str, dict[int, np.ndarray]] = {topic: {} for topic in selected_by_topic}
    camera_topics = set(profile.cameras.values())

    reader = open_bag_reader(raw.bag_dir, raw.profile)
    reader.open()
    try:
        connections = [c for c in reader.connections if c.topic in selected_by_topic]
        for connection, _, rawdata in tracked(
            reader.messages(connections=connections),
            f"decode {raw.bag_dir.name}",
            connection_total(connections),
        ):
            topic = connection.topic
            index = counters[topic]
            counters[topic] += 1
            if index not in selected_by_topic[topic]:
                continue
            try:
                message = reader.deserialize(rawdata, connection.msgtype)
                if topic in camera_topics:
                    # Kept whole here; tiles are cut per camera below, so a
                    # mosaic shared by three cameras is still decoded once.
                    value = decode_image(message, connection.msgtype)
                else:
                    value = decode_depth(message, connection.msgtype)
                    value = resize_letterbox(
                        value, cfg.image_height, cfg.image_width, is_depth=True
                    )
                decoded[topic][index] = value
            except Exception as exc:
                raise ConversionError(
                    f"Failed to decode selected {topic} message {index}: {exc}"
                ) from exc
    finally:
        reader.close()

    images: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    for name, indices in image_indices.items():
        topic = profile.cameras[name]
        missing = sorted(set(indices.tolist()) - set(decoded[topic]))
        if missing:
            raise ConversionError(f"Selected RGB frames missing for {name}: {missing[:10]}")
        tile = profile.camera_tiles.get(name)
        frames = []
        for i in indices:
            frame = decoded[topic][int(i)]
            if tile is not None:
                frame = tile.apply(frame)
            frames.append(resize_letterbox(frame, cfg.image_height, cfg.image_width))
        images[name] = np.stack(frames).astype(np.uint8, copy=False)
    for name, indices in depth_indices.items():
        topic = profile.depths[name]
        missing = sorted(set(indices.tolist()) - set(decoded[topic]))
        if missing:
            raise ConversionError(f"Selected depth frames missing for {name}: {missing[:10]}")
        depths[name] = np.stack([decoded[topic][int(i)] for i in indices]).astype(
            np.float32, copy=False
        )
    return images, depths
