#!/usr/bin/env python3
"""Convert one ROS 2 sqlite bag into compact videos and per-topic CSV files."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


VIDEO_TYPE = "sensor_msgs/msg/CompressedImage"


def safe_name(topic: str) -> str:
    return topic.strip("/").replace("/", "__") or "root"


def value_to_cell(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return json.dumps(list(value), separators=(",", ":"))
    try:
        return json.dumps(list(value), separators=(",", ":"))
    except TypeError:
        return str(value)


def flatten_message(message: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in message.get_fields_and_field_types():
        value = getattr(message, field)
        key = f"{prefix}.{field}" if prefix else field
        if hasattr(value, "get_fields_and_field_types"):
            result.update(flatten_message(value, key))
        else:
            result[key] = value_to_cell(value)
    return result


def header_stamp_ns(message: Any) -> int | str:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return ""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def find_db(bag_dir: Path) -> Path:
    db_files = sorted(bag_dir.glob("*.db3"))
    if len(db_files) != 1:
        raise RuntimeError(f"expected exactly one .db3 in {bag_dir}, found {len(db_files)}")
    return db_files[0]


def video_fps(conn: sqlite3.Connection, topic_id: int) -> float:
    count, first, last = conn.execute(
        "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id=?",
        (topic_id,),
    ).fetchone()
    if count < 2 or last <= first:
        return 30.0
    return float((count - 1) * 1_000_000_000 / (last - first))


def start_encoder(
    output: Path, fps: float, quality: int, preset: str, encoder: str,
    hardware_decode: bool,
) -> subprocess.Popen:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    ]
    if hardware_decode:
        command.extend([
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", "mjpeg_cuvid",
        ])
    command.extend([
        "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", f"{fps:.9f}",
        "-i", "pipe:0", "-an",
    ])
    if encoder == "hevc_nvenc":
        command.extend([
            "-c:v", "hevc_nvenc", "-preset", preset, "-tune", "hq",
            "-rc", "vbr", "-cq", str(quality), "-b:v", "0",
            "-tag:v", "hvc1",
        ])
        if not hardware_decode:
            command.extend(["-pix_fmt", "yuv420p"])
    else:
        command.extend([
            "-c:v", "libx265", "-preset", preset, "-crf", str(quality),
            "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
        ])
    command.extend(["-movflags", "+faststart", str(output)])
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def convert_video_topic(
    db_path: Path,
    temp_dir: Path,
    topic_id: int,
    topic: str,
    type_name: str,
    count: int,
    quality: int,
    preset: str,
    encoder: str,
    hardware_decode: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        fps = video_fps(conn, topic_id)
        name = safe_name(topic)
        video_path = temp_dir / "video" / f"{name}.mp4"
        timestamp_path = temp_dir / "topic_csv" / f"{name}__frames.csv"
        message_class = get_message(type_name)
        process = start_encoder(
            video_path, fps, quality, preset, encoder, hardware_decode,
        )
        assert process.stdin is not None
        rows = conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic_id,),
        )
        with timestamp_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_index", "bag_timestamp_ns", "header_stamp_ns", "format"])
            for frame_index, (timestamp, data) in enumerate(rows):
                message = deserialize_message(data, message_class)
                process.stdin.write(bytes(message.data))
                writer.writerow([frame_index, timestamp, header_stamp_ns(message), message.format])
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {topic}, exit code {return_code}")
        elapsed = time.monotonic() - started
        print(
            f"[video done] {topic}: {count} frames in {elapsed:.2f}s "
            f"({count / elapsed:.1f} fps)", flush=True,
        )
        return {
            "name": topic, "type": type_name, "message_count": count,
            "kind": "video", "fps": fps, "encode_elapsed_s": elapsed,
            "video": str(video_path.relative_to(temp_dir)),
            "timestamps": str(timestamp_path.relative_to(temp_dir)),
        }
    finally:
        conn.close()


def convert(
    bag_dir: Path, output_dir: Path, quality: int, preset: str, encoder: str,
    video_workers: int, hardware_decode: bool,
) -> dict[str, Any]:
    db_path = find_db(bag_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    temp_dir = output_dir.with_name(output_dir.name + ".partial")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    video_dir = temp_dir / "video"
    csv_dir = temp_dir / "topic_csv"
    video_dir.mkdir(parents=True)
    csv_dir.mkdir()

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    topics = conn.execute("SELECT id, name, type FROM topics ORDER BY id").fetchall()
    manifest: dict[str, Any] = {
        "format_version": 1,
        "source_bag": str(bag_dir),
        "source_db_bytes": db_path.stat().st_size,
        "video_codec": f"hevc/{encoder}",
        "video_quality": quality,
        "video_preset": preset,
        "video_workers": video_workers,
        "hardware_decode": hardware_decode,
        "topics": [],
    }
    video_topics: list[tuple[int, str, str, int]] = []
    try:
        for index, (topic_id, topic, type_name) in enumerate(topics, 1):
            name = safe_name(topic)
            message_class = get_message(type_name)
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,)
            ).fetchone()[0]
            print(f"[{index}/{len(topics)}] {topic} ({type_name}, {count} messages)", flush=True)
            entry: dict[str, Any] = {"name": topic, "type": type_name, "message_count": count}
            if type_name == VIDEO_TYPE:
                video_topics.append((topic_id, topic, type_name, count))
                continue
            rows = conn.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (topic_id,),
            )
            csv_path = csv_dir / f"{name}.csv"
            first = rows.fetchone()
            if first is None:
                fields: list[str] = []
                with csv_path.open("w", newline="") as stream:
                    csv.writer(stream).writerow(["bag_timestamp_ns", "header_stamp_ns"])
            else:
                timestamp, data = first
                message = deserialize_message(data, message_class)
                flattened = flatten_message(message)
                fields = list(flattened)
                with csv_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=["bag_timestamp_ns", "header_stamp_ns", *fields],
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerow({
                        "bag_timestamp_ns": timestamp,
                        "header_stamp_ns": header_stamp_ns(message),
                        **flattened,
                    })
                    for timestamp, data in rows:
                        message = deserialize_message(data, message_class)
                        writer.writerow({
                            "bag_timestamp_ns": timestamp,
                            "header_stamp_ns": header_stamp_ns(message),
                            **flatten_message(message),
                        })
            entry.update({
                "kind": "topic_csv", "csv": str(csv_path.relative_to(temp_dir)),
                "fields": fields,
            })
            manifest["topics"].append(entry)
        print(f"encoding {len(video_topics)} video topics with {video_workers} worker(s)", flush=True)
        with ThreadPoolExecutor(max_workers=video_workers) as executor:
            futures = [
                executor.submit(
                    convert_video_topic, db_path, temp_dir, *video_topic,
                    quality, preset, encoder, hardware_decode,
                )
                for video_topic in video_topics
            ]
            for future in as_completed(futures):
                manifest["topics"].append(future.result())
    except Exception:
        conn.close()
        raise
    conn.close()
    manifest_path = temp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_dir, output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS 2 bag directory containing one .db3")
    parser.add_argument("output", type=Path, help="new archive output directory")
    parser.add_argument(
        "--encoder", choices=("libx265", "hevc_nvenc"), default="libx265",
        help="HEVC encoder (default: libx265)",
    )
    parser.add_argument(
        "--quality", "--crf", dest="quality", type=int, default=26,
        help="CRF for libx265 or CQ for hevc_nvenc (default: 26)",
    )
    parser.add_argument(
        "--preset", default=None,
        help="encoder preset (default: medium for libx265, p1 for hevc_nvenc)",
    )
    parser.add_argument(
        "--video-workers", type=int, default=None,
        help="parallel video encoders (default: 1 for libx265, 3 for hevc_nvenc)",
    )
    parser.add_argument(
        "--cpu-decode", action="store_true",
        help="disable NVIDIA MJPEG hardware decoding in hevc_nvenc mode",
    )
    args = parser.parse_args()
    preset = args.preset or ("p1" if args.encoder == "hevc_nvenc" else "medium")
    video_workers = args.video_workers or (3 if args.encoder == "hevc_nvenc" else 1)
    hardware_decode = args.encoder == "hevc_nvenc" and not args.cpu_decode
    manifest = convert(
        args.bag.resolve(), args.output.resolve(), args.quality, preset, args.encoder,
        max(1, video_workers), hardware_decode,
    )
    output_bytes = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    source_bytes = int(manifest["source_db_bytes"])
    print(f"done: {args.output}")
    print(f"size: {source_bytes / 2**30:.3f} GiB -> {output_bytes / 2**30:.3f} GiB "
          f"({output_bytes / source_bytes:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
