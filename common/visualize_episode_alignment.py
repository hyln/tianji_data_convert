#!/usr/bin/env python3
"""Render three camera views above synchronized left/right gripper plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq


VIDEO_KEYS = ("head", "hik_camera_6", "hik_camera_7")
VIDEO_FEATURES = tuple(f"observation.images.{key}" for key in VIDEO_KEYS)
FPS = 30
PANEL_W, VIDEO_H, PLOT_H = 256, 256, 320
OUT_W, OUT_H = PANEL_W * 3, VIDEO_H + PLOT_H


def episode_row(root: Path, episode: int) -> dict:
    for path in sorted(root.glob("meta/episodes/chunk-*/*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "=", episode)])
        if table.num_rows:
            return table.to_pylist()[0]
    raise ValueError(f"episode {episode} not found")


def episode_data(root: Path, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = pads.dataset(str(root / "data"), format="parquet", partitioning=None)
    table = dataset.to_table(
        columns=["timestamp", "frame_index", "observation.state", "action"],
        filter=pc.field("episode_index") == episode,
    ).sort_by("frame_index")
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    return timestamps, state, action


def open_episode_videos(root: Path, meta: dict) -> list[cv2.VideoCapture]:
    captures = []
    for feature in VIDEO_FEATURES:
        chunk = int(meta[f"videos/{feature}/chunk_index"])
        file = int(meta[f"videos/{feature}/file_index"])
        start_s = float(meta[f"videos/{feature}/from_timestamp"])
        path = root / "videos" / feature / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(start_s * FPS))
        captures.append(capture)
    return captures


def curve_points(values: np.ndarray, x0: int, y0: int, width: int, height: int) -> np.ndarray:
    x = np.linspace(x0, x0 + width - 1, len(values)).astype(np.int32)
    y = np.rint(y0 + (1.0 - np.clip(values, 0.0, 1.0)) * (height - 1)).astype(np.int32)
    return np.column_stack((x, y)).reshape((-1, 1, 2))


def plot_background(
    timestamps: np.ndarray, state: np.ndarray, action: np.ndarray, episode: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    canvas = np.full((PLOT_H, OUT_W, 3), (24, 27, 32), dtype=np.uint8)
    cv2.putText(
        canvas, f"Episode {episode} | gripper alignment (0=open, 1=closed)",
        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA,
    )
    cv2.line(canvas, (495, 42), (515, 42), (0, 180, 255), 3)
    cv2.putText(canvas, "command", (520, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1)
    cv2.line(canvas, (640, 42), (660, 42), (255, 210, 40), 3)
    cv2.putText(canvas, "feedback", (665, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 210, 40), 1)

    boxes = [(45, 75, 320, 200), (425, 75, 320, 200)]
    for side, (x, y, w, h), state_col, action_col in zip(
        ("LEFT GRIPPER", "RIGHT GRIPPER"), boxes, (20, 21), (14, 15), strict=True
    ):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 95, 105), 1)
        for value in (0.0, 0.5, 1.0):
            yy = round(y + (1.0 - value) * h)
            cv2.line(canvas, (x, yy), (x + w, yy), (50, 54, 62), 1)
            cv2.putText(canvas, f"{value:.1f}", (x - 37, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 185), 1)
        cv2.putText(canvas, side, (x + 84, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [curve_points(action[:, action_col], x, y, w, h)], False, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [curve_points(state[:, state_col], x, y, w, h)], False, (255, 210, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, "0s", (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 185), 1)
        cv2.putText(canvas, f"{timestamps[-1]:.1f}s", (x + w - 45, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 185), 1)
    return canvas, boxes


def start_writer(output: Path) -> subprocess.Popen:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
        "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render(root: Path, output: Path, episode: int) -> None:
    meta = episode_row(root, episode)
    timestamps, state, action = episode_data(root, episode)
    if len(timestamps) != int(meta["length"]):
        raise RuntimeError(f"metadata length={meta['length']} but parquet rows={len(timestamps)}")
    captures = open_episode_videos(root, meta)
    plot_base, boxes = plot_background(timestamps, state, action, episode)
    writer = start_writer(output)
    assert writer.stdin is not None
    try:
        for index, timestamp in enumerate(timestamps):
            camera_frames = []
            for name, capture in zip(VIDEO_KEYS, captures, strict=True):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"{name}: failed at episode frame {index}")
                frame = cv2.resize(frame, (PANEL_W, VIDEO_H), interpolation=cv2.INTER_AREA)
                cv2.rectangle(frame, (0, 0), (PANEL_W, 27), (0, 0, 0), -1)
                cv2.putText(frame, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                camera_frames.append(frame)
            top = np.hstack(camera_frames)
            plot = plot_base.copy()
            for x, y, w, h in boxes:
                marker_x = x + round(index * (w - 1) / max(1, len(timestamps) - 1))
                cv2.line(plot, (marker_x, y), (marker_x, y + h), (80, 255, 100), 2)
            cv2.putText(
                plot, f"frame {index:04d}/{len(timestamps)-1:04d}   t={timestamp:6.2f}s",
                (260, 307), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 100), 1, cv2.LINE_AA,
            )
            writer.stdin.write(np.vstack((top, plot)).tobytes())
    finally:
        for capture in captures:
            capture.release()
        writer.stdin.close()
    code = writer.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited with {code}")
    summary = {
        "dataset": str(root), "episode": episode, "frames": len(timestamps),
        "duration_s": float(timestamps[-1]), "output": str(output),
        "left_action_range": [float(action[:, 14].min()), float(action[:, 14].max())],
        "right_action_range": [float(action[:, 15].min()), float(action[:, 15].max())],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode", type=int, default=85)
    args = parser.parse_args()
    render(args.dataset.resolve(), args.output.resolve(), args.episode)


if __name__ == "__main__":
    main()
