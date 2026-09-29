#!/usr/bin/env python3
"""Render three cameras above synchronized 14-joint command-feedback errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from visualize_episode_alignment import (
    FPS, OUT_H, OUT_W, PANEL_W, PLOT_H, VIDEO_H, VIDEO_KEYS,
    episode_data, episode_row, open_episode_videos,
)


COLORS = (
    (80, 180, 255), (255, 150, 70), (80, 220, 120), (220, 100, 220),
    (60, 230, 230), (230, 170, 100), (180, 180, 180),
)


def points(values: np.ndarray, x: int, y: int, w: int, h: int, limit: float) -> np.ndarray:
    xx = np.linspace(x, x + w - 1, len(values)).astype(np.int32)
    yy = np.rint(y + h / 2 - np.clip(values, -limit, limit) / limit * (h / 2 - 2)).astype(np.int32)
    return np.column_stack((xx, yy)).reshape((-1, 1, 2))


def plot_background(error_deg: np.ndarray, episode: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]], float]:
    canvas = np.full((PLOT_H, OUT_W, 3), (24, 27, 32), dtype=np.uint8)
    robust = float(np.percentile(np.abs(error_deg), 99))
    limit = max(0.5, np.ceil(robust * 2.0) / 2.0)
    cv2.putText(
        canvas, f"Episode {episode} | joint tracking error = command - feedback (deg)",
        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA,
    )
    boxes = [(45, 72, 320, 195), (425, 72, 320, 195)]
    for arm, (x, y, w, h), offset in zip(("LEFT ARM", "RIGHT ARM"), boxes, (0, 7), strict=True):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 95, 105), 1)
        for value in (-limit, 0.0, limit):
            yy = round(y + h / 2 - value / limit * h / 2)
            cv2.line(canvas, (x, yy), (x + w, yy), (52, 56, 64), 1)
            cv2.putText(canvas, f"{value:+.1f}", (x - 42, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 175, 185), 1)
        cv2.putText(canvas, arm, (x + 115, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 230, 230), 1, cv2.LINE_AA)
        for joint in range(7):
            cv2.polylines(
                canvas, [points(error_deg[:, offset + joint], x, y, w, h, limit)],
                False, COLORS[joint], 1, cv2.LINE_AA,
            )
        cv2.putText(canvas, "0s", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 175, 185), 1)
    for joint, color in enumerate(COLORS, 1):
        lx = 105 + (joint - 1) * 82
        cv2.line(canvas, (lx, 45), (lx + 18, 45), color, 2)
        cv2.putText(canvas, f"J{joint}", (lx + 22, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    return canvas, boxes, limit


def writer(output: Path) -> subprocess.Popen:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
        "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)


def render(root: Path, output: Path, episode: int) -> None:
    meta = episode_row(root, episode)
    timestamps, state, action = episode_data(root, episode)
    error_deg = np.rad2deg(action[:, :14] - state[:, :14])
    captures = open_episode_videos(root, meta)
    base, boxes, limit = plot_background(error_deg, episode)
    process = writer(output)
    assert process.stdin is not None
    try:
        for index, timestamp in enumerate(timestamps):
            views = []
            for name, capture in zip(VIDEO_KEYS, captures, strict=True):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"{name}: failed at frame {index}")
                frame = cv2.resize(frame, (PANEL_W, VIDEO_H), interpolation=cv2.INTER_AREA)
                cv2.rectangle(frame, (0, 0), (PANEL_W, 27), (0, 0, 0), -1)
                cv2.putText(frame, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                views.append(frame)
            plot = base.copy()
            for arm_index, (x, y, w, h) in enumerate(boxes):
                marker_x = x + round(index * (w - 1) / max(1, len(timestamps) - 1))
                cv2.line(plot, (marker_x, y), (marker_x, y + h), (80, 255, 100), 2)
                current = error_deg[index, arm_index * 7:(arm_index + 1) * 7]
                cv2.putText(
                    plot, f"now max|err|={np.max(np.abs(current)):.2f} deg",
                    (x + 72, 304), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 100), 1,
                )
            process.stdin.write(np.vstack((np.hstack(views), plot)).tobytes())
    finally:
        for capture in captures:
            capture.release()
        process.stdin.close()
    if process.wait():
        raise RuntimeError("ffmpeg failed")

    rms = np.sqrt(np.mean(np.square(error_deg), axis=0))
    maximum = np.max(np.abs(error_deg), axis=0)
    summary = {
        "dataset": str(root), "episode": episode, "frames": len(timestamps),
        "duration_s": float(timestamps[-1]), "plot_limit_deg": limit,
        "joint_rms_error_deg": {f"{arm}{j}": float(rms[offset + j - 1]) for arm, offset in (("L", 0), ("R", 7)) for j in range(1, 8)},
        "joint_max_abs_error_deg": {f"{arm}{j}": float(maximum[offset + j - 1]) for arm, offset in (("L", 0), ("R", 7)) for j in range(1, 8)},
        "output": str(output),
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
