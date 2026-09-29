"""Detect left-wrist command targets beyond the calibrated weighted envelope."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


FILTER_NAME = "left_wrist_weighted_expectation_over_118"


@dataclass(frozen=True)
class WeightedExpectationLimitConfig:
    j6_weight: float = 1.025
    threshold_deg: float = 118.0
    merge_gap_s: float = 0.12

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_CONFIG = WeightedExpectationLimitConfig()


def _read_csv(path: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    timestamps = np.array(
        [int(row["header_stamp_ns"] or row["receive_timestamp_ns"]) for row in rows],
        dtype=np.int64,
    )
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"non-increasing timestamps: {path}")
    return timestamps, rows


def _zero_timestamp(topic_dir: Path, fallback: int) -> tuple[int, str]:
    candidates = [topic_dir / "perception__sensors__hik_camera_8__image__frames.csv"]
    candidates.extend(sorted(topic_dir.glob("*__frames.csv")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            return int(_read_csv(path)[0][0]), path.name
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return int(fallback), "command_fallback"


def _events(
    times: np.ndarray,
    score: np.ndarray,
    threshold_deg: float,
    merge_gap_s: float,
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(score > threshold_deg)
    if not len(indices):
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for raw_index in indices[1:]:
        index = int(raw_index)
        if times[index] - times[groups[-1][-1]] <= merge_gap_s:
            groups[-1].append(index)
        else:
            groups.append([index])

    result = []
    for group in groups:
        start_index = group[0]
        end_index = group[-1]
        following = min(end_index + 1, len(times) - 1)
        start_s = float(times[start_index])
        end_s = max(start_s, float(times[following]))
        peak_index = max(group, key=lambda index: float(score[index]))
        result.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "sample_count": len(group),
                "peak_time_s": float(times[peak_index]),
                "peak_deg": float(score[peak_index]),
            }
        )
    return result


def scan_session(
    session: Path | str,
    config: WeightedExpectationLimitConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return a serializable decision for one compressed recording session."""

    session = Path(session)
    result: dict[str, Any] = {
        "session": session.name,
        "source": str(session),
        "filter": FILTER_NAME,
        "status": "failed",
        "rejected": False,
        "events": [],
    }
    try:
        topic_dir = session / "topic_csv"
        command_path = topic_dir / "motion_control__record__joint_activate_cmd.csv"
        timestamps, rows = _read_csv(command_path)
        names = json.loads(rows[0]["name"])
        j6_index = names.index("Joint6_L")
        j7_index = names.index("Joint7_L")
        positions = np.array([json.loads(row["position"]) for row in rows], dtype=float)
        if not np.isfinite(positions[:, [j6_index, j7_index]]).all():
            raise ValueError(f"non-finite command joints: {command_path}")
        j6_deg = np.rad2deg(positions[:, j6_index])
        j7_deg = np.rad2deg(positions[:, j7_index])
        score_deg = config.j6_weight * np.abs(j6_deg) + np.abs(j7_deg)
        zero_ns, zero_source = _zero_timestamp(topic_dir, int(timestamps[0]))
        times = (timestamps - zero_ns) / 1e9
        peak_index = int(np.argmax(score_deg))
        events = _events(
            times, score_deg, config.threshold_deg, config.merge_gap_s
        )
        result.update(
            status="ok",
            rejected=bool(events),
            reason=(
                "left wrist weighted command target exceeds 118 degrees"
                if events
                else None
            ),
            zero_timestamp_ns=int(zero_ns),
            zero_timestamp_source=zero_source,
            sample_count=len(rows),
            exceeding_sample_count=int(np.count_nonzero(score_deg > config.threshold_deg)),
            peak_time_s=float(times[peak_index]),
            peak_deg=float(score_deg[peak_index]),
            peak_j6_deg=float(j6_deg[peak_index]),
            peak_j7_deg=float(j7_deg[peak_index]),
            events=events,
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result
