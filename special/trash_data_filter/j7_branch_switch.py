"""Detect left-wrist J7 branch switches near the J6/J7 coupling boundary.

The detector is read-only.  A matching session is reported as rejected, but
this module never moves or deletes the source recording.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


FILTER_NAME = "left_wrist_j7_branch_switch"


@dataclass(frozen=True)
class J7BranchSwitchConfig:
    window_s: float = 1.0
    stride_s: float = 0.1
    grid_dt_s: float = 0.02
    enable_head_guard_s: float = 0.3
    enable_tail_guard_s: float = 0.05
    max_source_gap_s: float = 0.1
    j6_limit_deg: float = 58.0
    min_j6_limit_fraction: float = 0.8
    max_j6_range_deg: float = 0.6
    min_j7_net_change_deg: float = 55.0
    min_j7_monotonic_ratio: float = 0.8
    coupling_j6_weight: float = 1.025
    coupling_limit_deg: float = 107.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_CONFIG = J7BranchSwitchConfig()


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


def _commands(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps, rows = _read_csv(path)
    names = json.loads(rows[0]["name"])
    j6_index = names.index("Joint6_L")
    j7_index = names.index("Joint7_L")
    joints_deg = np.rad2deg(
        np.array(
            [
                [json.loads(row["position"])[j6_index], json.loads(row["position"])[j7_index]]
                for row in rows
            ],
            dtype=np.float64,
        )
    )
    if not np.isfinite(joints_deg).all():
        raise ValueError(f"non-finite command joints: {path}")
    return timestamps, joints_deg


def _base_timestamp(topic_dir: Path, fallback: int) -> tuple[int, str]:
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


def _enabled_spans(
    timestamps: np.ndarray,
    rows: list[dict[str, str]],
    base_timestamp_ns: int,
    command_end_s: float,
) -> list[tuple[float, float]]:
    times = (timestamps - base_timestamp_ns) / 1e9
    enabled = np.array(
        [
            row["left_enable"].lower() == "true" and row["left_arm_state"] == "2"
            for row in rows
        ],
        dtype=bool,
    )
    spans: list[tuple[float, float]] = []
    starts = np.flatnonzero(enabled & ~np.r_[False, enabled[:-1]])
    for start in starts:
        disabled = np.flatnonzero(~enabled[start + 1 :])
        stop = times[start + 1 + disabled[0]] if len(disabled) else command_end_s
        spans.append((float(times[start]), float(stop)))
    return spans


def _merge_windows(windows: list[dict[str, Any]], stride_s: float) -> list[dict[str, Any]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: float(item["start_s"]))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for item in ordered[1:]:
        if float(item["start_s"]) <= float(groups[-1][-1]["end_s"]) + stride_s + 1e-9:
            groups[-1].append(item)
        else:
            groups.append([item])

    events: list[dict[str, Any]] = []
    for group in groups:
        best = max(group, key=lambda item: float(item["j7_abs_net_deg"]))
        events.append(
            {
                **best,
                "event_start_s": float(group[0]["start_s"]),
                "event_end_s": float(group[-1]["end_s"]),
                "matching_window_count": len(group),
            }
        )
    return events


def scan_session(
    session: Path | str,
    config: J7BranchSwitchConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return a serializable rejection decision for one compressed session."""

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
        command_ns, command_joints = _commands(
            topic_dir / "motion_control__record__joint_activate_cmd.csv"
        )
        base_ns, base_source = _base_timestamp(topic_dir, int(command_ns[0]))
        command_times = (command_ns - base_ns) / 1e9
        state_ns, state_rows = _read_csv(topic_dir / "motion_control__states__control.csv")
        spans = _enabled_spans(state_ns, state_rows, base_ns, float(command_times[-1]))

        width = int(round(config.window_s / config.grid_dt_s)) + 1
        stride = max(1, int(round(config.stride_s / config.grid_dt_s)))
        windows_checked = 0
        matches: list[dict[str, Any]] = []
        max_j7_abs_net_deg = 0.0
        max_coupling_score_deg = float(
            np.max(
                config.coupling_j6_weight * np.abs(command_joints[:, 0])
                + np.abs(command_joints[:, 1])
            )
        )

        for span_start, span_stop in spans:
            lo = max(span_start + config.enable_head_guard_s, float(command_times[0]))
            hi = min(span_stop - config.enable_tail_guard_s, float(command_times[-1]))
            if hi - lo < config.window_s:
                continue
            grid = np.arange(lo, hi + config.grid_dt_s * 0.1, config.grid_dt_s)
            previous = np.searchsorted(command_times, grid, side="right") - 1
            following = np.minimum(previous + 1, len(command_times) - 1)
            safe_previous = np.maximum(previous, 0)
            valid = (previous >= 0) & (
                (command_times[following] - command_times[safe_previous])
                <= config.max_source_gap_s
            )
            j6 = np.interp(grid, command_times, command_joints[:, 0])
            j7 = np.interp(grid, command_times, command_joints[:, 1])

            for start in range(0, len(grid) - width + 1, stride):
                stop = start + width
                if not np.all(valid[start:stop]):
                    continue
                windows_checked += 1
                window_j6 = j6[start:stop]
                window_j7 = j7[start:stop]
                j7_path = float(np.sum(np.abs(np.diff(window_j7))))
                j7_net = float(window_j7[-1] - window_j7[0])
                j7_abs_net = abs(j7_net)
                max_j7_abs_net_deg = max(max_j7_abs_net_deg, j7_abs_net)
                monotonic_ratio = j7_abs_net / j7_path if j7_path > 1e-9 else 0.0
                j6_limit_fraction = float(
                    np.mean(np.abs(window_j6) >= config.j6_limit_deg)
                )
                j6_range = float(np.ptp(window_j6))
                if not (
                    j6_limit_fraction >= config.min_j6_limit_fraction
                    and j6_range <= config.max_j6_range_deg
                    and j7_abs_net >= config.min_j7_net_change_deg
                    and monotonic_ratio >= config.min_j7_monotonic_ratio
                ):
                    continue
                coupling_score = (
                    config.coupling_j6_weight * np.abs(window_j6) + np.abs(window_j7)
                )
                matches.append(
                    {
                        "start_s": float(grid[start]),
                        "end_s": float(grid[stop - 1]),
                        "j6_mean_deg": float(np.mean(window_j6)),
                        "j6_range_deg": j6_range,
                        "j6_limit_fraction": j6_limit_fraction,
                        "j7_start_deg": float(window_j7[0]),
                        "j7_end_deg": float(window_j7[-1]),
                        "j7_net_deg": j7_net,
                        "j7_abs_net_deg": j7_abs_net,
                        "j7_path_deg": j7_path,
                        "j7_monotonic_ratio": monotonic_ratio,
                        "max_coupling_score_deg": float(np.max(coupling_score)),
                    }
                )

        events = _merge_windows(matches, config.stride_s)
        result.update(
            status="ok",
            rejected=bool(events),
            reason=(
                "left wrist J7 branch switch near coupling boundary" if events else None
            ),
            zero_timestamp_ns=int(base_ns),
            zero_timestamp_source=base_source,
            active_span_count=len(spans),
            windows_checked=windows_checked,
            max_j7_abs_net_deg=max_j7_abs_net_deg,
            max_coupling_score_deg=max_coupling_score_deg,
            coupling_margin_deg=config.coupling_limit_deg - max_coupling_score_deg,
            events=events,
        )
    except Exception as error:  # Keep batch reports useful when one session is malformed.
        result["error"] = f"{type(error).__name__}: {error}"
    return result
