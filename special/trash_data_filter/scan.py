#!/usr/bin/env python3
"""Scan compressed Tianji sessions and report trash-data filter decisions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

try:
    from .j7_branch_switch import (
        DEFAULT_CONFIG as J7_CONFIG,
        FILTER_NAME as J7_FILTER,
        scan_session as scan_j7_branch_switch,
    )
    from .weighted_expectation_limit import (
        DEFAULT_CONFIG as WEIGHTED_CONFIG,
        FILTER_NAME as WEIGHTED_FILTER,
        scan_session as scan_weighted_expectation_limit,
    )
except ImportError:  # Allow direct execution: python special/trash_data_filter/scan.py ...
    from j7_branch_switch import (  # type: ignore[no-redef]
        DEFAULT_CONFIG as J7_CONFIG,
        FILTER_NAME as J7_FILTER,
        scan_session as scan_j7_branch_switch,
    )
    from weighted_expectation_limit import (  # type: ignore[no-redef]
        DEFAULT_CONFIG as WEIGHTED_CONFIG,
        FILTER_NAME as WEIGHTED_FILTER,
        scan_session as scan_weighted_expectation_limit,
    )


FILTERS = {
    J7_FILTER: (scan_j7_branch_switch, J7_CONFIG),
    WEIGHTED_FILTER: (scan_weighted_expectation_limit, WEIGHTED_CONFIG),
}


def _sessions(source: Path) -> list[Path]:
    return [
        path
        for path in sorted(source.iterdir())
        if path.is_dir()
        and (path / "topic_csv" / "motion_control__record__joint_activate_cmd.csv").exists()
    ]


def archive_rejected_sessions(
    source: Path,
    records: list[dict[str, Any]],
    archive_root: Path,
    *,
    batch_name: str | None = None,
) -> dict[str, Any]:
    """Move rejected sessions out of the source and archive them under Documents."""

    source = source.resolve()
    archive_root = archive_root.resolve()
    batch = (
        archive_root
        / source.name
        / (batch_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    sessions_dir = batch / "sessions"
    rejected = [
        record
        for record in records
        if record["status"] == "ok" and record["rejected"]
    ]
    moves: list[dict[str, str]] = []

    # Resolve every target before changing the filesystem.  This prevents a
    # malformed report from moving paths outside the requested source root.
    planned: list[tuple[Path, Path]] = []
    for record in rejected:
        path = Path(str(record["source"])).resolve()
        if path.parent != source:
            raise ValueError(f"refusing to move path outside source: {path}")
        destination = sessions_dir / path.name
        if not path.is_dir():
            raise FileNotFoundError(path)
        if destination.exists():
            raise FileExistsError(destination)
        planned.append((path, destination))

    if planned:
        sessions_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = batch / "move_manifest.json"
    manifest: dict[str, Any] = {
        "status": "moving",
        "source_root": str(source),
        "archive_root": str(archive_root),
        "batch": str(batch),
        "planned_count": len(planned),
        "moves": moves,
    }
    if planned:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    try:
        for path, destination in planned:
            shutil.move(str(path), str(destination))
            moves.append({"source": str(path), "destination": str(destination)})
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise
    if planned:
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "mode": "archive_move",
        "archive_root": str(archive_root),
        "batch": str(batch) if planned else None,
        "sessions_dir": str(sessions_dir) if planned else None,
        "moved_count": len(moves),
        "moves": moves,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory containing compressed sessions")
    parser.add_argument(
        "--filter",
        choices=tuple(FILTERS),
        default=J7_FILTER,
        help=f"filter rule to apply (default: {J7_FILTER})",
    )
    parser.add_argument("--output", type=Path, help="write the complete JSON report here")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--move",
        action="store_true",
        help="move matches out of source into a reviewable Documents archive",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="archive root (default: ~/Documents/tianji_trash_data)",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    sessions = _sessions(source)
    scanner, config = FILTERS[args.filter]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(scanner, sessions))
    rejected = [record for record in records if record["status"] == "ok" and record["rejected"]]
    failures = [record for record in records if record["status"] != "ok"]
    report = {
        "filter": args.filter,
        "config": config.as_dict(),
        "source": str(source),
        "session_count": len(records),
        "rejected_count": len(rejected),
        "failure_count": len(failures),
        "rejected_session_ids": [record["session"] for record in rejected],
        "records": records,
    }
    if args.move and not failures:
        archive_root = (
            args.archive_dir.resolve()
            if args.archive_dir
            else Path.home() / "Documents" / "tianji_trash_data"
        )
        report["operation"] = archive_rejected_sessions(
            source, records, archive_root
        )
        batch = report["operation"]["batch"]
        if batch:
            batch_path = Path(batch)
            (batch_path / "trash_data_filter_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )
            (batch_path / "candidate_ids.txt").write_text(
                "\n".join(report["rejected_session_ids"]) + "\n"
            )
            (batch_path / "README.md").write_text(
                "# Tianji 异常数据归档\n\n"
                f"原始目录：`{source}`\n\n"
                f"过滤器：`{args.filter}`\n\n"
                f"归档数量：{len(rejected)}\n\n"
                "`sessions/` 保存完整原始session；`move_manifest.json` 保存恢复路径；"
                "`trash_data_filter_report.json` 保存判定指标。\n"
            )
    else:
        report["operation"] = {
            "mode": "dry_run",
            "moved_count": 0,
            "reason": "scan failure" if failures else "--move not specified",
        }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(
        json.dumps(
            {
                "filter": args.filter,
                "session_count": len(records),
                "rejected_count": len(rejected),
                "failure_count": len(failures),
                "rejected_session_ids": report["rejected_session_ids"],
                "operation": report["operation"],
                "output": str(args.output) if args.output else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
