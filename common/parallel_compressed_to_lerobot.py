#!/usr/bin/env python3
"""Parallel compressed_records -> one multi-episode LeRobot v3 dataset.

Each session is encoded exactly once into a one-episode shard.  Shards are
created concurrently, then LeRobot's official aggregate_datasets() merges
Parquet metadata and concatenates H.264 streams without re-encoding them.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORKER = Path(__file__).with_name("compressed_session_to_lerobot.py")
CONVERSION_REVISION = "feedback_effort_five_view_v12"
CONVERSION_MODES = ("legacy", "enable_only", "full_rollout_labeled", "feedback_effort_labeled")
DEFAULT_TASK = "<control mode> joint </control mode> tianji teleoperation"


def complete_sessions(source: Path, session_prefix: str | None = None) -> list[Path]:
    sessions = []
    paths = [source] if (source / "manifest.json").exists() else sorted(source.iterdir())
    for path in paths:
        manifest_path = path / "manifest.json"
        if (not path.is_dir() or not manifest_path.exists()
                or (session_prefix is not None and not path.name.startswith(session_prefix))):
            continue
        try:
            if json.loads(manifest_path.read_text()).get("status") == "complete":
                sessions.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    return sessions


def shard_valid(path: Path, image_size: int = 256, *, task: str = DEFAULT_TASK,
                video_crf: int = 18, session: Path | None = None,
                conversion_mode: str = "legacy", source_kind: str = "unspecified",
                empty_scene: bool = False, fk_library=None, fk_config=None, camera_intrinsics=None) -> bool:
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
        report = json.loads((path / "conversion_report.json").read_text())
        from camera_metadata import load_camera_intrinsics, view_sources
        expected_views = view_sources(load_camera_intrinsics(camera_intrinsics, image_size) if camera_intrinsics else {})
        if {key for key in info.get("features", {}) if key.startswith("observation.images.")} != {"observation.images." + name for name in expected_views}:
            return False
        if camera_intrinsics is not None:
            from feedback_state import file_sha256
            if report.get("camera_intrinsics_sha256") != file_sha256(camera_intrinsics):
                return False
        elif report.get("camera_intrinsics_sha256") is not None:
            return False
        if conversion_mode == "feedback_effort_labeled":
            from feedback_state import STATE_NAMES, extra_features, file_sha256, source_fingerprint
            if (report.get("source_kind") != source_kind
                    or report.get("empty_scene") != empty_scene
                    or fk_library is None or fk_config is None
                    or report.get("fk_library_sha256") != file_sha256(fk_library)
                    or report.get("fk_config_sha256") != file_sha256(fk_config)
                    or (session is not None and report.get("source_fingerprint") != source_fingerprint(session))
                    or info.get("features", {}).get("observation.state", {}).get("names") != STATE_NAMES
                    or any(key not in info.get("features", {}) for key in extra_features())):
                return False
        return (
            info.get("total_episodes") == 1
            and int(info.get("total_frames", 0)) > 0
            and report.get("conversion_revision") == CONVERSION_REVISION
            and report.get("image_size", 256) == image_size
            and report.get("task") == task
            and report.get("video_crf") == video_crf
            and report.get("conversion_mode", "legacy") == conversion_mode
            and report.get("resampling_enabled") is False
            and info.get("fps", 0) > 0
            and info.get("fps") == report.get("fps")
            and (session is None or report.get("source") == str(session.resolve()))
            and all(
                info.get("features", {}).get(f"observation.images.{camera}", {}).get("shape")
                == [image_size, image_size, 3]
                for camera in expected_views
            )
            and (
                "observation.state" not in info.get("features", {})
                or info["features"]["observation.state"].get("shape")
                == ([36] if conversion_mode == "feedback_effort_labeled" else
                    [24] if conversion_mode == "full_rollout_labeled" else [22])
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def convert_shard(
    index: int,
    session: Path,
    shard: Path,
    log_path: Path,
    *,
    task: str,
    video_crf: int,
    image_size: int = 256,
    conversion_mode: str = "legacy",
    source_kind: str = "unspecified", empty_scene: bool = False,
    fk_library=None, fk_config=None, camera_intrinsics=None,
) -> dict[str, object]:
    if image_size not in (224, 256, 512):
        raise ValueError("image_size must be 224, 256 or 512")
    started = time.perf_counter()
    repo_id = f"local/tianji_rdc_shard_{index:06d}"
    # Empty camera indices cannot yield synchronized frames. Skip before
    # importing the worker's ML/video stack for every such session.
    empty_cameras = []
    for camera in (8, 6, 7):
        stem = f"perception__sensors__hik_camera_{camera}__image"
        index_path = session / "topic_csv" / f"{stem}__frames.csv"
        if not index_path.exists():
            index_path = session / "topic_csv" / f"{stem}__compressed__frames.csv"
        with index_path.open(newline="") as stream:
            if next(csv.DictReader(stream), None) is None:
                empty_cameras.append(camera)
    if empty_cameras:
        reason = f"no synchronized frames: empty camera indices {empty_cameras}"
        log_path.write_text(f"[skip_invalid] {session.name}: {reason}\n")
        return {
            "index": index, "session": session.name, "status": "skip_invalid",
            "reason": reason, "elapsed_s": round(time.perf_counter() - started, 6),
        }
    marker_path = session / "topic_csv" / "rollout__error_marker.csv"
    if conversion_mode == "enable_only" and marker_path.exists():
        with marker_path.open(newline="") as stream:
            if next(csv.DictReader(stream), None) is not None:
                reason = "session contains rollout Y error marker"
                log_path.write_text(f"[skip_invalid] {session.name}: {reason}\n")
                return {
                    "index": index, "session": session.name, "status": "skip_invalid",
                    "reason": reason, "elapsed_s": round(time.perf_counter() - started, 6),
                }
    if conversion_mode == "enable_only":
        control_path = session / "topic_csv" / "motion_control__states__control.csv"
        with control_path.open(newline="") as stream:
            has_enable = any(
                row.get("left_enable", "").strip().lower() in {"1", "true", "yes"}
                or row.get("right_enable", "").strip().lower() in {"1", "true", "yes"}
                for row in csv.DictReader(stream)
            )
        if not has_enable:
            reason = "session contains no Enable frames"
            log_path.write_text(f"[skip_invalid] {session.name}: {reason}\n")
            return {
                "index": index, "session": session.name, "status": "skip_invalid",
                "reason": reason, "elapsed_s": round(time.perf_counter() - started, 6),
            }
    settings = dict(task=task, video_crf=video_crf, session=session,
                    conversion_mode=conversion_mode, source_kind=source_kind,
                    empty_scene=empty_scene, fk_library=fk_library, fk_config=fk_config)
    settings["camera_intrinsics"] = camera_intrinsics
    if shard_valid(shard, image_size, **settings):
        return {
            "index": index, "session": session.name, "status": "resumed",
            "repo_id": repo_id, "root": str(shard), "elapsed_s": 0.0,
        }
    if shard.exists():
        shutil.rmtree(shard)

    command = [
        sys.executable, str(WORKER), str(session),
        "--output", str(shard), "--repo-id", repo_id,
        "--task", task,
        "--video-crf", str(video_crf),
        "--image-size", str(image_size),
        "--conversion-mode", conversion_mode,
    ]
    command.extend(["--source-kind", source_kind])
    if empty_scene:
        command.append("--empty-scene")
    if fk_library is not None:
        command.extend(["--fk-library", str(fk_library), "--fk-config", str(fk_config)])
    if camera_intrinsics is not None:
        command.extend(["--camera-intrinsics", str(camera_intrinsics)])
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    process = subprocess.run(command, text=True, capture_output=True, env=env)
    log_path.write_text(process.stdout + process.stderr)
    elapsed = time.perf_counter() - started
    if process.returncode == 0 and shard_valid(shard, image_size, **settings):
        report_path = shard / "conversion_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        return {
            "index": index, "session": session.name, "status": "converted",
            "repo_id": repo_id, "root": str(shard), "elapsed_s": round(elapsed, 6),
            "worker_report": report,
        }
    detail = process.stdout + process.stderr
    if "no synchronized frames" in detail:
        return {
            "index": index, "session": session.name, "status": "skip_invalid",
            "reason": "no synchronized frames", "elapsed_s": round(elapsed, 6),
        }
    raise RuntimeError(
        f"session conversion failed: {session.name} (exit={process.returncode}); "
        f"see {log_path}"
    )


def convert_dataset(args: argparse.Namespace) -> int:
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    sessions = complete_sessions(source, args.session_prefix)
    if not sessions:
        raise RuntimeError(f"no complete sessions under {source}")

    work = output.with_name(output.name + ".rdc_work")
    shards_dir, logs_dir = work / "shards", work / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    annotations = None
    empty_sources = set()
    excluded = []
    if args.annotation_index is not None:
        annotations = json.loads(args.annotation_index.read_text())
        annotation_root = Path(annotations["source_root"])
        allowed = {(annotation_root / s["relative_directory"]).resolve()
                   for s in annotations["sources"]}
        empty_sources = {(annotation_root / s["source_id"]).resolve()
                         for s in annotations.get("non_insertion_samples", [])
                         if s.get("sample_type") == "empty_scene" and s.get("keep")}
        excluded += [{"session": s.name, "reason": "not in retained annotation sources"}
                     for s in sessions if s.resolve() not in allowed]
        sessions = [s for s in sessions if s.resolve() in allowed]
    if args.max_duration_s is not None:
        retained = []
        for session in sessions:
            manifest = json.loads((session / "manifest.json").read_text())
            duration = (int(manifest["stopped_ns"]) - int(manifest["started_ns"])) / 1e9
            if duration > args.max_duration_s:
                excluded.append({"session": session.name, "reason": "duration_exceeds_limit",
                                 "duration_s": duration})
            else:
                retained.append(session)
        sessions = retained
    if not sessions:
        raise ValueError("no sessions remain after source validity/annotation filters")
    fk_library = fk_config = fk_provenance = None
    if args.conversion_mode == "feedback_effort_labeled":
        from feedback_state import build_fk_library, default_motion_control_root
        fk_library, fk_config, fk_provenance = build_fk_library(
            args.motion_control_root or default_motion_control_root(), work / "fk")
    # Hugging Face datasets creates lock/cache files while reading shard
    # Parquet. Keep them beside the resumable work directory instead of
    # assuming that ~/.cache is writable.
    os.environ.setdefault("HF_DATASETS_CACHE", str(work / "hf_datasets_cache"))
    started = time.perf_counter()
    results: list[dict[str, object]] = []

    print(
        f"[preprocess] sessions={len(sessions)} workers={max(1, args.workers)} "
        f"work={work}", flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for index, session in enumerate(sessions):
            shard = shards_dir / f"{index:06d}_{session.name}"
            log_path = logs_dir / f"{index:06d}_{session.name}.log"
            future = executor.submit(
                convert_shard, index, session, shard, log_path,
                task=args.task, video_crf=args.video_crf,
                image_size=args.image_size,
                conversion_mode=args.conversion_mode,
                source_kind=args.source_kind, empty_scene=session.resolve() in empty_sources,
                fk_library=fk_library, fk_config=fk_config,
                camera_intrinsics=args.camera_intrinsics,
            )
            futures[future] = session
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[preprocess {done}/{len(sessions)}] {result['session']} "
                f"{result['status']} {float(result.get('elapsed_s', 0)):.2f}s",
                flush=True,
            )

    results.sort(key=lambda item: int(item["index"]))
    accepted = [item for item in results if item["status"] in {"converted", "resumed"}]
    if not accepted:
        raise RuntimeError("no session passed preprocessing")

    rates = {json.loads((Path(str(item["root"])) / "meta/info.json").read_text())["fps"]
             for item in accepted}
    if len(rates) != 1:
        raise ValueError(f"mixed nominal FPS in one dataset: {sorted(rates)}; split the source datasets")
    fps = rates.pop()

    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    print(f"[aggregate] episodes={len(accepted)} -> {partial}", flush=True)
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[str(item["repo_id"]) for item in accepted],
        roots=[Path(str(item["root"])) for item in accepted],
        aggr_repo_id=args.repo_id,
        aggr_root=partial,
    )
    elapsed = time.perf_counter() - started
    report = {
        "conversion_revision": CONVERSION_REVISION,
        "source": str(source), "output": str(output), "repo_id": args.repo_id,
        "workers": max(1, args.workers), "fps": fps,
        "task": args.task, "resampling_enabled": False,
        "conversion_mode": args.conversion_mode,
        "source_kind": args.source_kind,
        "fk_provenance": fk_provenance,
        "source_exclusions": excluded,
        "max_duration_s": args.max_duration_s,
        "session_prefix": args.session_prefix,
        "video_crf": args.video_crf,
        "image_size": args.image_size,
        "total_sessions": len(sessions), "converted_episodes": len(accepted),
        "skipped_invalid": sum(item["status"] == "skip_invalid" for item in results),
        "elapsed_s": round(elapsed, 6), "sessions": results,
    }
    (partial / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    if args.conversion_mode == "feedback_effort_labeled":
        from conversion_annotations import export_annotations
        export_annotations(partial, source, accepted, annotations, args.source_kind, fk_provenance)
        shutil.copy2(fk_config, partial / "meta/fk_config.MvKDCfg")
    if args.camera_intrinsics is not None:
        from camera_metadata import write_camera_intrinsics
        write_camera_intrinsics(partial, args.camera_intrinsics, args.image_size)
    os.replace(partial, output)
    if not args.keep_work:
        shutil.rmtree(work)
    print(
        f"[done] episodes={len(accepted)} skipped_invalid={report['skipped_invalid']} "
        f"elapsed={elapsed:.2f}s output={output}", flush=True,
    )
    return 0


def main() -> int:
    from conversion_config import DEFAULTS, load_config, validate_jobs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="compressed_records root or one session")
    parser.add_argument("output", type=Path, nargs="?", help="new merged LeRobot dataset")
    parser.add_argument("--config", type=Path, help="YAML batch config (cannot mix with other arguments)")
    parser.add_argument("--repo-id")
    parser.add_argument("--task")
    parser.add_argument("--image-size", type=int, choices=(224, 256, 512))
    parser.add_argument("--video-crf", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--keep-work", action="store_true", default=None)
    parser.add_argument("--conversion-mode", choices=CONVERSION_MODES)
    parser.add_argument("--session-prefix")
    parser.add_argument("--source-kind")
    parser.add_argument("--motion-control-root", type=Path)
    parser.add_argument("--annotation-index", type=Path)
    parser.add_argument("--camera-intrinsics", type=Path)
    parser.add_argument("--max-duration-s", type=float)
    args = parser.parse_args()
    if args.config:
        if any(value is not None for key, value in vars(args).items() if key != "config"):
            parser.error("--config cannot be combined with positional or conversion arguments")
        jobs = load_config(args.config)
        # Validate the whole batch before launching anything, not just each job
        # when reached. No robot operations and no automatic upload.
        validate_jobs(jobs)
        for job in jobs:
            log = job.output.with_name(job.output.name + ".conversion.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), str(job.source), str(job.output),
                       "--repo-id", job.repo_id, "--task", job.task,
                       "--image-size", str(job.image_size), "--video-crf", str(job.video_crf),
                       "--workers", str(job.workers),
                       "--conversion-mode", getattr(job, "conversion_mode", "legacy")]
            command.extend(["--source-kind", job.source_kind])
            for name in ("motion_control_root", "annotation_index", "max_duration_s", "camera_intrinsics"):
                value = getattr(job, name)
                if value is not None:
                    command.extend(["--" + name.replace("_", "-"), str(value)])
            if job.keep_work:
                command.append("--keep-work")
            print(f"[start] {job.output.name} log={log}", flush=True)
            env = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = "1"
            with log.open("a") as stream:
                stream.write("\n[command] " + json.dumps(command, ensure_ascii=False) + "\n")
                stream.flush()
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            if result.returncode:
                print(f"[failed] {job.output.name}; see {log}", flush=True)
                return result.returncode
            print(f"[complete] {job.output.name}", flush=True)
        return 0
    if args.source is None or args.output is None:
        parser.error("provide --config CONFIG or both source and output")
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.conversion_mode is None:
        args.conversion_mode = "legacy"
    args.source, args.output = args.source.resolve(), args.output.resolve()
    args.repo_id = args.repo_id or "local/" + args.output.name
    validate_jobs([args])
    return convert_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
