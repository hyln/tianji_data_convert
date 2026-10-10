#!/usr/bin/env python3
"""Convert compress_record MP4+CSV sessions into a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import time
from statistics import median
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = {
    "head": "perception__sensors__hik_camera_8__image",
    "hik_camera_6": "perception__sensors__hik_camera_6__image",
    "hik_camera_7": "perception__sensors__hik_camera_7__image",
}
STATE_STEM = "motion_control__record__joint_state"
ACTION_STEM = "motion_control__record__joint_activate_cmd"
EEF_STEMS = {
    "left": "motion_control__states__eef__left",
    "right": "motion_control__states__eef__right",
}
CONTROL_STEM = "motion_control__states__control"
ROLLOUT_ERROR_STEM = "rollout__error_marker"
HOME_ARM_STATES = {4, 7, 8}  # HOMING_JOINT, HOMING_IK, HOME_WAIT
INFERENCE_ARM_STATE = 3
PRE_ENABLE_FRAMES = 3
CONVERSION_REVISION = "feedback_effort_rollout37_v13"
CONVERSION_MODES = ("legacy", "enable_only", "full_rollout_labeled", "feedback_effort_labeled")
JOINT_NAMES = [f"joint_L{i}" for i in range(1, 8)] + [f"joint_R{i}" for i in range(1, 8)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def timestamp(row: dict[str, str]) -> int:
    return int(row["header_stamp_ns"] or row["receive_timestamp_ns"])


def infer_fps(times: list[int]) -> int:
    """Infer a nominal integer timebase; never use it to select/drop frames."""
    if len(times) < 2:
        raise ValueError("at least two head camera timestamps are required to infer FPS")
    intervals = [b - a for a, b in zip(times, times[1:])]
    if any(dt <= 0 for dt in intervals):
        raise ValueError("head camera timestamps must be strictly increasing")
    fps = round(1e9 / median(intervals))
    if fps < 1:
        raise ValueError("head camera nominal FPS must be at least 1")
    return fps


def latest(rows: list[dict[str, str]], times: list[int], t_ns: int) -> dict[str, str] | None:
    index = bisect_right(times, t_ns) - 1
    return rows[index] if index >= 0 else None


def vector(row: dict[str, str], key: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(row[key]), dtype=np.float32)


def csv_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def teleop_enabled(row: dict[str, str]) -> bool:
    return csv_bool(row["left_enable"]) or csv_bool(row["right_enable"])


def enable_or_home_or_inference(row: dict[str, str]) -> bool:
    """Keep Enable, Home, and all inference frames regardless of Enable."""
    return (
        teleop_enabled(row)
        or int(row["left_arm_state"]) == INFERENCE_ARM_STATE
        or int(row["right_arm_state"]) == INFERENCE_ARM_STATE
        or int(row["left_arm_state"]) in HOME_ARM_STATES
        or int(row["right_arm_state"]) in HOME_ARM_STATES
    )


def decode_video(path: Path, image_size: int = 256, *, frame_processor=None) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        resized = (frame_processor(bgr) if frame_processor is not None else
                   cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(np.ascontiguousarray(rgb))
    capture.release()
    return frames


def decode_camera(video_dir: Path, stem: str, image_size: int = 256, *, frame_processor=None) -> list[np.ndarray]:
    mp4 = video_dir / f"{stem}.mp4"
    if mp4.exists():
        if frame_processor is None:
            return decode_video(mp4, image_size)
        return decode_video(mp4, image_size, frame_processor=frame_processor)
    image_dir = video_dir / stem
    paths = sorted(image_dir.glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no MP4 or JPEG frames for camera: {stem}")
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to decode JPEG: {path}")
        resized = (frame_processor(bgr) if frame_processor is not None else
                   cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(np.ascontiguousarray(rgb))
    return frames


def camera_stem(csv_dir: Path, stem: str) -> str:
    """Use raw-image recordings, while retaining old /compressed session support."""
    if (csv_dir / f"{stem}__frames.csv").exists():
        return stem
    legacy = f"{stem}__compressed"
    if (csv_dir / f"{legacy}__frames.csv").exists():
        return legacy
    raise FileNotFoundError(csv_dir / f"{stem}__frames.csv")


def decode_camera_views(video_dir, stem, views, image_size):
    """Decode a physical camera once and derive all its views from each frame."""
    frames = {name: [] for name in views}

    def append(bgr):
        for name, processor in views.items():
            processed = (processor(bgr) if processor is not None else
                         cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA))
            if processed.shape != (image_size, image_size, 3):
                raise ValueError(f"wrong processed image shape for {name}: {processed.shape}")
            frames[name].append(np.ascontiguousarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)))

    path = video_dir / f"{stem}.mp4"
    if path.exists():
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"failed to open video: {path}")
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                append(bgr)
        finally:
            capture.release()
    else:
        paths = sorted((video_dir / stem).glob("frame_*.jpg"))
        if not paths:
            raise FileNotFoundError(f"no MP4 or JPEG frames for camera: {stem}")
        for path in paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to decode JPEG: {path}")
            append(bgr)
    return frames


def convert(
    session: Path, output: Path, repo_id: str, task: str, video_crf: int,
    dataset: LeRobotDataset | None = None,
    *, image_size: int = 256, conversion_mode: str = "legacy", frame_filter: str = "mode",
    fk_library: Path | None = None, fk_config: Path | None = None,
    source_kind: str = "unspecified", empty_scene: bool = False,
    camera_intrinsics: Path | None = None,
) -> tuple[LeRobotDataset, dict[str, object]]:
    if image_size not in (224, 256, 512):
        raise ValueError("image_size must be 224, 256 or 512")
    if conversion_mode not in CONVERSION_MODES:
        raise ValueError(f"conversion_mode must be one of {CONVERSION_MODES}")
    total_started = time.perf_counter()
    feedback_mode = conversion_mode in ("feedback_effort_labeled", "full_rollout_labeled")
    if frame_filter not in ("mode", "legacy") or (frame_filter == "legacy" and conversion_mode not in ("legacy", "feedback_effort_labeled")):
        raise ValueError("invalid frame_filter for conversion_mode")
    feedback_states = None
    processors = {}
    calibration = {}
    from camera_metadata import view_sources
    if camera_intrinsics is not None:
        from camera_metadata import load_camera_intrinsics
        from wrist_undistortion import camera_processors
        calibration = load_camera_intrinsics(camera_intrinsics, image_size)
        processors = camera_processors(calibration, image_size)
    view_to_source = view_sources(calibration)
    manifest = json.loads((session / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"session is not complete: {session}")

    csv_dir, video_dir = session / "topic_csv", session / "video"
    state_rows = read_csv(csv_dir / f"{STATE_STEM}.csv")
    action_rows = read_csv(csv_dir / f"{ACTION_STEM}.csv")
    eef_rows = {} if feedback_mode else {
        side: read_csv(csv_dir / f"{stem}.csv") for side, stem in EEF_STEMS.items()}
    control_rows = read_csv(csv_dir / f"{CONTROL_STEM}.csv")
    marker_path = csv_dir / f"{ROLLOUT_ERROR_STEM}.csv"
    rollout_error_rows = read_csv(marker_path) if marker_path.exists() else []
    episode_has_y = bool(rollout_error_rows)
    if conversion_mode == "enable_only" and episode_has_y:
        raise RuntimeError("session contains rollout Y error marker")
    state_times = [timestamp(row) for row in state_rows]
    action_times = [timestamp(row) for row in action_rows]
    eef_times = {side: [timestamp(row) for row in rows] for side, rows in eef_rows.items()}
    control_times = [timestamp(row) for row in control_rows]
    if feedback_mode:
        from feedback_state import FeedbackFK, build_states, joint_vectors, SOURCE_KINDS
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"unsupported source kind: {source_kind}")
        if fk_library is None or fk_config is None:
            raise ValueError("feedback conversion requires --fk-library and --fk-config")
        for name, times in (("state", state_times), ("action", action_times),
                            ("control", control_times)):
            if any(b < a for a, b in zip(times, times[1:])):
                raise ValueError(f"{name} timestamps are not monotonic")
        feedback_states = build_states(state_rows, FeedbackFK(fk_library, fk_config))
        feedback_actions = joint_vectors(action_rows, "position", include_grippers=True)
    csv_s = time.perf_counter() - total_started

    decode_started = time.perf_counter()
    camera_stems = {name: camera_stem(csv_dir, stem) for name, stem in CAMERAS.items()}
    camera_rows = {
        name: read_csv(csv_dir / f"{stem}__frames.csv")
        for name, stem in camera_stems.items()
    }
    empty_cameras = [name for name, rows in camera_rows.items() if not rows]
    if empty_cameras:
        raise RuntimeError(
            "no synchronized frames passed Enable/Home filtering: "
            f"empty camera indices: {', '.join(empty_cameras)}"
        )
    camera_times = {
        name: [timestamp(row) for row in rows] for name, rows in camera_rows.items()
    }
    fps = infer_fps(camera_times["head"])
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"mixed nominal FPS in one dataset: {dataset.fps} vs {fps}")
    if len(view_to_source) > len(CAMERAS):
        camera_frames = {}
        for source, stem in camera_stems.items():
            views = {name: processors.get(name) for name, physical in view_to_source.items() if physical == source}
            camera_frames.update(decode_camera_views(video_dir, stem, views, image_size))
    else:
        camera_frames = {
            name: (decode_camera(video_dir, stem, image_size, frame_processor=processors[name])
                   if name in processors else decode_camera(video_dir, stem, image_size))
            for name, stem in camera_stems.items()
        }
    for name, source in view_to_source.items():
        if len(camera_rows[source]) != len(camera_frames[name]):
            raise RuntimeError(
                f"{name}: timestamp rows={len(camera_rows[source])}, decoded frames={len(camera_frames[name])}"
            )
    decode_s = time.perf_counter() - decode_started

    features = {
        "observation.state": {
            "dtype": "float32", "shape": (22,),
            "names": JOINT_NAMES + ["eef_left_x", "eef_left_y", "eef_left_z",
                                     "eef_right_x", "eef_right_y", "eef_right_z",
                                     "gripper_L", "gripper_R"],
        },
        "action": {
            "dtype": "float32", "shape": (16,),
            "names": [f"{name}_action" for name in JOINT_NAMES] +
                     ["gripper_L_cmd", "gripper_R_cmd"],
        },
    }
    if feedback_mode:
        from feedback_state import STATE_NAMES, extra_features, frame_labels
        state_names = STATE_NAMES + (["is_teleop"] if conversion_mode == "full_rollout_labeled" else [])
        features["observation.state"] = {"dtype": "float32", "shape": (len(state_names),), "names": state_names}
        features.update(extra_features())
    for name in view_to_source:
        features[f"observation.images.{name}"] = {
            "dtype": "video", "shape": (image_size, image_size, 3), "names": ("height", "width", "channels")
        }

    if dataset is None and output.exists():
        raise FileExistsError(output)
    primary_times = camera_times["head"]
    # Every original head frame participates. Only the established control-state
    # and synchronization-validity filters may exclude a frame.
    primary_indices = range(len(primary_times))
    first_enable_ns = next(
        (t_ns for row, t_ns in zip(control_rows, control_times) if teleop_enabled(row)),
        None,
    )
    pre_enable_indices = set()
    if first_enable_ns is not None:
        before_enable = [
            index for index in primary_indices
            if primary_times[index] < first_enable_ns
        ]
        pre_enable_indices.update(before_enable[-PRE_ENABLE_FRAMES:])

    assemble_started = time.perf_counter()
    candidates: list[tuple[int, dict]] = []
    inactive_frames = 0
    pre_enable_frames_kept = 0
    for primary_index in primary_indices:
        t_ns = primary_times[primary_index]
        control_row = latest(control_rows, control_times, t_ns)
        if conversion_mode == "feedback_effort_labeled" and frame_filter != "legacy":
            # Preserve all synchronized original head frames, including rollout,
            # intervention, idle/reset and explicitly retained empty scenes.
            keep_pre_enable, keep_active = False, True
        elif conversion_mode == "enable_only":
            keep_pre_enable = False
            keep_active = control_row is not None and teleop_enabled(control_row)
        elif conversion_mode == "full_rollout_labeled":
            keep_pre_enable = False
            keep_active = control_row is not None
        else:
            keep_pre_enable = primary_index in pre_enable_indices
            before_first_enable = first_enable_ns is not None and t_ns < first_enable_ns
            keep_active = (
                not before_first_enable
                and control_row is not None
                and enable_or_home_or_inference(control_row)
            )
        if not keep_pre_enable and not keep_active:
            inactive_frames += 1
            continue
        state_row = latest(state_rows, state_times, t_ns)
        action_row = latest(action_rows, action_times, t_ns)
        if state_row is None or action_row is None:
            continue
        if feedback_mode:
            state = feedback_states[bisect_right(state_times, t_ns) - 1]
            commanded = feedback_actions[bisect_right(action_times, t_ns) - 1]
        else:
            left = latest(eef_rows["left"], eef_times["left"], t_ns)
            right = latest(eef_rows["right"], eef_times["right"], t_ns)
            if left is None or right is None:
                continue
            measured = vector(state_row, "position")
            commanded = vector(action_row, "position")
            if measured.shape != (16,) or commanded.shape != (16,):
                continue
            state = np.concatenate([
                measured[:14],
                np.asarray([left[f"pose.position.{axis}"] for axis in "xyz"], dtype=np.float32),
                np.asarray([right[f"pose.position.{axis}"] for axis in "xyz"], dtype=np.float32),
                measured[14:16],
            ]).astype(np.float32)
        if conversion_mode == "full_rollout_labeled":
            state = np.concatenate([
                state,
                np.asarray([
                    float(teleop_enabled(control_row)),
                ], dtype=np.float32),
            ])
        frame = {"observation.state": state, "action": commanded, "task": task}
        if feedback_mode:
            frame.update(frame_labels(control_row, source_kind, episode_has_y, empty_scene))
            source_values = {"timestamp_ns": t_ns, "head_frame_index": primary_index,
                             "joint_state_timestamp_ns": timestamp(state_row),
                             "action_timestamp_ns": timestamp(action_row),
                             "control_timestamp_ns": timestamp(control_row) if control_row else -1}
            frame.update({f"source.{key}": np.asarray([value], dtype=np.int64)
                          for key, value in source_values.items()})
        valid = True
        for name, source in view_to_source.items():
            times = camera_times[source]
            index = bisect_right(times, t_ns) - 1
            if index < 0:
                valid = False
                break
            frame[f"observation.images.{name}"] = camera_frames[name][index]
        if valid:
            candidates.append((t_ns, frame))
            if keep_pre_enable:
                pre_enable_frames_kept += 1

    # Before the first Enable, keep only its three immediately preceding original
    # frames regardless of control state. Keep active Enable/Home/Inference
    # frames throughout the remaining recording, without a release-based cutoff.
    # Inference is retained regardless of Enable. Home motion is also useful
    # supervision, so HOMING_JOINT/HOMING_IK/HOME_WAIT frames are retained.
    kept = len(candidates)
    if not kept:
        raise RuntimeError("no synchronized frames passed Enable/Home filtering")
    active_duration_s = (
        (candidates[-1][0] - candidates[0][0]) / 1e9 if kept > 1 else 0.0
    )
    assemble_s = time.perf_counter() - assemble_started

    encode_started = time.perf_counter()
    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=repo_id, root=output, fps=fps, features=features, use_videos=True,
            vcodec="h264", image_writer_processes=0, image_writer_threads=0,
            streaming_encoding=True, encoder_queue_maxsize=256,
        )
        # LeRobot v3 defaults to CRF 30. Tianji's established conversion recipe uses
        # CRF 18 to avoid visible artifacts after resizing the source.
        dataset._streaming_encoder.crf = int(video_crf)
    for _, frame in candidates:
        # Offline conversion must wait for encoding, never drop frames. There
        # is only one producer; consumers can only free queue slots while we
        # wait. Let feed_frame report any encoder-thread failure as usual.
        encoder = dataset._streaming_encoder
        for key, queue in encoder._frame_queues.items():
            while queue.full() and encoder._threads[key].is_alive():
                time.sleep(0.01)
        dataset.add_frame(frame)
    dataset.save_episode()
    encode_s = time.perf_counter() - encode_started
    total_s = time.perf_counter() - total_started
    print(
        f"converted session={session.name} frames={kept} fps={fps} "
        f"effective_duration_s={active_duration_s:.3f} "
        f"pre_enable_kept={pre_enable_frames_kept} "
        f"inactive_filtered={inactive_frames} video_crf={video_crf} "
        f"image_size={image_size} output={output}\n"
        f"[timing] csv={csv_s:.3f}s decode_resize={decode_s:.3f}s "
        f"align_filter={assemble_s:.3f}s encode_save={encode_s:.3f}s total={total_s:.3f}s"
    )
    return dataset, {
        "session": session.name,
        "status": "converted",
        "conversion_revision": CONVERSION_REVISION,
        "conversion_mode": conversion_mode,
        "frame_filter": frame_filter,
        "episode_has_y": episode_has_y,
        "source_kind": source_kind,
        "empty_scene": empty_scene,
        "state_schema": ("feedback_angles_effort_fk_grippers_37_teleop" if conversion_mode == "full_rollout_labeled"
                         else "feedback_angles_effort_fk_grippers_36" if feedback_mode else "legacy_cmd_fk"),
        "eef_csv_used": not feedback_mode,
        "undistorted_cameras": sorted(name for name, processor in processors.items() if processor.undistorted),
        "processed_cameras": sorted(processors),
        "output_views": view_to_source,
        "image_preprocessing": calibration.get("image_preprocessing", {"mode": "resize"}),
        "control_unknown_frames": sum(int(f.get("annotation.control_valid", [1])[0]) == 0 for _, f in candidates),
        "teleop_frames": sum(int(f.get("annotation.is_teleop", [0])[0]) for _, f in candidates),
        "inference_frames": sum(int(f.get("annotation.is_inference", [0])[0]) for _, f in candidates),
        "image_size": image_size,
        "fps": fps,
        "source_head_frames": len(primary_times),
        "resampling_enabled": False,
        "frames": kept,
        "effective_duration_s": round(active_duration_s, 6),
        "pre_enable_frames_kept": pre_enable_frames_kept,
        "tail_cutoff_enabled": False,
        "inactive_frames_filtered": inactive_frames,
        "timing_s": {
            "csv": round(csv_s, 6), "decode_resize": round(decode_s, 6),
            "align_filter": round(assemble_s, 6), "encode_save": round(encode_s, 6),
            "total": round(total_s, 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/tianji_compressed")
    parser.add_argument("--task", default="<control mode> joint </control mode> tianji teleoperation")
    parser.add_argument("--image-size", type=int, choices=(224, 256, 512), default=256,
                        help="square output video resolution (default: 256)")
    parser.add_argument("--video-crf", type=int, default=18,
                        help="H.264 CRF; lower is higher quality (default: 18)")
    parser.add_argument("--conversion-mode", choices=CONVERSION_MODES, default="legacy")
    parser.add_argument("--frame-filter", choices=("mode", "legacy"), default="mode")
    parser.add_argument("--fk-library", type=Path)
    parser.add_argument("--fk-config", type=Path)
    parser.add_argument("--source-kind", default="unspecified")
    parser.add_argument("--empty-scene", action="store_true")
    parser.add_argument("--camera-intrinsics", type=Path)
    args = parser.parse_args()
    source = args.session.resolve()
    output = args.output.resolve()
    sessions = (
        [source]
        if (source / "manifest.json").exists()
        else sorted(
            path for path in source.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )
    )
    if not sessions:
        raise RuntimeError(f"no sessions found under {source}")

    dataset = None
    report: list[dict[str, object]] = []
    for index, session in enumerate(sessions, 1):
        print(f"[batch {index}/{len(sessions)}] {session.name}")
        try:
            dataset, result = convert(
                session, output, args.repo_id, args.task,
                args.video_crf, dataset, image_size=args.image_size,
                conversion_mode=args.conversion_mode,
                frame_filter=args.frame_filter,
                fk_library=args.fk_library, fk_config=args.fk_config,
                source_kind=args.source_kind, empty_scene=args.empty_scene,
                camera_intrinsics=args.camera_intrinsics,
            )
            report.append(result)
        except RuntimeError as exc:
            if not any(message in str(exc) for message in (
                "no synchronized frames passed Enable/Home filtering",
                "session contains rollout Y error marker",
            )):
                raise
            print(f"[skip_invalid] {session.name}: {exc}")
            report.append({
                "session": session.name,
                "status": "skip_invalid",
                "reason": str(exc),
            })

    if dataset is None:
        raise RuntimeError("no session passed Enable/Home filtering")
    dataset.finalize()
    summary = {
        "conversion_revision": CONVERSION_REVISION,
        "conversion_mode": args.conversion_mode,
        "frame_filter": args.frame_filter,
        "image_size": args.image_size,
        "video_crf": args.video_crf,
        "task": args.task,
        "source_kind": args.source_kind,
        "empty_scene": args.empty_scene,
        "fk_library_sha256": None,
        "fk_config_sha256": None,
        "camera_intrinsics_sha256": None,
        "fps": dataset.fps,
        "resampling_enabled": False,
        "source": str(source),
        "total_sessions": len(sessions),
        "converted_episodes": sum(item["status"] == "converted" for item in report),
        "skipped_invalid": sum(item["status"] == "skip_invalid" for item in report),
        "total_frames": sum(int(item.get("frames", 0)) for item in report),
        "sessions": report,
    }
    if args.conversion_mode in ("feedback_effort_labeled", "full_rollout_labeled"):
        from feedback_state import file_sha256, source_fingerprint
        summary["fk_library_sha256"] = file_sha256(args.fk_library)
        summary["fk_config_sha256"] = file_sha256(args.fk_config)
        summary["source_fingerprint"] = source_fingerprint(sessions[0]) if len(sessions) == 1 else None
    if args.camera_intrinsics is not None:
        from feedback_state import file_sha256
        from camera_metadata import write_camera_intrinsics
        summary["camera_intrinsics_sha256"] = file_sha256(args.camera_intrinsics)
        write_camera_intrinsics(output, args.camera_intrinsics, args.image_size)
    (output / "conversion_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"[batch complete] episodes={summary['converted_episodes']} "
        f"skipped_invalid={summary['skipped_invalid']} "
        f"frames={summary['total_frames']} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
