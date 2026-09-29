"""Measured joint position/effort and offline vendor FK; never sends commands."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

MODE = "feedback_effort_labeled"
SOURCE_KINDS = ("unspecified", "rollout_with_teleop_intervention")
RAW_JOINT_NAMES = [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)]
RAW_NAMES = RAW_JOINT_NAMES + ["gripper_left", "gripper_right"]
JOINT_NAMES = [f"joint_{side}{i}" for side in ("L", "R") for i in range(1, 8)]
STATE_NAMES = (JOINT_NAMES + [f"{name}_effort" for name in JOINT_NAMES]
               + [f"eef_{side}_{axis}" for side in ("left", "right") for axis in "xyz"]
               + ["gripper_L", "gripper_R"])
LABEL_NAMES = ["is_rollout_batch", "control_valid", "is_teleop", "left_enable",
               "right_enable", "left_arm_state", "right_arm_state", "is_inference",
               "has_error_marker", "is_empty_scene"]
SOURCE_NAMES = ["timestamp_ns", "head_frame_index", "joint_state_timestamp_ns",
                "action_timestamp_ns", "control_timestamp_ns"]


def default_motion_control_root():
    return (Path(__file__).resolve().parents[2] / "tianji_arm_teleop/src/mavrin/motion_control")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(session):
    """Hash numeric inputs; record video stat identity without re-reading video."""
    inputs = {str(p.relative_to(session)): file_sha256(p)
              for p in [session / "manifest.json", *sorted((session / "topic_csv").glob("*.csv"))]}
    video_files = sorted(p for p in (session / "video").rglob("*") if p.is_file())
    videos = {str(p.relative_to(session)): [p.stat().st_size, p.stat().st_mtime_ns]
              for p in video_files}
    return hashlib.sha256(json.dumps([inputs, videos], sort_keys=True).encode()).hexdigest()


def build_fk_library(motion_control_root: Path, build_dir: Path):
    """Content-addressed offline SDK build; call once before launching workers."""
    sdk = motion_control_root / "kinematicsSDK"
    config = motion_control_root / "config/ccs_m6_40.MvKDCfg"
    bridge = Path(__file__).with_name("feedback_fk.cpp")
    sources = sorted(sdk.glob("*.cpp"))
    if not sources or not config.is_file():
        raise FileNotFoundError(f"missing vendor SDK/config under {motion_control_root}")
    inputs = [bridge, config] + sorted(sdk.glob("*.h")) + sources
    hashes = {str(p): file_sha256(p) for p in inputs}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / f"feedback_fk_{digest[:20]}.so"
    if not library.exists():
        partial = library.with_suffix(f".{os.getpid()}.partial")
        command = ["g++", "-O2", "-shared", "-fPIC", "-std=c++17", "-I", str(sdk),
                   str(bridge), *map(str, sources), "-o", str(partial)]
        result = subprocess.run(command, capture_output=True, text=True)
        (build_dir / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"FK SDK build failed; see {build_dir / 'build.log'}")
        os.replace(partial, library)
    return library, config, {"implementation": "FX_Robot_Kine_FK",
                            "coordinate_frame": "base_link", "xyz_units": "m",
                            "joint_units": "rad", "input_hashes": hashes,
                            "library_sha256": file_sha256(library),
                            "config_sha256": file_sha256(config)}


class FeedbackFK:
    def __init__(self, library: Path, config: Path):
        self.lib = ctypes.CDLL(str(library))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.metrics_init.argtypes = [ctypes.c_char_p]
        self.lib.metrics_init.restype = ctypes.c_int
        self.lib.metrics_fk.argtypes = [ctypes.c_int, ctypes.c_int, ptr, ptr]
        self.lib.metrics_fk.restype = ctypes.c_int
        if self.lib.metrics_init(str(config).encode()) != 1:
            raise ValueError(f"could not initialize FK with {config}")

    def xyz_m(self, joints):
        joints = np.asarray(joints)
        if joints.ndim != 2 or joints.shape[1] != 14 or not np.isfinite(joints).all():
            raise ValueError("FK requires finite [N,14] actual joint angles in radians")
        xyz = []
        for arm in (0, 1):
            values = np.ascontiguousarray(joints[:, arm * 7:(arm + 1) * 7], dtype=np.float64)
            result = np.zeros((len(values), 3), dtype=np.float64)
            if self.lib.metrics_fk(arm, len(values), values, result) != 1:
                raise ValueError("vendor FK failed")
            xyz.append(result / 1000.0)
        result = np.concatenate(xyz, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite FK result")
        return result


def joint_vectors(rows, field, *, include_grippers):
    """Resolve joints by name; missing effort is an error, never zero-filled."""
    expected = RAW_NAMES if include_grippers else RAW_JOINT_NAMES
    vectors = []
    for index, row in enumerate(rows):
        names = json.loads(row["name"])
        values = json.loads(row[field])
        if len(set(names)) != len(names) or len(values) != len(names):
            raise ValueError(f"invalid joint names/{field} lengths at CSV row {index}")
        try:
            vectors.append([values[names.index(name)] for name in expected])
        except ValueError as exc:
            raise ValueError(f"missing joint at CSV row {index}: {exc}") from exc
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (len(rows), len(expected)) or not np.isfinite(result).all():
        raise ValueError(f"invalid or empty {field} data")
    return result


def build_states(rows, fk):
    position = joint_vectors(rows, "position", include_grippers=True)
    effort = joint_vectors(rows, "effort", include_grippers=False)
    # Calculate from the exact float32 angles stored in State, not a different
    # timestamp, the EEF topic, or commanded/action joints.
    return np.concatenate([position[:, :14], effort, fk.xyz_m(position[:, :14]),
                           position[:, 14:]], axis=1).astype(np.float32)


def extra_features():
    return {f"{prefix}.{name}": {"dtype": "int64", "shape": (1,), "names": [name]}
            for prefix, names in (("annotation", LABEL_NAMES), ("source", SOURCE_NAMES)) for name in names}


def frame_labels(control, source_kind, has_y, empty_scene):
    enabled = lambda key: control is not None and control[key].lower() in {"true", "1", "yes"}
    left, right = enabled("left_enable"), enabled("right_enable")
    states = [int(control[f"{side}_arm_state"]) if control is not None else -1
              for side in ("left", "right")]
    values = [source_kind == SOURCE_KINDS[1], control is not None, left or right,
              left, right, *states, 3 in states, has_y, empty_scene]
    return {f"annotation.{key}": np.asarray([value], dtype=np.int64)
            for key, value in zip(LABEL_NAMES, values)}
