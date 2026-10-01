"""Small, strict YAML configuration for sequential conversion jobs."""
from argparse import Namespace
from pathlib import Path

import yaml

DEFAULTS = {
    "image_size": 256,
    "video_crf": 18,
    "workers": 4,
    "keep_work": False,
    "task": "<control mode> joint </control mode> tianji teleoperation",
    "conversion_mode": "legacy",
    "frame_filter": "mode",
    "source_kind": "unspecified",
    "motion_control_root": None,
    "annotation_index": None,
    "camera_intrinsics": None,
    "max_duration_s": None,
}


def mapping(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown, key=str)}")
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def relative_path(root, value, label):
    value = Path(text(value, label))
    if value.is_absolute() or ".." in value.parts or value == Path("."):
        raise ValueError(f"{label} must be a relative path below its root")
    result = (root / value).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError(f"{label} escapes its root")
    return result


def load_config(path: Path) -> list[Namespace]:
    path = path.expanduser().resolve()
    config = mapping(yaml.safe_load(path.read_text()),
                     {"source_root", "output_root", "conversion", "datasets"}, "config")
    roots = {}
    for key in ("source_root", "output_root"):
        value = Path(text(config.get(key), key)).expanduser()
        roots[key] = (path.parent / value).resolve()
    settings = dict(DEFAULTS)
    settings.update(mapping(config.get("conversion", {}), set(DEFAULTS), "conversion"))
    for key in ("motion_control_root", "annotation_index", "camera_intrinsics"):
        if settings[key] is not None:
            settings[key] = (path.parent / Path(text(settings[key], key)).expanduser()).resolve()
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a nonempty list")
    jobs = []
    for index, item in enumerate(datasets):
        item = mapping(item, {"source", "output", "repo_id", "source_kind"}, f"datasets[{index}]")
        source = relative_path(roots["source_root"], item.get("source"), "source")
        output = relative_path(roots["output_root"], item.get("output"), "output")
        repo_id = text(item.get("repo_id", "local/" + output.name), "repo_id")
        job_settings = dict(settings)
        job_settings["source_kind"] = item.get("source_kind", settings["source_kind"])
        jobs.append(Namespace(source=source, output=output, repo_id=repo_id, **job_settings))
    validate_jobs(jobs)
    return jobs


def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents


def validate_jobs(jobs):
    destinations = []
    sources = [job.source.resolve() for job in jobs]
    for job in jobs:
        if not job.source.is_dir():
            raise FileNotFoundError(job.source)
        if job.output.exists():
            raise FileExistsError(job.output)
        if type(job.image_size) is not int or job.image_size not in (224, 256, 512):
            raise ValueError("image_size must be 224, 256 or 512")
        if type(job.video_crf) is not int or not 0 <= job.video_crf <= 51:
            raise ValueError("video_crf must be an integer in [0, 51]")
        if type(job.workers) is not int or job.workers < 1:
            raise ValueError("workers must be a positive integer")
        if type(job.keep_work) is not bool:
            raise ValueError("keep_work must be a YAML boolean")
        text(job.task, "task")
        text(job.repo_id, "repo_id")
        if getattr(job, "frame_filter", "mode") not in ("mode", "legacy"):
            raise ValueError("frame_filter must be mode or legacy")
        if getattr(job, "frame_filter", "mode") == "legacy" and job.conversion_mode not in ("legacy", "feedback_effort_labeled"):
            raise ValueError("legacy frame_filter requires legacy or feedback_effort_labeled schema")
        if getattr(job, "conversion_mode", "legacy") not in (
                "legacy", "enable_only", "full_rollout_labeled", "feedback_effort_labeled"):
            raise ValueError("unknown conversion_mode")
        if getattr(job, "source_kind", "unspecified") not in (
                "unspecified", "rollout_with_teleop_intervention"):
            raise ValueError("unknown source_kind")
        duration = getattr(job, "max_duration_s", None)
        if duration is not None and (type(duration) not in (int, float) or not 0 < duration < float("inf")):
            raise ValueError("max_duration_s must be a finite positive number")
        for key in ("motion_control_root", "annotation_index", "camera_intrinsics"):
            value = getattr(job, key, None)
            if value is not None and not Path(value).exists():
                raise FileNotFoundError(value)
        camera_intrinsics = getattr(job, "camera_intrinsics", None)
        if camera_intrinsics is not None:
            from camera_metadata import load_camera_intrinsics
            load_camera_intrinsics(camera_intrinsics, job.image_size)
        # Protect inputs and every other job from output/cache/log collisions.
        for suffix in ("", ".partial", ".rdc_work", ".conversion.log"):
            target = job.output.with_name(job.output.name + suffix).resolve()
            if any(overlaps(target, source) for source in sources):
                raise ValueError(f"output/cache/log overlaps source: {target}")
            if any(overlaps(target, other) for other in destinations):
                raise ValueError(f"duplicate or overlapping output/cache/log: {target}")
            destinations.append(target)
