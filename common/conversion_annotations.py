"""Map existing virtual insertion annotations to retained LeRobot frame indices."""
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from feedback_state import STATE_NAMES


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def export_annotations(output, source, accepted, annotations, source_kind, fk_provenance):
    tables = [pq.read_table(path, columns=["episode_index", "frame_index", "source.head_frame_index"])
              for path in sorted((output / "data").rglob("*.parquet"))]
    import pyarrow as pa
    data = pa.concat_tables(tables).to_pandas()
    sources, mapped = [], []
    source_annotations = {s["episode_name"]: s for s in (annotations or {}).get("sources", [])}
    for episode_index, item in enumerate(accepted):
        worker = json.loads((Path(item["root"]) / "conversion_report.json").read_text())
        rows = data[data.episode_index == episode_index].sort_values("frame_index")
        head_indices = np.array([np.asarray(v).reshape(-1)[0] for v in rows["source.head_frame_index"]], dtype=np.int64)
        raw = source_annotations.get(item["session"])
        raw_id = raw["source_id"] if raw else f"{source.name}/{item['session']}"
        entry = {"episode_index": episode_index, "source_id": raw_id,
                 "source_directory": worker["source"], "source_kind": source_kind,
                 "frames": len(rows), "worker_report": worker}
        if raw:
            entry["source_annotation"] = raw
        sources.append(entry)
        for segment in ((annotations or {}).get("episodes", [])
                        + (annotations or {}).get("non_insertion_samples", [])):
            if segment["source_id"] != raw_id:
                continue
            head = [v for k, v in segment.get("videos", {}).items() if "hik_camera_8" in k]
            if len(head) != 1:
                raise ValueError(f"missing/ambiguous head video annotation for {raw_id}")
            mask = np.zeros(len(rows), dtype=bool)
            for start, stop in head[0]["frame_spans"]:
                mask |= (head_indices >= start) & (head_indices < stop)
            annotated = copy.deepcopy(segment)
            annotated["lerobot_episode_index"] = episode_index
            annotated["lerobot_frame_spans"] = spans(mask)
            annotated["lerobot_frame_count"] = int(mask.sum())
            annotated["mapping_basis"] = "original_head_camera_frame_index; zero-based half-open spans"
            mapped.append(annotated)
    result = {"schema_version": "tianji.feedback_effort.virtual_annotations.v1",
              "description": "One original recording per episode; insertion candidates remain editable annotations.",
              "source_kind": source_kind,
              "state_layout": {"angles": [0, 14], "effort": [14, 28], "feedback_fk_xyz": [28, 34],
                               "grippers": [34, 36], "names": STATE_NAMES},
              "fk_provenance": fk_provenance,
              "label_semantics": {
                  "is_rollout_batch": "User-confirmed rollout-with-intervention batch; unspecified batches are 0, not confirmed pure teleop.",
                  "is_teleop": "left_enable OR right_enable from latest recorded control state; an Enable signal, not visual proof of intervention.",
                  "is_inference": "Either recorded arm_state equals 3; may coexist with other-arm Enable.",
                  "control_valid": "0 before first available control timestamp; mode/Enable labels then unknown (arm_state=-1).",
                  "has_error_marker": "Any recorded Y/error_marker in the entire source; no marker does not prove every segment succeeded.",
                  "is_empty_scene": "User-retained empty scene, score 6 in virtual annotations.",
                  "effort": "Raw measured joint_state.effort, unscaled; no claim of calibrated contact force.",
                  "source.timestamp_ns": "Original head-camera timestamp, int64 nanoseconds; dataset timestamp remains frame_index/fps."},
              "sources": sources, "virtual_segments": mapped,
              "excluded_sources": (annotations or {}).get("excluded_sources", [])}
    (output / "meta/source_annotations.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
