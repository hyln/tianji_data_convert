"""Validate and preserve approximate/shared camera intrinsics as metadata."""
import json
import math
from pathlib import Path

FIVE_VIEW_SOURCES = {"head": "head", "hik_camera_6": "hik_camera_6",
                     "hik_camera_7": "hik_camera_7", "hik_camera_6_crop": "hik_camera_6",
                     "hik_camera_7_crop": "hik_camera_7"}


def view_sources(calibration):
    if calibration.get("image_preprocessing", {}).get("mode") == "five_view":
        return dict(FIVE_VIEW_SOURCES)
    return {name: name for name in ("head", "hik_camera_6", "hik_camera_7")}


def _validate_five_view(data, image_size):
    base = data.get("source_camera_matrix")
    _validate_matrix(base)
    width, height = data.get("source_image_width"), data.get("source_image_height")
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("source image dimensions must be positive integers")
    processing = data["image_preprocessing"]
    views = processing.get("views", {})
    features = {"observation.images." + name for name in FIVE_VIEW_SOURCES}
    if set(views) != set(FIVE_VIEW_SOURCES) or set(data.get("per_camera_intrinsics", {})) != features:
        raise ValueError("five-view recipe must define exactly five output views")
    if set(data.get("shared_base_intrinsics_across", [])) != features:
        raise ValueError("shared base intrinsics must name all five views")
    if processing.get("crop_resize_before_crop") is not False:
        raise ValueError("wrist crop views must disable resizing")
    for name, source in FIVE_VIEW_SOURCES.items():
        view = views[name]
        operation = "rectify_crop" if name.endswith("_crop") else "resize" if name == "head" else "rectify_resize"
        if view.get("source_camera") != source or view.get("operation") != operation:
            raise ValueError("five-view source or processing operation is inconsistent")
        wanted = [row[:] for row in base]
        if operation == "rectify_crop":
            rectangle = view.get("crop_xywh")
            if (not isinstance(rectangle, list) or len(rectangle) != 4
                    or any(type(v) is not int for v in rectangle)):
                raise ValueError("crop must be integer x,y,width,height")
            x, y, w, h = rectangle
            if x < 0 or y < 0 or x + w > width or y + h > height or w != image_size or h != image_size:
                raise ValueError("crop must fit within source and match output image size")
            if rectangle != processing.get("crop_rectangles", {}).get(source):
                raise ValueError("crop rectangle conflicts with view definition")
            wanted[0][2] -= x
            wanted[1][2] -= y
        else:
            sx, sy = image_size / width, image_size / height
            wanted[0] = [v * sx for v in wanted[0]]
            wanted[1] = [v * sy for v in wanted[1]]
            wanted[0][2] += (sx - 1) / 2
            wanted[1][2] += (sy - 1) / 2
        intrinsics = data["per_camera_intrinsics"]["observation.images." + name]
        if intrinsics.get("image_width") != image_size or intrinsics.get("image_height") != image_size:
            raise ValueError("per-view resolution differs from output")
        if intrinsics.get("undistorted") != (source != "head"):
            raise ValueError("per-view undistortion flag is inconsistent")
        matrix = intrinsics.get("camera_matrix")
        _validate_matrix(matrix)
        if any(abs(matrix[i][j] - wanted[i][j]) > 1e-8 for i in range(3) for j in range(3)):
            raise ValueError("output camera_matrix does not match view processing")
    correction = data.get("wrist_undistortion", {})
    if not correction.get("enabled") or set(correction.get("cameras", [])) != {"hik_camera_6", "hik_camera_7"}:
        raise ValueError("five-view recipe must correct exactly wrist cameras 6 and 7")


def _validate_matrix(matrix):
    if (not isinstance(matrix, list) or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in matrix for v in row)
            or matrix[0][0] <= 0 or matrix[1][1] <= 0 or matrix[2] != [0, 0, 1]):
        raise ValueError("camera_matrix must be a finite 3x3 intrinsic matrix")


def load_camera_intrinsics(path: Path, image_size: int):
    data = json.loads(path.read_text())
    if data.get("image_width") != image_size or data.get("image_height") != image_size:
        raise ValueError("camera intrinsics resolution does not match output images")
    expected = {"observation.images.head", "observation.images.hik_camera_6", "observation.images.hik_camera_7"}
    preprocessing = data.get("image_preprocessing", {})
    if preprocessing.get("mode") == "five_view":
        _validate_five_view(data, image_size)
    elif preprocessing.get("mode") == "direct_crop":
        if preprocessing.get("resize_before_crop") is not False:
            raise ValueError("direct crop must explicitly disable resizing")
        if set(data.get("shared_base_intrinsics_across", [])) != expected:
            raise ValueError("shared base intrinsics must name all three camera features")
        base = data.get("source_camera_matrix")
        _validate_matrix(base)
        width, height = data.get("source_image_width"), data.get("source_image_height")
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise ValueError("source image dimensions must be positive integers")
        rectangles = preprocessing.get("crop_rectangles", {})
        if set(rectangles) != {"head", "hik_camera_6", "hik_camera_7"}:
            raise ValueError("crop rectangles must name all three cameras")
        per_camera = data.get("per_camera_intrinsics", {})
        if set(per_camera) != expected:
            raise ValueError("per-camera output intrinsics must name all three features")
        for name, rectangle in rectangles.items():
            if (not isinstance(rectangle, list) or len(rectangle) != 4
                    or any(type(v) is not int for v in rectangle)):
                raise ValueError("crop must be integer x,y,width,height")
            x, y, w, h = rectangle
            if x < 0 or y < 0 or x + w > width or y + h > height or w != image_size or h != image_size:
                raise ValueError("crop must fit within source and match output image size")
            intrinsics = per_camera[f"observation.images.{name}"]
            if intrinsics.get("image_width") != w or intrinsics.get("image_height") != h:
                raise ValueError("per-camera resolution differs from crop")
            matrix = intrinsics.get("camera_matrix")
            _validate_matrix(matrix)
            wanted = [row[:] for row in base]
            wanted[0][2] -= x
            wanted[1][2] -= y
            if any(abs(matrix[i][j] - wanted[i][j]) > 1e-8 for i in range(3) for j in range(3)):
                raise ValueError("output camera_matrix does not match crop origin")
        correction = data.get("wrist_undistortion", {})
        if not correction.get("enabled") or set(correction.get("cameras", [])) != {"hik_camera_6", "hik_camera_7"}:
            raise ValueError("crop recipe must correct exactly wrist cameras 6 and 7")
    else:
        if preprocessing.get("mode") is not None:
            raise ValueError("unsupported image preprocessing mode")
        _validate_matrix(data.get("camera_matrix"))
        if set(data.get("shared_across", [])) != expected:
            raise ValueError("shared intrinsics must explicitly name all three camera features")
    return data


def write_camera_intrinsics(dataset: Path, path: Path, image_size: int):
    data = load_camera_intrinsics(path, image_size)
    target = dataset / "meta/camera_intrinsics.json"
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if target.exists() and json.loads(target.read_text()) != data:
        raise FileExistsError(f"refusing to replace different camera intrinsics: {target}")
    target.write_text(payload)
