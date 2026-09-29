"""Deterministic wrist-camera preprocessing shared by conversion and inference."""
import cv2
import numpy as np


class WristUndistorter:
    undistorted = True
    def __init__(self, calibration, camera, image_size):
        processing = calibration["wrist_undistortion"]
        if processing["model"] != "equidistant":
            raise ValueError("only explicitly configured equidistant correction is supported")
        self.width = int(calibration["source_image_width"])
        self.height = int(calibration["source_image_height"])
        self.image_size = image_size
        self.intermediate = image_size * int(processing["supersampling_factor"])
        k = np.asarray(calibration["source_camera_matrix"], dtype=np.float64)
        d = np.asarray(processing["coefficients"][camera], dtype=np.float64)
        if d.shape != (4,) or not np.isfinite(d).all():
            raise ValueError("equidistant requires exactly four finite coefficients")
        sx, sy = self.intermediate / self.width, self.intermediate / self.height
        pixel_affine = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
        self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(
            k, d, np.eye(3), pixel_affine @ k,
            (self.intermediate, self.intermediate), cv2.CV_32FC1)

    def __call__(self, bgr):
        if bgr.shape != (self.height, self.width, 3):
            raise ValueError(f"recorded image shape {bgr.shape} differs from calibrated ROI "
                             f"{(self.height, self.width, 3)}")
        rectified = cv2.remap(bgr, self.map_x, self.map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        return cv2.resize(rectified, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)


class DirectCropper:
    """Crop original-scale pixels, with wrist rectification before the crop."""
    def __init__(self, calibration, camera, image_size):
        self.width = int(calibration["source_image_width"])
        self.height = int(calibration["source_image_height"])
        self.x, self.y, width, height = calibration["image_preprocessing"]["crop_rectangles"][camera]
        if (width != image_size or height != image_size or self.x < 0 or self.y < 0
                or self.x + width > self.width or self.y + height > self.height):
            raise ValueError("crop must fit within source and match output image size")
        self.image_size = image_size
        processing = calibration.get("wrist_undistortion", {})
        self.undistorted = bool(processing.get("enabled") and camera in processing["cameras"])
        if self.undistorted:
            if processing["model"] != "equidistant":
                raise ValueError("only explicitly configured equidistant correction is supported")
            k = np.asarray(calibration["source_camera_matrix"], dtype=np.float64)
            d = np.asarray(processing["coefficients"][camera], dtype=np.float64)
            if d.shape != (4,) or not np.isfinite(d).all():
                raise ValueError("equidistant requires exactly four finite coefficients")
            # Slice full-resolution maps: bitwise equivalent to rectifying the
            # whole frame then slicing it, while remapping only the output ROI.
            mx, my = cv2.fisheye.initUndistortRectifyMap(
                k, d, np.eye(3), k, (self.width, self.height), cv2.CV_32FC1)
            self.map_x = mx[self.y:self.y + height, self.x:self.x + width].copy()
            self.map_y = my[self.y:self.y + height, self.x:self.x + width].copy()

    def __call__(self, bgr):
        if bgr.shape != (self.height, self.width, 3):
            raise ValueError("recorded image shape differs from calibrated ROI")
        if self.undistorted:
            return cv2.remap(bgr, self.map_x, self.map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        return bgr[self.y:self.y + self.image_size, self.x:self.x + self.image_size].copy()


def camera_processors(calibration, image_size):
    mode = calibration.get("image_preprocessing", {}).get("mode")
    if mode == "five_view":
        return {
            "hik_camera_6": WristUndistorter(calibration, "hik_camera_6", image_size),
            "hik_camera_7": WristUndistorter(calibration, "hik_camera_7", image_size),
            "hik_camera_6_crop": DirectCropper(calibration, "hik_camera_6", image_size),
            "hik_camera_7_crop": DirectCropper(calibration, "hik_camera_7", image_size),
        }
    if mode == "direct_crop":
        return {name: DirectCropper(calibration, name, image_size)
                for name in ("head", "hik_camera_6", "hik_camera_7")}
    if mode is not None:
        raise ValueError(f"unsupported image preprocessing mode: {mode}")
    processing = calibration.get("wrist_undistortion", {})
    if not processing.get("enabled", False):
        return {}
    if set(processing["cameras"]) != {"hik_camera_6", "hik_camera_7"}:
        raise ValueError("wrist correction must explicitly select cameras 6 and 7")
    return {name: WristUndistorter(calibration, name, image_size) for name in processing["cameras"]}
