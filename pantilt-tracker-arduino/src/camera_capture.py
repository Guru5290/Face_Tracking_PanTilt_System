"""
Shared camera opening for Jetson CSI (Argus/GStreamer) vs V4L2 (/dev/video*).
"""
import os
import time

import cv2

from config import CAMERA_FLIP_METHOD, FRAME_HEIGHT, FRAME_WIDTH


def build_csi_gstreamer_pipeline(
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
    fps=30,
    flip=CAMERA_FLIP_METHOD,
    sensor_id=0,
):
    if flip not in range(8):
        flip = 0
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )


def open_v4l2(device):
    """device: int index or string like '/dev/video0'."""
    if isinstance(device, str) and device.startswith('/'):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    else:
        idx = int(device)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    return cap


def open_camera(
    backend='csi',
    v4l2_device=0,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
    fps=30,
    flip=CAMERA_FLIP_METHOD,
    sensor_id=0,
):
    """
    Open VideoCapture. backend: 'csi' | 'v4l2'
    """
    backend = (backend or 'csi').strip().lower()
    if backend == 'v4l2':
        cap = open_v4l2(v4l2_device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        return cap

    pipe = build_csi_gstreamer_pipeline(
        width=width, height=height, fps=fps, flip=flip, sensor_id=sensor_id
    )
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    return cap


def warmup_read(cap, max_attempts=60, delay_sec=0.05):
    """
    Discard bad first frames; Argus/OpenCV sometimes returns empty once or twice.
    """
    for _ in range(max_attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(delay_sec)
    return False, None


def parse_v4l2_device(value):
    if value is None:
        return 0
    s = str(value).strip()
    if s.startswith('/dev/'):
        return s
    return int(s)
