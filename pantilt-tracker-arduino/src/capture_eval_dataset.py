import argparse
import csv
import os
import time

import cv2

from camera_capture import open_camera, parse_v4l2_device, warmup_read
from config import (
    CAMERA_BACKEND,
    CAMERA_FLIP_METHOD,
    CAMERA_FPS,
    CSI_SENSOR_ID,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    V4L2_DEVICE,
)


def save_contact_sheet(image_paths, output_path, columns=4, thumb_w=320, thumb_h=180):
    if not image_paths:
        return
    rows = (len(image_paths) + columns - 1) // columns
    canvas = 255 * (cv2.UMat(rows * thumb_h, columns * thumb_w, cv2.CV_8UC3).get())

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        r = idx // columns
        c = idx % columns
        y1, y2 = r * thumb_h, (r + 1) * thumb_h
        x1, x2 = c * thumb_w, (c + 1) * thumb_w
        canvas[y1:y2, x1:x2] = thumb
        name = os.path.basename(path)
        cv2.rectangle(canvas, (x1, y2 - 24), (x2, y2), (0, 0, 0), -1)
        cv2.putText(canvas, name, (x1 + 5, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(output_path, canvas)


_ARGUS_HELP = """
CSI camera failed (Argus). Try:
  1) On the Jetson HOST (not inside Docker): sudo systemctl restart nvargus-daemon
  2) Stop other apps using the camera (another terminal, nvgstcamera, etc.).
  3) Match resolution/fps (often IMX219 works with --width 1920 --height 1080 --fps 30).
  4) Docker: mount the video device, e.g. --device /dev/video0 -v /dev/video0:/dev/video0
     then run with --camera v4l2 (CSI via Argus inside Docker is unreliable on some setups).
  5) Easiest: run capture on the host without Docker:
       python3 src/capture_eval_dataset.py --output-dir eval_images ...
  6) Re-plug CSI ribbon / cold boot if hardware is loose.
"""


def _open_and_warmup(backend, dev, w, h, f, flip, sid, warmup_sec):
    """Returns (cap, ok_frame). Caller must release cap when ok is True."""
    cap = open_camera(
        backend=backend,
        v4l2_device=dev,
        width=w,
        height=h,
        fps=f,
        flip=flip,
        sensor_id=sid,
    )
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None, False
    time.sleep(max(0.0, warmup_sec))
    ok, _probe = warmup_read(cap)
    return cap, ok


def capture_dataset(
    output_dir,
    prefix,
    max_images,
    interval_sec,
    warmup_sec,
    camera_backend=None,
    v4l2_device=None,
    width=None,
    height=None,
    fps=None,
    sensor_id=None,
    fallback_v4l2=True,
):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'capture_manifest.csv')
    image_paths = []

    backend = (camera_backend or CAMERA_BACKEND or 'csi').strip().lower()
    w = int(width if width is not None else FRAME_WIDTH)
    h = int(height if height is not None else FRAME_HEIGHT)
    f = int(fps if fps is not None else CAMERA_FPS)
    sid = int(sensor_id if sensor_id is not None else CSI_SENSOR_ID)
    dev = parse_v4l2_device(v4l2_device if v4l2_device is not None else V4L2_DEVICE)

    print(f'Camera backend: {backend} (CSI sensor-id={sid}, V4L2 device={dev}) resolution {w}x{h}@{f}')

    cap, ok = _open_and_warmup(
        backend, dev, w, h, f, CAMERA_FLIP_METHOD, sid, warmup_sec
    )

    if not ok and cap is not None:
        cap.release()
        cap = None

    if not ok and backend == 'csi' and fallback_v4l2:
        print('CSI/Argus did not produce frames — trying V4L2 fallback (same resolution)...')
        backend = 'v4l2'
        cap, ok = _open_and_warmup(
            'v4l2', dev, w, h, f, CAMERA_FLIP_METHOD, sid, warmup_sec
        )
        if not ok and cap is not None:
            cap.release()
            cap = None

    if cap is None or not ok:
        if cap is not None:
            cap.release()
        print(_ARGUS_HELP)
        raise RuntimeError(
            'Failed to capture: camera did not open or no frames arrived.'
        )

    print(f'Capture using backend: {backend}. Starting...')

    with open(manifest_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filename', 'timestamp_unix', 'capture_idx'])

        print('Press Ctrl+C to stop early.')
        try:
            for idx in range(max_images):
                ret, frame = cap.read()
                if not ret:
                    print('Camera read failed, stopping capture.')
                    break

                ts = time.time()
                filename = f'{prefix}_{idx:04d}.jpg'
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, frame)
                writer.writerow([filename, f'{ts:.3f}', idx])
                image_paths.append(path)
                print(f'Captured {filename}')

                if idx < max_images - 1:
                    time.sleep(max(0.0, interval_sec))
        except KeyboardInterrupt:
            print('\nCapture interrupted by user.')
        finally:
            cap.release()

    sheet_path = os.path.join(output_dir, 'contact_sheet.jpg')
    save_contact_sheet(image_paths, sheet_path)
    print(f'Capture complete: {len(image_paths)} images')
    print(f'Manifest: {manifest_path}')
    print(f'Contact sheet: {sheet_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Capture an evaluation image set from Jetson camera.')
    parser.add_argument('--output-dir', default='eval_images', help='Directory where images are stored')
    parser.add_argument('--prefix', default='frame', help='Image filename prefix')
    parser.add_argument('--max-images', type=int, default=120, help='Maximum number of frames to capture')
    parser.add_argument('--interval-sec', type=float, default=0.4, help='Seconds between captures')
    parser.add_argument('--warmup-sec', type=float, default=1.0, help='Camera warmup before capture')
    parser.add_argument(
        '--camera',
        choices=('csi', 'v4l2'),
        default=None,
        help='csi = PiCamera CSI via nvarguscamerasrc (default from CAMERA_BACKEND env); v4l2 = /dev/video*',
    )
    parser.add_argument('--v4l2-device', default=None, help='V4L2 index (0) or path (/dev/video0)')
    parser.add_argument('--width', type=int, default=None, help='Frame width (default from config)')
    parser.add_argument('--height', type=int, default=None, help='Frame height')
    parser.add_argument('--fps', type=int, default=None, help='Frames per second for pipeline')
    parser.add_argument('--sensor-id', type=int, default=None, help='CSI sensor-id for nvarguscamerasrc')
    parser.add_argument(
        '--no-fallback-v4l2',
        action='store_true',
        help='If CSI fails, do not automatically retry with V4L2',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    capture_dataset(
        output_dir=args.output_dir,
        prefix=args.prefix,
        max_images=max(1, args.max_images),
        interval_sec=max(0.0, args.interval_sec),
        warmup_sec=max(0.0, args.warmup_sec),
        camera_backend=args.camera,
        v4l2_device=args.v4l2_device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sensor_id=args.sensor_id,
        fallback_v4l2=not args.no_fallback_v4l2,
    )
