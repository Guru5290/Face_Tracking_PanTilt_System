import argparse
import sys

from servo_controller import ServoController

try:
    from face_centering import FaceCenteringMode
except ModuleNotFoundError as exc:
    if getattr(exc, 'name', '') == 'dlib' or 'dlib' in str(exc):
        sys.stderr.write(
            '\nMissing Python package: dlib (required for landmarks and dlib face detector).\n\n'
            'Fix one of:\n'
            '  • Jetson host: sudo apt-get install -y cmake build-essential '
            'libopenblas-base liblapack-dev && pip3 install dlib\n'
            '  • Use the project Docker image (dlib is built there): see README "Build and Run (Docker)".\n\n'
        )
    raise
from config import *
from model_assets import ensure_runtime_models
from camera_capture import open_camera, parse_v4l2_device, warmup_read
from live_eval_capture import LiveEvalSession


def parse_args():
    parser = argparse.ArgumentParser(description='Pan-tilt face tracking with optional live eval capture.')
    parser.add_argument('--no-preview', action='store_true', help='Disable MJPEG preview stream')
    parser.add_argument(
        '--eval-session',
        action='store_true',
        help='Save clean camera frames during tracking (same session as tracker) for detector benchmarking',
    )
    parser.add_argument('--eval-dir', default='eval_sessions', help='Base folder for live capture sessions')
    parser.add_argument('--eval-every-n', type=int, default=15, help='Save one frame every N tracker loops')
    parser.add_argument('--eval-max-frames', type=int, default=400, help='Maximum frames to save per session')
    parser.add_argument(
        '--static-image',
        metavar='PATH',
        default=None,
        help='Run detector + landmarks on one image only (no camera, no Arduino). Use --static-save or --static-show.',
    )
    parser.add_argument(
        '--static-save',
        metavar='PATH',
        default=None,
        help='Write annotated JPEG from --static-image (recommended on headless / Docker)',
    )
    parser.add_argument(
        '--static-show',
        action='store_true',
        help='Open an OpenCV window for --static-image (needs a display)',
    )
    parser.add_argument(
        '--print-fps',
        action='store_true',
        help='Print processing FPS every second and show it on the MJPEG preview (or set PRINT_FPS=1)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_runtime_models()

    print('\nPan-Tilt Face Tracking System')

    if args.static_image:
        if args.eval_session:
            print('Note: --eval-session is ignored with --static-image.', file=sys.stderr)
        print('Mode: static image check (no camera, no Arduino)')
        try:
            FaceCenteringMode(None).check_static_image(
                args.static_image,
                save_path=args.static_save,
                show_window=args.static_show,
            )
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        return

    print('Mode: Face Lock + Auto Scan Recovery')

    print(f'Camera backend: {CAMERA_BACKEND}, flip-method: {CAMERA_FLIP_METHOD}')
    cap = open_camera(
        backend=CAMERA_BACKEND,
        v4l2_device=parse_v4l2_device(V4L2_DEVICE),
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=CAMERA_FPS,
        flip=CAMERA_FLIP_METHOD,
        sensor_id=CSI_SENSOR_ID,
    )

    if not cap.isOpened():
        print('The camera cannot be opened. Set CAMERA_BACKEND=v4l2 or fix CSI/Argus.')
        sys.exit(1)

    ret, test_frame = warmup_read(cap)
    print(f"Camera test: ret={ret}, shape={test_frame.shape if ret else 'None'}")
    if not ret:
        print('No valid frames from camera. Try CAMERA_BACKEND=v4l2 or restart nvargus-daemon.')
        cap.release()
        sys.exit(1)
    if ret and test_frame.mean() < 5:
        print("WARNING: Frame is nearly black/green -- pixel format issue")

    servo = ServoController()

    live_eval = None
    if args.eval_session:
        live_eval = LiveEvalSession(
            base_dir=args.eval_dir,
            every_n_frames=max(1, args.eval_every_n),
            max_frames=max(1, args.eval_max_frames),
        )

    try:
        FaceCenteringMode(servo).run(
            cap,
            show_preview=not args.no_preview,
            live_eval_session=live_eval,
            print_fps=args.print_fps or PRINT_FPS,
        )
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        if live_eval is not None:
            live_eval.close()
        cap.release()
        servo.cleanup()
        print('Servo moved to home position and serial port released.')

if __name__ == '__main__':
    main()
