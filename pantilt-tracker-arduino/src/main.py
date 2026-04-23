import cv2
import sys
import os
import urllib.request
from servo_controller import ServoController
from face_centering  import FaceCenteringMode
from patrol_mode     import PatrolMode
from motion_tracking import MotionTrackingMode
from config import *

CASCADE_URL = ('https://raw.githubusercontent.com/opencv/opencv/master/'
               'data/haarcascades/haarcascade_frontalface_default.xml')

def download_cascade():
    if not os.path.exists(FACE_CASCADE_PATH):
        print('Downloading face cascade classifier ')
        urllib.request.urlretrieve(CASCADE_URL, FACE_CASCADE_PATH)
        print('done.')

def gstreamer_pipeline(
    width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30,
    flip=CAMERA_FLIP_METHOD
):
    if flip not in range(8):
        print(f'Invalid CAMERA_FLIP_METHOD={flip}, using 0')
        flip = 0
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink max-buffers=1 drop=true"
    )

def main():
    download_cascade()

    print('\nPan-Tilt Tracking System')
    print('1) Face Lock + Auto Scan Recovery')
    print('2) Patrol Mode')
    print('3) Motion Tracking')

    choice = input('Select mode 1, 2 or 3: ').strip()
    if choice not in ('1', '2', '3'):
        print('Invalid choice. Exiting.')
        sys.exit(1)

    # Here the camera is opened using GStreamer and nvarguscamerasrc for
    # CSI camera using Jetson ISP via Argus
    print(f'Camera flip-method: {CAMERA_FLIP_METHOD}')
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print('The camera cannot be opened')
        sys.exit(1)

    # Verify a real frame comes through
    ret, test_frame = cap.read()
    print(f"Camera test: ret={ret}, shape={test_frame.shape if ret else 'None'}")
    if ret and test_frame.mean() < 5:
        print("WARNING: Frame is nearly black/green -- pixel format issue")

    # ServoController now communicates with Arduino Mega over USB serial
    servo = ServoController()

    try:
        show = '--no-preview' not in sys.argv
        if choice == '1':
            FaceCenteringMode(servo).run(cap, show)
        elif choice == '2':
            PatrolMode(servo).run(cap, show)
        elif choice == '3':
            MotionTrackingMode(servo).run(cap, show)
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        cap.release()
        servo.cleanup()
        print('Servo moved to home position and serial port released.')

if __name__ == '__main__':
    main()
