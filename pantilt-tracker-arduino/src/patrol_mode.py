import cv2
import time
from config import *
from stream_server import start_stream_server, set_latest_frame


class PatrolMode:
    def __init__(self, servo):
        self.servo = servo

    def run(self, cap, show_preview=True):
        print('Patrol Mode Active. Press Ctrl+C to exit.')
        if show_preview:
            start_stream_server(port=8080)

        self.servo.home()

        pan = float(PAN_MIN)
        tilt = float(TILT_CENTER)
        pan_direction = 1
        tilt_direction = 1
        tilt_hold_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Camera read error.')
                break

            pan += PATROL_PAN_STEP * pan_direction
            if pan >= PAN_MAX:
                pan = float(PAN_MAX)
                pan_direction = -1
                tilt_hold_counter = 0
            elif pan <= PAN_MIN:
                pan = float(PAN_MIN)
                pan_direction = 1
                tilt_hold_counter = 0

            if tilt_hold_counter >= PATROL_TILT_HOLD_STEPS:
                tilt += PATROL_TILT_STEP * tilt_direction
                if tilt >= TILT_MAX:
                    tilt = float(TILT_MAX)
                    tilt_direction = -1
                elif tilt <= TILT_MIN:
                    tilt = float(TILT_MIN)
                    tilt_direction = 1
                tilt_hold_counter = 0
            else:
                tilt_hold_counter += 1

            self.servo.set_pan(pan)
            self.servo.set_tilt(tilt)

            if show_preview:
                cv2.putText(frame, 'Patrol Mode', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f'pan:{int(pan)} tilt:{int(tilt)}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                set_latest_frame(frame)

            time.sleep(PATROL_STEP_DELAY)
