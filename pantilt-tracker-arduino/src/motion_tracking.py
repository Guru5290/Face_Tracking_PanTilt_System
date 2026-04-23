import cv2
import time
from collections import deque
from config import *
from stream_server import start_stream_server, set_latest_frame


class MotionTrackingMode:
    def __init__(self, servo):
        self.servo = servo
        self.motion_history = deque(maxlen=MOTION_HISTORY)
        self.last_motion_ts = None

    def _idle_sweep(self):
        if self.servo.pan_angle >= PAN_MAX:
            self._sweep_direction = -1
        elif self.servo.pan_angle <= PAN_MIN:
            self._sweep_direction = 1
        target_pan = self.servo.pan_angle + (self._sweep_direction * 2.0)
        self.servo.set_pan(target_pan)

    def run(self, cap, show_preview=True):
        print('Motion Tracking Active. Press Ctrl+C to exit.')
        if show_preview:
            start_stream_server(port=8080)

        self.servo.home()
        self._sweep_direction = 1

        ret, first = cap.read()
        if not ret:
            print('Camera read error.')
            return

        prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Camera read error.')
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0)

            diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=MOTION_DILATE_ITERATIONS)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_targets = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_MOTION_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                motion_targets.append((x, y, w, h, area))

            status = 'No motion'
            if motion_targets:
                x, y, w, h, _ = max(motion_targets, key=lambda m: m[4])
                cx = x + (w // 2)
                cy = y + (h // 2)

                self.motion_history.append((cx, cy))
                smooth_cx = int(sum(p[0] for p in self.motion_history) / len(self.motion_history))
                smooth_cy = int(sum(p[1] for p in self.motion_history) / len(self.motion_history))

                error_x = smooth_cx - FRAME_CENTER_X
                error_y = smooth_cy - FRAME_CENTER_Y

                pan_step = max(-MAX_SERVO_DELTA_PER_FRAME, min(MAX_SERVO_DELTA_PER_FRAME, error_x * 0.02))
                tilt_step = max(-MAX_SERVO_DELTA_PER_FRAME, min(MAX_SERVO_DELTA_PER_FRAME, error_y * 0.02))

                if abs(pan_step) >= SERVO_MIN_COMMAND_DELTA:
                    self.servo.set_pan(self.servo.pan_angle + pan_step)
                if abs(tilt_step) >= SERVO_MIN_COMMAND_DELTA:
                    self.servo.set_tilt(self.servo.tilt_angle + tilt_step)

                self.last_motion_ts = time.time()
                status = 'Tracking motion'

                if show_preview:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
                    cv2.circle(frame, (smooth_cx, smooth_cy), 6, (0, 255, 255), -1)
            else:
                now = time.time()
                if self.last_motion_ts is None:
                    self.last_motion_ts = now
                elif (now - self.last_motion_ts) >= MOTION_IDLE_SWEEP_SECONDS:
                    self._idle_sweep()
                    status = 'Idle sweep'

            if show_preview:
                cv2.circle(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 8, (0, 0, 255), 2)
                cv2.putText(frame, 'Motion Tracking', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                set_latest_frame(frame)

            prev_gray = gray
