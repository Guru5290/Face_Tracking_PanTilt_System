import cv2
import time
from collections import deque
from config import *
from stream_server import start_stream_server, set_latest_frame


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral   = 0

    def compute(self, error, dt=0.033):
        # Clamp integral to prevent windup — this is the main cause of drifting away
        self.integral += error * dt
        self.integral  = max(-INTEGRAL_MAX, min(INTEGRAL_MAX, self.integral))

        derivative      = (error - self.prev_error) / dt
        output          = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0
        self.integral   = 0


class FacePositionSmoother:
    """
    Averages face position over the last N frames to eliminate jitter
    caused by detection noise. Without this, the servo reacts to every
    tiny detection wobble and hunts back and forth.
    """
    def __init__(self, n=SMOOTH_FRAMES):
        self.cx_buf = deque(maxlen=n)
        self.cy_buf = deque(maxlen=n)

    def update(self, cx, cy):
        self.cx_buf.append(cx)
        self.cy_buf.append(cy)

    def get(self):
        if not self.cx_buf:
            return None, None
        return int(sum(self.cx_buf) / len(self.cx_buf)), \
               int(sum(self.cy_buf) / len(self.cy_buf))

    def reset(self):
        self.cx_buf.clear()
        self.cy_buf.clear()


class FaceCenteringMode:
    def __init__(self, servo):
        self.servo        = servo
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.pid_pan      = PIDController(PAN_KP,  PAN_KI,  PAN_KD)
        self.pid_tilt     = PIDController(TILT_KP, TILT_KI, TILT_KD)
        self.smoother     = FacePositionSmoother()

        # Track consecutive missed frames so we don't reset PID too eagerly
        self.missed_frames = 0
        self.MISS_TOLERANCE = 5  # frames before giving up and resetting PID
        self.face_lost_since = None
        self.pan_search_direction = 1
        self.tilt_search_direction = 1
        self.tilt_hold_counter = 0

    def _clamped_delta(self, raw_delta):
        limited = max(-MAX_SERVO_DELTA_PER_FRAME, min(MAX_SERVO_DELTA_PER_FRAME, raw_delta))
        if abs(limited) < SERVO_MIN_COMMAND_DELTA:
            return 0.0
        return limited

    def _search_step(self):
        """
        Patrol-like scan used while face is lost.
        Scans horizontally and periodically shifts tilt to cover more scene.
        """
        next_pan = self.servo.pan_angle + (PATROL_PAN_STEP * self.pan_search_direction)
        if next_pan >= PAN_MAX:
            next_pan = PAN_MAX
            self.pan_search_direction = -1
            self.tilt_hold_counter = 0
        elif next_pan <= PAN_MIN:
            next_pan = PAN_MIN
            self.pan_search_direction = 1
            self.tilt_hold_counter = 0
        self.servo.set_pan(next_pan)

        if self.tilt_hold_counter >= PATROL_TILT_HOLD_STEPS:
            next_tilt = self.servo.tilt_angle + (PATROL_TILT_STEP * self.tilt_search_direction)
            if next_tilt >= TILT_MAX:
                next_tilt = TILT_MAX
                self.tilt_search_direction = -1
            elif next_tilt <= TILT_MIN:
                next_tilt = TILT_MIN
                self.tilt_search_direction = 1
            self.servo.set_tilt(next_tilt)
            self.tilt_hold_counter = 0
        else:
            self.tilt_hold_counter += 1

    def run(self, cap, show_preview=True):
        print('Face Centering Active. Press Ctrl+C to exit.')

        if show_preview:
            start_stream_server(port=8080)

        self.servo.home()
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Camera read error.')
                break

            now = time.time()
            dt  = max(now - prev_time, 0.001)
            dt  = min(dt, 0.1)   # clamp dt: prevents huge derivative spike after lag
            prev_time = now

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor  = FACE_SCALE_FACTOR,
                minNeighbors = FACE_MIN_NEIGHBORS,
                minSize      = FACE_MIN_SIZE
            )

            if len(faces) > 0:
                self.missed_frames = 0
                self.face_lost_since = None

                # Track the largest face (closest person)
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_cx = x + w // 2
                face_cy = y + h // 2

                # Smooth position to remove detection jitter
                self.smoother.update(face_cx, face_cy)
                smooth_cx, smooth_cy = self.smoother.get()

                error_x = smooth_cx - FRAME_CENTER_X
                error_y = smooth_cy - FRAME_CENTER_Y

                # Only move if face is outside dead zone
                if abs(error_x) > DEAD_ZONE:
                    pan_delta = self._clamped_delta(self.pid_pan.compute(error_x, dt))
                    if pan_delta != 0.0:
                        self.servo.set_pan(self.servo.pan_angle + pan_delta)
                else:
                    # Inside dead zone — tell PID error is zero so integral doesn't build
                    self.pid_pan.compute(0, dt)

                if abs(error_y) > DEAD_ZONE:
                    tilt_delta = self._clamped_delta(self.pid_tilt.compute(error_y, dt))
                    if tilt_delta != 0.0:
                        self.servo.set_tilt(self.servo.tilt_angle + tilt_delta)
                else:
                    self.pid_tilt.compute(0, dt)

                if show_preview:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, (face_cx, face_cy), 5, (0, 255, 0), -1)
                    # Show smoothed target position
                    cv2.circle(frame, (smooth_cx, smooth_cy), 8, (255, 165, 0), 2)
                    # Show error values on frame for debugging
                    cv2.putText(frame, f'err x:{error_x:+.0f} y:{error_y:+.0f}',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                self.missed_frames += 1
                if self.face_lost_since is None:
                    self.face_lost_since = now
                # Only reset PID after several missed frames — prevents thrashing
                # when detection briefly drops a frame
                if (
                    self.missed_frames >= self.MISS_TOLERANCE
                    and (now - self.face_lost_since) >= FACE_LOST_HOLD_SECONDS
                ):
                    self.pid_pan.reset()
                    self.pid_tilt.reset()
                    self.smoother.reset()
                    self._search_step()

            if show_preview:
                cv2.circle(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 8, (0, 0, 255), 2)
                mode_text = 'Face Lock' if len(faces) > 0 else 'Face Lost - Scanning'
                cv2.putText(frame, mode_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                set_latest_frame(frame)
