import cv2
import dlib
import time
from collections import deque
from config import *
from stream_server import start_stream_server, set_latest_frame


def _draw_bold_fps_overlay(frame_bgr, text):
    """
    Large label bottom-right on the MJPEG preview. OpenCV has no real bold fonts;
    use a heavy font, thick stroke, and a dark outline (high contrast vs. scene).
    org is bottom-left of the text in OpenCV.
    """
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.65
    thick = 5
    color = (255, 255, 255)
    margin_x, margin_y = 14, 20
    (tw, th), _baseline = cv2.getTextSize(text, font, scale, thick)
    x = max(8, w - tw - margin_x)
    y = max(th + 8, h - margin_y)
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, -2), (-2, 2), (2, 2)):
        cv2.putText(
            frame_bgr, text, (x + dx, y + dy), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA,
        )
    cv2.putText(frame_bgr, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


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
        self.servo = servo
        self.detector_backend = DETECTOR_BACKEND if DETECTOR_BACKEND in ('dlib', 'haar', 'opencv_dnn') else 'dlib'
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dnn_net = cv2.dnn.readNetFromCaffe(OPENCV_DNN_PROTOTXT_PATH, OPENCV_DNN_MODEL_PATH)
        self.dnn_runtime = 'cpu'
        if OPENCV_DNN_TARGET == 'cuda':
            try:
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.dnn_runtime = 'cuda'
            except Exception:
                # Fallback for OpenCV builds without CUDA DNN support.
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.dnn_runtime = 'cpu-fallback'
        else:
            self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.landmark_predictor = dlib.shape_predictor(DLIB_LANDMARK_MODEL_PATH)
        self.pid_pan = PIDController(PAN_KP, PAN_KI, PAN_KD)
        self.pid_tilt = PIDController(TILT_KP, TILT_KI, TILT_KD)
        self.smoother = FacePositionSmoother()

        # Track consecutive missed frames so we don't reset PID too eagerly
        self.missed_frames = 0
        self.MISS_TOLERANCE = 5  # frames before giving up and resetting PID
        self.face_lost_since = None
        self.pan_search_direction = 1
        self.tilt_search_direction = 1
        self.tilt_hold_counter = 0
        self.filtered_error_x = 0.0
        self.filtered_error_y = 0.0
        self.pan_axis_moving = False
        self.tilt_axis_moving = False

    def _clamped_delta(self, raw_delta, max_delta, min_cmd_delta):
        limited = max(-max_delta, min(max_delta, raw_delta))
        if abs(limited) < min_cmd_delta:
            return 0.0
        return limited

    def _error_filter(self, raw, previous, alpha):
        alpha = max(0.0, min(0.98, alpha))
        return (alpha * previous) + ((1.0 - alpha) * raw)

    def _update_axis_state(self, is_moving, error_abs, hold_zone, break_zone):
        hold = max(0, hold_zone)
        brk = max(hold + 1, break_zone)
        if is_moving and error_abs <= hold:
            return False
        if (not is_moving) and error_abs >= brk:
            return True
        return is_moving

    def _detect_faces(self, frame_bgr, gray):
        if self.detector_backend == 'haar':
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_SCALE_FACTOR,
                minNeighbors=FACE_MIN_NEIGHBORS,
                minSize=FACE_MIN_SIZE
            )
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        if self.detector_backend == 'opencv_dnn':
            h, w = frame_bgr.shape[:2]
            blob = cv2.dnn.blobFromImage(
                frame_bgr,
                scalefactor=1.0,
                size=(300, 300),
                mean=(104.0, 177.0, 123.0),
                swapRB=False,
                crop=False,
            )
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < OPENCV_DNN_CONFIDENCE_THRESHOLD:
                    continue
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                faces.append((x1, y1, bw, bh))
            return faces

        rects = self.dlib_detector(gray, DLIB_HOG_UPSAMPLE)
        out = []
        for rect in rects:
            x = max(rect.left(), 0)
            y = max(rect.top(), 0)
            w = max(rect.right() - x, 1)
            h = max(rect.bottom() - y, 1)
            out.append((int(x), int(y), int(w), int(h)))
        return out

    def _extract_landmarks(self, gray, face_box):
        x, y, w, h = face_box
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self.landmark_predictor(gray, rect)
        return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    def _search_step(self):
        """
        Patrol-like scan used while face is lost.
        Scans horizontally and periodically shifts tilt to cover more scene.
        """
        next_pan = self.servo.pan_angle + (SEARCH_PAN_STEP * self.pan_search_direction)
        if next_pan >= PAN_MAX:
            next_pan = PAN_MAX
            self.pan_search_direction = -1
            self.tilt_hold_counter = 0
        elif next_pan <= PAN_MIN:
            next_pan = PAN_MIN
            self.pan_search_direction = 1
            self.tilt_hold_counter = 0
        self.servo.set_pan(next_pan)

        if self.tilt_hold_counter >= SEARCH_TILT_HOLD_STEPS:
            next_tilt = self.servo.tilt_angle + (SEARCH_TILT_STEP * self.tilt_search_direction)
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

    def _detector_tag(self):
        if self.detector_backend == 'opencv_dnn':
            return f'opencv_dnn_{self.dnn_runtime}'
        return self.detector_backend

    def check_static_image(self, image_path, save_path=None, show_window=False):
        """
        Run face detection + 68-point landmarks on a single image (no camera, no servos).
        Uses the image's own width/height for the reference center crosshair.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')

        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(frame, gray)

        runtime_label = self.detector_backend.upper()
        if self.detector_backend == 'opencv_dnn':
            runtime_label = f'OPENCV_DNN/{self.dnn_runtime.upper()}'
        print(f'Static image check ({runtime_label}): {image_path}')
        print(f'Image size: {w}x{h}, faces detected: {len(faces)}')

        if len(faces) > 0:
            x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
            landmarks = self._extract_landmarks(gray, (x, y, bw, bh))
            face_cx = x + bw // 2
            face_cy = y + bh // 2
            error_x = face_cx - cx0
            error_y = face_cy - cy0
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(frame, (face_cx, face_cy), 5, (0, 255, 0), -1)
            for px, py in landmarks:
                cv2.circle(frame, (px, py), 3, (0, 0, 0), -1)
                cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)
            print(f'Largest face bbox: ({x},{y}) {bw}x{bh}')
            print(f'Center offset (pixels from image center): x={error_x:+d} y={error_y:+d}')

        cv2.circle(frame, (cx0, cy0), 8, (0, 0, 255), 2)
        cv2.putText(
            frame, 'Static image check', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
        )
        if len(faces) == 0:
            cv2.putText(
                frame, 'No face', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
            )
        else:
            cv2.putText(
                frame,
                f'err vs center (px): x={error_x:+d} y={error_y:+d}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
        det_text = f'Detector: {self.detector_backend.upper()}'
        if self.detector_backend == 'opencv_dnn':
            det_text += f' ({self.dnn_runtime.upper()})'
        cv2.putText(frame, det_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        if save_path:
            if not cv2.imwrite(save_path, frame):
                raise OSError(f'Failed to write: {save_path}')
            print(f'Annotated image written to: {save_path}')
        if show_window:
            cv2.imshow('Static image check', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif save_path is None:
            print('Tip: use --static-save PATH to write an annotated JPEG (useful without a display).')

    def run(self, cap, show_preview=True, live_eval_session=None, print_fps=False):
        runtime_label = self.detector_backend.upper()
        if self.detector_backend == 'opencv_dnn':
            runtime_label = f'OPENCV_DNN/{self.dnn_runtime.upper()}'
        print(f'Face Centering Active ({runtime_label}). Press Ctrl+C to exit.')

        if show_preview:
            start_stream_server(port=8080)

        self.servo.home()
        prev_time = time.time()
        fps_loop_count = 0
        fps_window_start = time.time()
        self._preview_fps_display = None  # None until first 1s measurement window completes

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Camera read error.')
                break

            frame_clean = frame.copy()

            now = time.time()
            dt  = max(now - prev_time, 0.001)
            dt  = min(dt, 0.1)   # clamp dt: prevents huge derivative spike after lag
            prev_time = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces(frame, gray)
            bbox_main = None

            if len(faces) > 0:
                self.missed_frames = 0
                self.face_lost_since = None

                # Track the largest face (closest person)
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                bbox_main = (x, y, w, h)
                landmarks = self._extract_landmarks(gray, (x, y, w, h))
                face_cx = x + w // 2
                face_cy = y + h // 2

                # Smooth position to remove detection jitter
                self.smoother.update(face_cx, face_cy)
                smooth_cx, smooth_cy = self.smoother.get()

                error_x = smooth_cx - FRAME_CENTER_X
                error_y = smooth_cy - FRAME_CENTER_Y
                self.filtered_error_x = self._error_filter(error_x, self.filtered_error_x, PAN_ERROR_FILTER_ALPHA)
                self.filtered_error_y = self._error_filter(error_y, self.filtered_error_y, TILT_ERROR_FILTER_ALPHA)

                self.pan_axis_moving = self._update_axis_state(
                    self.pan_axis_moving,
                    abs(self.filtered_error_x),
                    PAN_LOCK_HOLD_ZONE,
                    PAN_LOCK_BREAK_ZONE
                )
                if self.pan_axis_moving:
                    pan_delta = self._clamped_delta(
                        self.pid_pan.compute(self.filtered_error_x, dt),
                        PAN_MAX_SERVO_DELTA_PER_FRAME,
                        PAN_SERVO_MIN_COMMAND_DELTA
                    )
                    if pan_delta != 0.0:
                        self.servo.set_pan(self.servo.pan_angle + pan_delta)
                else:
                    self.pid_pan.compute(0, dt)

                self.tilt_axis_moving = self._update_axis_state(
                    self.tilt_axis_moving,
                    abs(self.filtered_error_y),
                    TILT_LOCK_HOLD_ZONE,
                    TILT_LOCK_BREAK_ZONE
                )
                if self.tilt_axis_moving:
                    tilt_delta = self._clamped_delta(
                        self.pid_tilt.compute(self.filtered_error_y, dt),
                        TILT_MAX_SERVO_DELTA_PER_FRAME,
                        TILT_SERVO_MIN_COMMAND_DELTA
                    )
                    if tilt_delta != 0.0:
                        self.servo.set_tilt(self.servo.tilt_angle + tilt_delta)
                else:
                    self.pid_tilt.compute(0, dt)

                if show_preview:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, (face_cx, face_cy), 5, (0, 255, 0), -1)
                    for px, py in landmarks:
                        # Dual-pass draw for strong contrast on any background.
                        cv2.circle(frame, (px, py), 3, (0, 0, 0), -1)
                        cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)
                    # Show smoothed target position
                    cv2.circle(frame, (smooth_cx, smooth_cy), 8, (255, 165, 0), 2)
                    # Show error values on frame for debugging
                    cv2.putText(frame, f'err x:{error_x:+.0f} y:{error_y:+.0f}',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f'filt x:{self.filtered_error_x:+.0f} y:{self.filtered_error_y:+.0f}',
                        (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 220, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f'axis pan:{"MOVE" if self.pan_axis_moving else "HOLD"} '
                        f'tilt:{"MOVE" if self.tilt_axis_moving else "HOLD"}',
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (120, 255, 120),
                        2
                    )

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
                    self.filtered_error_x = 0.0
                    self.filtered_error_y = 0.0
                    self.pan_axis_moving = False
                    self.tilt_axis_moving = False
                    self._search_step()
                    time.sleep(SEARCH_STEP_DELAY)

            if live_eval_session is not None:
                live_eval_session.maybe_save(
                    frame_clean,
                    self.servo.pan_angle,
                    self.servo.tilt_angle,
                    self._detector_tag(),
                    bbox_main is not None,
                    bbox_main,
                )

            if print_fps:
                fps_loop_count += 1
                t_fps = time.time()
                win = t_fps - fps_window_start
                if win >= 1.0:
                    self._preview_fps_display = fps_loop_count / win
                    print(f'Processing FPS: {self._preview_fps_display:.1f}')
                    fps_loop_count = 0
                    fps_window_start = t_fps

            if show_preview:
                cv2.circle(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 8, (0, 0, 255), 2)
                mode_text = 'Face Lock' if len(faces) > 0 else 'Face Lost - Scanning'
                cv2.putText(frame, mode_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                detector_text = f'Detector: {self.detector_backend.upper()}'
                if self.detector_backend == 'opencv_dnn':
                    detector_text += f' ({self.dnn_runtime.upper()})'
                cv2.putText(frame, detector_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                if print_fps:
                    if self._preview_fps_display is not None:
                        _draw_bold_fps_overlay(frame, f'{self._preview_fps_display:.1f} FPS')
                    else:
                        _draw_bold_fps_overlay(frame, '--- FPS')
                set_latest_frame(frame)
