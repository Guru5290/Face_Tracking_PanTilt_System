import os

# --- Arduino Serial Settings ---
ARDUINO_PORT = None       # Set to e.g. '/dev/ttyACM0' or leave None for auto-detect
ARDUINO_BAUD = 115200     # Higher baud = faster commands = tighter tracking

# --- Servo Angle Limits ---
PAN_MIN    = 0
PAN_MAX    = 180
PAN_CENTER = 90

TILT_MIN    = 40
TILT_MAX    = 140
TILT_CENTER = 90

# --- Camera / Frame Settings ---
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
FRAME_CENTER_X = FRAME_WIDTH  // 2
FRAME_CENTER_Y = FRAME_HEIGHT // 2
# GStreamer nvvidconv flip-method:
# 0=none, 1=counterclockwise 90, 2=rotate 180, 3=clockwise 90,
# 4=horizontal mirror, 5=upper-right diagonal, 6=vertical mirror, 7=upper-left diagonal
CAMERA_FLIP_METHOD = int(os.getenv('CAMERA_FLIP_METHOD', '0'))
# Camera input: "csi" = nvarguscamerasrc (PiCamera module), "v4l2" = USB or /dev/video*
CAMERA_BACKEND = os.getenv('CAMERA_BACKEND', 'csi').strip().lower()
V4L2_DEVICE = os.getenv('V4L2_DEVICE', '0').strip()
CSI_SENSOR_ID = int(os.getenv('CSI_SENSOR_ID', '0'))
CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
# When true, log processing FPS and draw it on the MJPEG preview (same as --print-fps).
PRINT_FPS = os.getenv('PRINT_FPS', '').strip().lower() in ('1', 'true', 'yes')

# --- Detection backend ---
# Options: "dlib", "haar", "opencv_dnn" (Jetson GPU-capable via CUDA backend)
DETECTOR_BACKEND = os.getenv('DETECTOR_BACKEND', 'dlib').strip().lower()

# --- PID Tuning ---
# KP: how aggressively to chase the face. Higher = faster pursuit, too high = oscillation
# KI: corrects for steady-state offset. Keep very small to avoid windup drift
# KD: dampens overshoot. Prevents the servo overshooting and hunting back and forth
PAN_KP  = 0.05
PAN_KI  = 0.0
PAN_KD  = 0.004

TILT_KP = 0.06
TILT_KI = 0.0
TILT_KD = 0.005

# --- Dead zone: pixels from center to ignore (prevents micro-jitter when face is centred) ---
DEAD_ZONE = 40
# Lock hysteresis for "hold still unless subject really moved"
LOCK_HOLD_ZONE = int(os.getenv('LOCK_HOLD_ZONE', str(DEAD_ZONE)))
LOCK_BREAK_ZONE = int(os.getenv('LOCK_BREAK_ZONE', str(DEAD_ZONE + 18)))
PAN_LOCK_HOLD_ZONE = int(os.getenv('PAN_LOCK_HOLD_ZONE', str(LOCK_HOLD_ZONE)))
PAN_LOCK_BREAK_ZONE = int(os.getenv('PAN_LOCK_BREAK_ZONE', str(LOCK_BREAK_ZONE)))
TILT_LOCK_HOLD_ZONE = int(os.getenv('TILT_LOCK_HOLD_ZONE', str(LOCK_HOLD_ZONE + 6)))
TILT_LOCK_BREAK_ZONE = int(os.getenv('TILT_LOCK_BREAK_ZONE', str(LOCK_BREAK_ZONE + 12)))

# --- Integral windup clamp: prevents integral from accumulating too much ---
INTEGRAL_MAX = 30.0

# --- Face detection smoothing: number of frames to average face position over ---
SMOOTH_FRAMES = 4
# Exponential filter on tracking error (higher = smoother, lower = more reactive)
ERROR_FILTER_ALPHA = float(os.getenv('ERROR_FILTER_ALPHA', '0.72'))
PAN_ERROR_FILTER_ALPHA = float(os.getenv('PAN_ERROR_FILTER_ALPHA', str(ERROR_FILTER_ALPHA)))
TILT_ERROR_FILTER_ALPHA = float(os.getenv('TILT_ERROR_FILTER_ALPHA', '0.84'))

# Servo output smoothing (stability controls)
MAX_SERVO_DELTA_PER_FRAME = 3.5
SERVO_MIN_COMMAND_DELTA   = 0.45
PAN_MAX_SERVO_DELTA_PER_FRAME = float(os.getenv('PAN_MAX_SERVO_DELTA_PER_FRAME', str(MAX_SERVO_DELTA_PER_FRAME)))
TILT_MAX_SERVO_DELTA_PER_FRAME = float(os.getenv('TILT_MAX_SERVO_DELTA_PER_FRAME', '2.8'))
PAN_SERVO_MIN_COMMAND_DELTA = float(os.getenv('PAN_SERVO_MIN_COMMAND_DELTA', str(SERVO_MIN_COMMAND_DELTA)))
TILT_SERVO_MIN_COMMAND_DELTA = float(os.getenv('TILT_SERVO_MIN_COMMAND_DELTA', '0.70'))
# Serial command debouncing to reduce physical servo chatter
SERVO_SEND_MIN_INTERVAL_SEC = float(os.getenv('SERVO_SEND_MIN_INTERVAL_SEC', '0.045'))
PAN_SEND_MIN_STEP_DEG = int(os.getenv('PAN_SEND_MIN_STEP_DEG', '1'))
TILT_SEND_MIN_STEP_DEG = int(os.getenv('TILT_SEND_MIN_STEP_DEG', '2'))

# If face is lost briefly, hold last target before resetting control
FACE_LOST_HOLD_SECONDS = 1.0

# --- Face-lost scan behavior ---
SEARCH_PAN_STEP   = 3
SEARCH_TILT_STEP  = 5
SEARCH_STEP_DELAY = 0.04
SEARCH_TILT_HOLD_STEPS = 25

# --- Face Detection Settings ---
FACE_CASCADE_PATH  = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
FACE_SCALE_FACTOR  = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE      = (80, 80)

# --- Dlib settings ---
DLIB_HOG_UPSAMPLE = int(os.getenv('DLIB_HOG_UPSAMPLE', '1'))
DLIB_LANDMARK_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'shape_predictor_68_face_landmarks.dat'
)
DLIB_LANDMARK_MODEL_URL = (
    'https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
)

# --- OpenCV DNN face detector settings (ResNet SSD) ---
OPENCV_DNN_PROTOTXT_PATH = os.path.join(
    os.path.dirname(__file__),
    'deploy.prototxt'
)
OPENCV_DNN_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'res10_300x300_ssd_iter_140000.caffemodel'
)
OPENCV_DNN_CONFIDENCE_THRESHOLD = float(os.getenv('OPENCV_DNN_CONFIDENCE_THRESHOLD', '0.55'))
# "cuda" -> DNN_BACKEND_CUDA + DNN_TARGET_CUDA, "cpu" -> OpenCV default backend/target
OPENCV_DNN_TARGET = os.getenv('OPENCV_DNN_TARGET', 'cuda').strip().lower()
