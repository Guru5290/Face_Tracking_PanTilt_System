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
DEAD_ZONE = 15

# --- Integral windup clamp: prevents integral from accumulating too much ---
INTEGRAL_MAX = 30.0

# --- Face detection smoothing: number of frames to average face position over ---
SMOOTH_FRAMES = 4

# Servo output smoothing (stability controls)
MAX_SERVO_DELTA_PER_FRAME = 3.5
SERVO_MIN_COMMAND_DELTA   = 0.25

# If face is lost briefly, hold last target before resetting control
FACE_LOST_HOLD_SECONDS = 5.0

# --- Patrol Mode Settings ---
PATROL_PAN_STEP   = 3
PATROL_TILT_STEP  = 5
PATROL_STEP_DELAY = 0.04
PATROL_TILT_HOLD_STEPS = 25

# --- Motion Tracking Settings ---
MIN_MOTION_AREA = 1500
MOTION_HISTORY  = 5
MOTION_BLUR_SIZE = 21
MOTION_THRESHOLD = 25
MOTION_DILATE_ITERATIONS = 2
MOTION_IDLE_SWEEP_SECONDS = 4.0

# --- Face Detection Settings ---
FACE_CASCADE_PATH  = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
FACE_SCALE_FACTOR  = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE      = (140, 140)
