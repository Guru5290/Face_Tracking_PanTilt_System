# Pan-Tilt Face Tracking Camera System

This project is now focused only on face tracking for reporting and evaluation.
It supports multiple face detectors (dlib HOG+SVM, Haar, OpenCV DNN), with dlib landmarks for robust tracking visualization.

## What Changed

- Removed motion tracking and patrol-only modes from the app flow
- Added dlib HOG + SVM and OpenCV DNN detector backends for runtime tracking
- Added dlib 68-point landmarks overlay for report screenshots
- Kept Haar cascade as a benchmark baseline only
- Added a detector comparison script with metrics + chart + qualitative images

## Core Runtime Behavior

- Face found: lock and track with PID pan-tilt control
- Face lost: auto-scan sweep
- Face found again: immediate relock and continue tracking

## Detection Stack

- Runtime detector options:
  - `dlib`: dlib HOG + Linear SVM (CPU baseline)
  - `opencv_dnn`: OpenCV SSD face detector (Jetson GPU-capable with CUDA backend)
  - `haar`: Haar cascade baseline
- Landmarks: dlib 68-point predictor (`shape_predictor_68_face_landmarks.dat`)
- Baseline comparators for results section: OpenCV Haar + dlib HOG + OpenCV DNN

## Repository Layout

- `src/main.py`: face tracking entrypoint
- `src/face_centering.py`: dlib face lock + landmarks + scan recovery
- `src/servo_controller.py`: serial commands to Arduino
- `src/stream_server.py`: MJPEG preview stream
- `src/camera_capture.py`: open CSI (Argus) or V4L2 camera consistently
- `src/model_assets.py`: runtime model download/decompression
- `src/live_eval_capture.py`: save frames **during live tracking** for benchmarking
- `src/capture_eval_dataset.py`: optional static-only capture (no servos / second camera session)
- `src/prelabel_dataset.py`: auto-generate initial labels.csv
- `src/review_labels.py`: fast OpenCV label reviewer/editor
- `src/benchmark_detectors.py`: Haar vs dlib benchmark + plots
- `src/config.py`: tuning and detection settings
- `arduino/pan_tilt_servo.ino`: Arduino firmware

Dependency files:

- `requirements.txt` — tracker stack (`pyserial`, `imutils`, `dlib`, …); **no pip NumPy/Matplotlib** (use APT on Jetson per Troubleshooting).
- `requirements-report.txt` — optional `matplotlib` for benchmark plots (also satisfied by `python3-matplotlib`).

## Hardware Requirements

- NVIDIA Jetson board + CSI camera
- Arduino Mega (or compatible USB-serial Arduino)
- 2 servos (pan + tilt)
- External 5V servo supply (recommended)
- Common ground between Arduino and servo supply

## Arduino Setup

1. Open `arduino/pan_tilt_servo.ino`
2. Upload to Arduino Mega
3. Ensure baud = `115200`

## Build and Run (Docker)

```bash
docker build -t pantilt:latest .
```

Run face tracking:

```bash
docker run --rm -it \
  --runtime=nvidia \
  --privileged \
  --device /dev/ttyACM0 \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  -e CAMERA_FLIP_METHOD=0 \
  -e DETECTOR_BACKEND=opencv_dnn \
  -e OPENCV_DNN_TARGET=cuda \
  -p 8080:8080 \
  pantilt:latest
```

CPU baseline run (for comparison):

```bash
docker run --rm -it \
  --runtime=nvidia \
  --privileged \
  --device /dev/ttyACM0 \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  -e CAMERA_FLIP_METHOD=0 \
  -e DETECTOR_BACKEND=dlib \
  -p 8080:8080 \
  pantilt:latest
```

**Static photo check** (detector + landmarks, no camera or Arduino; needs NVIDIA runtime if OpenCV is CUDA-linked):

```bash
docker run --rm -it \
  --runtime=nvidia \
  --privileged \
  -v "$(pwd):/work" \
  -e DETECTOR_BACKEND=opencv_dnn \
  -e OPENCV_DNN_TARGET=cuda \
  pantilt:latest \
  python3 src/main.py --static-image /work/my_photo.jpg --static-save /work/annotated.jpg
```

Use `--static-show` only when a display is available (e.g. native `python3 src/main.py` on a desktop).

Preview stream:

- `http://<jetson-ip>:8080/`

**Processing FPS (live):** use `python3 src/main.py --print-fps` or set **`PRINT_FPS=1`** (e.g. Docker `-e PRINT_FPS=1`). Logs once per second; the MJPEG preview shows the same value in the **bottom-right** (large white text with black outline — wait ~1s for the first number; until then you’ll see `--- FPS`). This rate is the full loop, not the camera’s nominal FPS alone. Switch `DETECTOR_BACKEND` among `haar`, `dlib`, and `opencv_dnn` to compare.

**Processing FPS (offline, all three detectors):** `benchmark_detectors.py` writes `fps` and `avg_latency_ms` per detector to `detector_metrics.csv`.

## Camera Mirror/Orientation

Use `CAMERA_FLIP_METHOD`:

- `0`: no flip
- `2`: rotate 180
- `4`: horizontal mirror
- `6`: vertical mirror

## Runtime Model Files

On first run, the app auto-downloads:

- Haar cascade XML
- dlib `shape_predictor_68_face_landmarks.dat` (downloaded as `.bz2`, then decompressed)
- OpenCV DNN face model (`deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`)

## Face Tracking Tuning (in `src/config.py`)

- `PAN_KP`, `TILT_KP`: response speed
- `PAN_KD`, `TILT_KD`: damping for overshoot
- `DEAD_ZONE`: center tolerance to reduce jitter
- `SMOOTH_FRAMES`: face center smoothing window
- `MAX_SERVO_DELTA_PER_FRAME`: max servo change each frame
- `SERVO_MIN_COMMAND_DELTA`: suppress tiny servo commands
- `LOCK_HOLD_ZONE`, `LOCK_BREAK_ZONE`: lock hysteresis (hold steady at center, move only when real shift is detected)
- `ERROR_FILTER_ALPHA`: smoothing on center error (reduces servo chatter while locked)
- `PAN_*` and `TILT_*` lock/filter/delta settings: axis-specific anti-jitter tuning (tilt can be stricter)
- `SERVO_SEND_MIN_INTERVAL_SEC`, `PAN_SEND_MIN_STEP_DEG`, `TILT_SEND_MIN_STEP_DEG`: serial command debouncing to suppress micro-jitter
- `FACE_LOST_HOLD_SECONDS`: delay before scan starts after missed face
- `SEARCH_PAN_STEP`, `SEARCH_TILT_STEP`, `SEARCH_TILT_HOLD_STEPS`: scan pattern while face is lost
- `DETECTOR_BACKEND`: `opencv_dnn`, `dlib`, or `haar`
- `OPENCV_DNN_TARGET`: `cuda` or `cpu`
- `OPENCV_DNN_CONFIDENCE_THRESHOLD`: OpenCV DNN detection confidence threshold

## Report workflow (live tracking → metrics for Haar / dlib / OpenCV DNN)

**Design:** One camera process runs face tracking. While it runs, you can save **clean frames** (no overlays) plus servo/detector metadata. Later, **`benchmark_detectors.py`** runs **all three detectors offline on those same saved images** and compares them using **`labels.csv`** as ground truth.

You do **not** run two camera apps at once. Capture happens **inside** the tracker loop.

### 1) Run tracking and record an evaluation session

Example (native):

```bash
python3 src/main.py \
  --eval-session \
  --eval-dir eval_sessions \
  --eval-every-n 15 \
  --eval-max-frames 300
```

Example (Docker — mount host folder so images persist):

```bash
docker run --rm -it \
  --runtime=nvidia \
  --privileged \
  --device /dev/ttyACM0 \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  -v "$(pwd)/eval_sessions:/app/eval_sessions" \
  -e DETECTOR_BACKEND=opencv_dnn \
  -e OPENCV_DNN_TARGET=cuda \
  -p 8080:8080 \
  pantilt:latest \
  python3 src/main.py --eval-session --eval-max-frames 300
```

What gets written:

- `eval_sessions/<timestamp>/track_00000.jpg ...` — raw frames during tracking (motion, pan/tilt, lighting as in operation)
- `eval_sessions/<timestamp>/tracking_manifest.csv` — timestamp, pan/tilt, which detector was driving tracking, face locked flag, tracked bbox

Set **`DETECTOR_BACKEND`** to whichever model you want **live** (often `opencv_dnn`). Benchmark still evaluates **all three** detectors on the saved frames once labels exist.

### 2) Build ground-truth `labels.csv` for those frames

Seed boxes with your strongest detector (recommended **`opencv_dnn`**), then correct mistakes in the reviewer:

```bash
python3 src/prelabel_dataset.py \
  --dataset-dir eval_sessions/<timestamp> \
  --output-csv labels.csv \
  --detector opencv_dnn
```

```bash
python3 src/review_labels.py \
  --dataset-dir eval_sessions/<timestamp> \
  --labels-csv labels.csv
```

**Reporting note:** If labels come entirely from OpenCV DNN without human edits, that detector’s precision/recall vs itself will look artificially strong. For defensible results, **fix obvious boxes** in `review_labels.py`, or state clearly that labels are “reference-assisted” and interpret OpenCV DNN metrics cautiously.

### 3) Run detector benchmark (no camera — reads JPEGs from disk)

```bash
python3 src/benchmark_detectors.py \
  --dataset-dir eval_sessions/<timestamp> \
  --labels-csv eval_sessions/<timestamp>/labels.csv \
  --output-dir report_artifacts
```

### Docker: prelabel, review, and benchmark

The L4T ML image’s OpenCV is **CUDA-linked**. `import cv2` loads CUDA libraries (for example `libcublas.so.10`). If you run the container **without** the NVIDIA runtime, you get `ImportError: libcublas.so.10: cannot open shared object file`. Pass **`--runtime=nvidia`** (and **`--privileged`** if that matches how you run the tracker) even for these disk-only steps.

Use the **repository root** as the host working directory so the volume matches `--dataset-dir eval_sessions/<timestamp>`:

```bash
cd /path/to/pantilt-tracker-arduino
docker run --rm -it \
  --runtime=nvidia \
  --privileged \
  -v "$(pwd)/eval_sessions:/app/eval_sessions" \
  pantilt:latest \
  python3 src/prelabel_dataset.py \
    --dataset-dir eval_sessions/<timestamp> \
    --output-csv labels.csv \
    --detector opencv_dnn
```

If your shell is already **`.../eval_sessions`** (the folder that contains `20260502_092655`), do **not** add another `eval_sessions` in the mount: use `-v "$(pwd):/app/eval_sessions"` and keep **`--dataset-dir eval_sessions/<timestamp>`** (paths are still relative to **`/app`** in the container).

Reviewer shortcuts:

- `n` / right arrow: next image
- `p` / left arrow: previous image
- `a`: add face box (ROI selector)
- `x`: delete last box
- `c`: clear boxes for current image
- `s`: save now
- `q`: quit (auto-saves pending edits)

`labels.csv` format (`filename,x,y,w,h` — one row per face):

```csv
filename,x,y,w,h
track_00042.jpg,320,140,180,180
```

Optional: `--iou-threshold 0.5` on `benchmark_detectors.py`.


## Serial Protocol

Python sends:

- `PAN:<angle>`
- `TILT:<angle>`

Arduino constraints:

- Pan: `0..180`
- Tilt: `40..140`

## Troubleshooting

- `ModuleNotFoundError: No module named 'dlib'` when running `python3 src/main.py` **on the host**:
  - The Docker image installs `dlib`; a bare Jetson Python environment does not. Either run inside Docker, or use **system NumPy/Matplotlib** (fastest on Jetson) plus pip for the rest:

```bash
sudo apt-get update && sudo apt-get install -y \
  cmake build-essential libopenblas-base liblapack-dev \
  python3-numpy python3-matplotlib
pip3 install --user packaging setuptools wheel
pip3 install --user -r requirements.txt
```

  For **`benchmark_detectors.py`** plots only: if `python3-matplotlib` is not enough, try `pip3 install --user -r requirements-report.txt` (may compile — Docker avoids this issue).

- **`pip install matplotlib` / NumPy fails from source** (Python 3.6 / Jetson):
  - Do **not** pull NumPy/Matplotlib via pip on the host; use **`sudo apt-get install python3-numpy python3-matplotlib`** and keep them out of `requirements.txt`.

- `ModuleNotFoundError: No module named 'packaging'` while **`pip3 install dlib`**:
  - Install tooling first, then use the pinned version from this repo (Jetson + Python 3.6 is unreliable with `dlib` 20.x):

```bash
pip3 install --user packaging setuptools wheel
pip3 install --user -r requirements.txt
```

- `RuntimeError: Unable to open shape_predictor...`:
  - Ensure first run had internet access for model download
- Dlib install fails in container:
  - rebuild image after Dockerfile changes (`cmake`, `build-essential`)
- `ImportError: libcublas.so.10` (or similar) on **`import cv2`** inside Docker:
  - The image’s OpenCV needs Jetson CUDA user-space libraries. Run with **`--runtime=nvidia`** (see **Docker: prelabel, review, and benchmark**). Alternatively run `python3 src/prelabel_dataset.py` / `benchmark_detectors.py` **on the host** with APT OpenCV if you prefer not to start the GPU stack for offline jobs.
- OpenCV DNN backend falls back to CPU:
  - set `-e OPENCV_DNN_TARGET=cpu` explicitly if CUDA backend is unavailable in your OpenCV build
  - verify with `tegrastats` during runtime (GPU load should rise in CUDA mode)
- No serial port:
  - pass correct `--device` and/or set `ARDUINO_PORT` in `src/config.py`
- Tracking oscillation:
  - reduce `PAN_KP` / `TILT_KP`, increase `PAN_KD` / `TILT_KD`
