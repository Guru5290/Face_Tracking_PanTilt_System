# Pan-Tilt Tracker (Arduino + Jetson)

Rigid and stable pan-tilt tracking system with 3 working modes:

1. Face lock with auto scan recovery (PID-controlled, jitter-reduced)
2. Patrol mode (scan pattern)
3. Motion tracking (frame-difference tracking + idle sweep)

This project uses:

- Jetson + PiCamera V2 + OpenCV for vision
- Arduino Mega for low-latency servo drive
- Serial control (`PAN:<angle>`, `TILT:<angle>`)

## Features

- Stable face lock with:
  - PID control (`KP/KI/KD`)
  - dead-zone to avoid micro-jitter
  - moving-average smoothing of detected face coordinates
  - servo step clamping per frame to reduce overshoot
  - hold-before-reset logic when face is briefly lost
  - automatic patrol scan while face is lost, with instant relock when found
- Patrol mode:
  - continuous left-right pan sweep
  - periodic tilt stepping to cover vertical field
- Motion tracking mode:
  - contour-based motion detection from frame differences
  - target center smoothing over recent detections
  - automatic idle sweep when no motion is found
- MJPEG preview stream available at `http://<jetson-ip>:8080/`

## Repository Layout

- `src/main.py`: entry point and mode selection
- `src/face_centering.py`: face lock mode
- `src/patrol_mode.py`: patrol scan mode
- `src/motion_tracking.py`: motion tracking mode
- `src/servo_controller.py`: serial servo commands to Arduino
- `src/stream_server.py`: shared MJPEG stream server
- `src/config.py`: tuning/configuration values
- `arduino/pan_tilt_servo.ino`: Arduino firmware

## Hardware Requirements

- NVIDIA Jetson board with CSI camera
- Arduino Mega (or compatible serial Arduino)
- 2x servos for pan + tilt
- Pan-tilt bracket
- Stable external 5V servo power supply (recommended)
- Common ground shared between Arduino and servo power

## Wiring

Arduino firmware expects:

- Pan servo signal: pin `9`
- Tilt servo signal: pin `10`

Important:

- Servo power should not come from USB alone under load.
- Tie grounds together (Jetson/Arduino/servo PSU) to avoid unstable behavior.

## Arduino Setup

1. Open `arduino/pan_tilt_servo.ino` in Arduino IDE.
2. Select board/port for Arduino Mega.
3. Upload firmware.
4. Keep baud at `115200` (must match `ARDUINO_BAUD` in `src/config.py`).

## Jetson / Python - Working in a container

### Docker 

```bash
docker build -t pantilt:latest .

docker run --rm -it   --runtime=nvidia   --privileged   --device /dev/ttyACM0   --volume /tmp/argus_socket:/tmp/argus_socket   -p 8080:8080   pantilt:latest

```

### Camera orientation / mirror control

You can control camera orientation from Docker using `CAMERA_FLIP_METHOD`:

- `0`: no flip (recommended if left/right is currently reversed)
- `4`: horizontal mirror (use only if you intentionally want mirrored view)
- `2`: rotate 180
- `6`: vertical mirror

Example (non-mirrored, normal left/right):

```bash
docker run --rm -it \
  --runtime=nvidia \
  --privileged \
  --device /dev/ttyACM0 \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  -e CAMERA_FLIP_METHOD=0 \
  -p 8080:8080 \
  pantilt:latest
```

Example (intentionally mirrored):

```bash
docker run --rm -it \
  --runtime=nvidia \
  --privileged \
  --device /dev/ttyACM0 \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  -e CAMERA_FLIP_METHOD=4 \
  -p 8080:8080 \
  pantilt:latest
```

Tip: the app prints `Camera flip-method: X` at startup so you can confirm active mode.

## Running the System

Start:

```bash
python3 src/main.py
```

Mode options:

- `1`: Face lock + auto scan recovery
- `2`: Patrol mode
- `3`: Motion tracking

To disable the stream preview :

```bash
python3 src/main.py --no-preview
```

## Stability and Rigidity Tuning

All tuning values are in `src/config.py`.

### Face lock responsiveness

- `PAN_KP`, `TILT_KP`: increase for faster reaction, decrease if oscillation appears
- `PAN_KD`, `TILT_KD`: increase to damp overshoot and hunting
- `PAN_KI`, `TILT_KI`: keep low unless persistent offset exists

### Jitter reduction

- `DEAD_ZONE`: bigger value = less micro movement near center
- `SMOOTH_FRAMES`: bigger value = smoother but slower target response
- `MAX_SERVO_DELTA_PER_FRAME`: hard cap on per-frame servo movement
- `SERVO_MIN_COMMAND_DELTA`: ignore tiny command changes

### Lost target behavior

- `FACE_LOST_HOLD_SECONDS`: how long to hold state before controller reset
- After this hold period, mode `1` automatically starts scan patrol and keeps scanning until a face is detected again.

### Motion mode sensitivity

- `MIN_MOTION_AREA`: increase to ignore small noise
- `MOTION_THRESHOLD`: increase to reduce false positives
- `MOTION_BLUR_SIZE`: larger blur reduces sensor noise sensitivity
- `MOTION_IDLE_SWEEP_SECONDS`: delay before idle sweep starts

### Patrol pattern coverage

- `PATROL_PAN_STEP`: horizontal scan speed
- `PATROL_TILT_STEP`: vertical step size
- `PATROL_TILT_HOLD_STEPS`: delay between vertical changes
- `PATROL_STEP_DELAY`: loop delay for smoothness/CPU balance

## Serial Protocol

Python sends:

- `PAN:<angle>`
- `TILT:<angle>`

Arduino constrains angles to:

- Pan: `0..180`
- Tilt: `40..140`

## Troubleshooting

- Arduino not found:
  - Set `ARDUINO_PORT` in `src/config.py` (example: `/dev/ttyACM0`)
  - Check cable and permissions
- Camera cannot open:
  - Confirm CSI camera and Jetson camera stack are available
  - Verify Docker has required device/runtime access
- Tracking jitters:
  - Increase `DEAD_ZONE`
  - Increase `SMOOTH_FRAMES`
  - Reduce `PAN_KP`/`TILT_KP`
  - Increase `PAN_KD`/`TILT_KD`
- Slow response:
  - Increase `KP` slightly
  - Reduce `SMOOTH_FRAMES`
  - Increase `MAX_SERVO_DELTA_PER_FRAME`

## Safety Notes

- Keep clear of moving pan-tilt mechanism while powered.
- Use adequate servo power supply.
- Start with conservative gains before aggressive tuning.

## Features Implemented for this small project - courtesy of Computer Controlled Manufacturing school project

- Face lock mode: implemented and stabilized
- Patrol mode: implemented
- Motion tracking mode: implemented
- Shared MJPEG preview stream: implemented

