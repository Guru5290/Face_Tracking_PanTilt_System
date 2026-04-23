import serial
import serial.tools.list_ports
import time
from config import *


def find_arduino_port():
    """Auto-detect Arduino Mega USB serial port."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Arduino Mega shows up as ttyACM or ttyUSB
        if 'ttyACM' in port.device or 'ttyUSB' in port.device:
            return port.device
    return None


class ServoController:

    def __init__(self):
        port = ARDUINO_PORT or find_arduino_port()
        if port is None:
            raise RuntimeError(
                'Arduino not found. Check USB cable and ARDUINO_PORT in config.py'
            )
        print(f'Connecting to Arduino on {port}...')
        self.ser = serial.Serial(port, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset after serial connection
        print('Arduino connected.')

        self.pan_angle  = PAN_CENTER
        self.tilt_angle = TILT_CENTER

        self.home()

    def _send(self, cmd):
        """Send a command string to Arduino over serial."""
        self.ser.write((cmd + '\n').encode())
        # No sleep here — let the main loop control timing, not the serial send

    def set_pan(self, angle):
        angle = max(PAN_MIN, min(PAN_MAX, float(angle)))
        self.pan_angle = angle
        self._send(f'PAN:{int(angle)}')

    def set_tilt(self, angle):
        angle = max(TILT_MIN, min(TILT_MAX, float(angle)))
        self.tilt_angle = angle
        self._send(f'TILT:{int(angle)}')

    def home(self):
        # Return both servos to center position
        self.set_pan(PAN_CENTER)
        self.set_tilt(TILT_CENTER)

    def cleanup(self):
        # Return to home and release serial port cleanly
        try:
            self.home()
            time.sleep(0.3)
        except Exception:
            pass
        finally:
            if self.ser.is_open:
                self.ser.close()
            print('Serial port closed.')
