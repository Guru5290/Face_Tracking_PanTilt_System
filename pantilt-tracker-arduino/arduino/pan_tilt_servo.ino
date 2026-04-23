#include <Servo.h>

Servo pan_servo;
Servo tilt_servo;

// Servo signal pins on Arduino Mega
const int PAN_PIN  = 9;
const int TILT_PIN = 10;

// Angle limits — must match config.py
const int PAN_MIN  = 0,   PAN_MAX  = 180, PAN_CENTER  = 90;
const int TILT_MIN = 40,  TILT_MAX = 140, TILT_CENTER = 90;

String inputBuffer = "";

void setup() {
  Serial.begin(115200);   // Must match ARDUINO_BAUD in config.py
  inputBuffer.reserve(32);

  pan_servo.attach(PAN_PIN);
  tilt_servo.attach(TILT_PIN);

  // Move to center on startup
  pan_servo.write(PAN_CENTER);
  tilt_servo.write(TILT_CENTER);

  Serial.println("Arduino ready.");
}

void loop() {
  // Read serial byte by byte to avoid blocking
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();

  if (cmd.startsWith("PAN:")) {
    int angle = cmd.substring(4).toInt();
    angle = constrain(angle, PAN_MIN, PAN_MAX);
    pan_servo.write(angle);
  }
  else if (cmd.startsWith("TILT:")) {
    int angle = cmd.substring(5).toInt();
    angle = constrain(angle, TILT_MIN, TILT_MAX);
    tilt_servo.write(angle);
  }
  else if (cmd == "HOME") {
    pan_servo.write(PAN_CENTER);
    tilt_servo.write(TILT_CENTER);
  }
}
