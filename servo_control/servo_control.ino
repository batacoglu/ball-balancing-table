
#include <Servo.h>

Servo ServoX;
Servo ServoY;

const float SERVO_X_MIN = 45.0;
const float SERVO_X_MAX = 145.0;
const float SERVO_Y_MIN = 39.0;
const float SERVO_Y_MAX = 139.0;

String inputString = "";

void setup() {
  Serial.begin(115200);
  ServoX.attach(9);
  ServoY.attach(12);
  ServoX.write((SERVO_X_MIN + SERVO_X_MAX) / 2.0);
  ServoY.write((SERVO_Y_MIN + SERVO_Y_MAX) / 2.0);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processLine(inputString);
      inputString = "";
    } else {
      inputString += c;
    }
  }
}

void processLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  int spacePos = line.indexOf(' ');
  if (spacePos == -1) {
    Serial.println("ERROR: expected two values separated by a space");
    return;
  }

  String partX = line.substring(0, spacePos);
  String partY = line.substring(spacePos + 1);
  partX.trim();
  partY.trim();

  if (partX.length() == 0 || partY.length() == 0) {
    Serial.println("ERROR: missing value");
    return;
  }

  float angleX = partX.toFloat();
  float angleY = partY.toFloat();

  angleX = constrain(angleX, SERVO_X_MIN, SERVO_X_MAX);
  angleY = constrain(angleY, SERVO_Y_MIN, SERVO_Y_MAX);

  ServoX.write(angleX);
  ServoY.write(angleY);

  Serial.print("OK: X=");
  Serial.print(angleX);
  Serial.print(" Y=");
  Serial.println(angleY);
}
