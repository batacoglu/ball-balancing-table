import argparse
import math
import time

import serial

from config import (
    L1_MM, DX_MM,
    SERVO_X_CENTER, SERVO_Y_CENTER,
    SERVO_X_MIN, SERVO_X_MAX,
    SERVO_Y_MIN, SERVO_Y_MAX,
    SIGN_X, SIGN_Y,
    TILT_MIN_DEG, TILT_MAX_DEG,
    KP, KD, KI,
    VEL_FILTER_ALPHA,
    INTEGRAL_CONTRIB_LIMIT_DEG,
    DEFAULT_PORT, DEFAULT_BAUD,
)
from vision import (
    build_camera, grab_gray, find_ball,
    load_homography, px_to_mm, HOMOGRAPHY_PATH,
)

_RATIO = DX_MM / L1_MM


def _tilt_to_phi_deg(theta_deg):

    theta_rad = math.radians(theta_deg)
    sin_phi = _RATIO * math.sin(theta_rad)
    sin_phi = max(-1.0, min(1.0, sin_phi))   # guard asin() domain
    return math.degrees(math.asin(sin_phi))


def tilt_to_servo_angles(theta_x_deg, theta_y_deg):

    theta_x_deg = max(TILT_MIN_DEG, min(TILT_MAX_DEG, theta_x_deg))
    theta_y_deg = max(TILT_MIN_DEG, min(TILT_MAX_DEG, theta_y_deg))

    phi_x = _tilt_to_phi_deg(theta_x_deg)
    phi_y = _tilt_to_phi_deg(theta_y_deg)

    servo_x = SERVO_X_CENTER + SIGN_X * phi_x
    servo_y = SERVO_Y_CENTER + SIGN_Y * phi_y

    servo_x = max(SERVO_X_MIN, min(SERVO_X_MAX, servo_x))
    servo_y = max(SERVO_Y_MIN, min(SERVO_Y_MAX, servo_y))

    return servo_x, servo_y

class NanoLink:

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, timeout=1.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)   # Nano resets on serial connect; let it boot

    def send_angles(self, servo_x_angle, servo_y_angle):
        line = f"{servo_x_angle:.2f} {servo_y_angle:.2f}\n"
        self.ser.write(line.encode("utf-8"))

    def read_response(self):
        if self.ser.in_waiting > 0:
            return self.ser.readline().decode("utf-8", errors="replace").strip()
        return None

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class NullLink:

    def send_angles(self, servo_x_angle, servo_y_angle):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class PIDController:

    def __init__(self, kp, kd, ki,
                 vel_filter_alpha=VEL_FILTER_ALPHA,
                 integral_limit_deg=INTEGRAL_CONTRIB_LIMIT_DEG,
                 tilt_min_deg=TILT_MIN_DEG, tilt_max_deg=TILT_MAX_DEG):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.vel_filter_alpha = vel_filter_alpha
        self.integral_limit_deg = integral_limit_deg
        self.tilt_min_deg = tilt_min_deg
        self.tilt_max_deg = tilt_max_deg
        self.reset()

    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self.prev_time = None
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0

    def _clamp_integral(self, integral_term_deg):
        return max(-self.integral_limit_deg,
                   min(self.integral_limit_deg, integral_term_deg))

    def update(self, x_mm, y_mm, timestamp):

        if self.prev_x is None:
            self.prev_x = x_mm
            self.prev_y = y_mm
            self.prev_time = timestamp
            return 0.0, 0.0

        dt = timestamp - self.prev_time
        if dt <= 0:
            dt = 0.01

        vx_raw = (x_mm - self.prev_x) / dt
        vy_raw = (y_mm - self.prev_y) / dt

        self.filtered_vx = (self.vel_filter_alpha * vx_raw
                            + (1 - self.vel_filter_alpha) * self.filtered_vx)
        self.filtered_vy = (self.vel_filter_alpha * vy_raw
                            + (1 - self.vel_filter_alpha) * self.filtered_vy)

        pd_x = self.kp * x_mm + self.kd * self.filtered_vx
        pd_y = self.kp * y_mm + self.kd * self.filtered_vy

        pd_x_saturated = pd_x <= self.tilt_min_deg or pd_x >= self.tilt_max_deg
        pd_y_saturated = pd_y <= self.tilt_min_deg or pd_y >= self.tilt_max_deg

        if not pd_x_saturated:
            self.integral_x += x_mm * dt
        if not pd_y_saturated:
            self.integral_y += y_mm * dt

        integral_term_x = self._clamp_integral(self.ki * self.integral_x)
        integral_term_y = self._clamp_integral(self.ki * self.integral_y)

        self.prev_x = x_mm
        self.prev_y = y_mm
        self.prev_time = timestamp

        return pd_x + integral_term_x, pd_y + integral_term_y

def run_loop(dry_run=False, use_pid=True):
    try:
        H = load_homography()
    except FileNotFoundError:
        print(f"ERROR: {HOMOGRAPHY_PATH} not found. "
              f"Run `python vision.py calibrate` first.")
        return

    picam2 = build_camera()
    link = NullLink() if dry_run else NanoLink()

    controller = PIDController(KP, KD, KI)
    if use_pid:
        print(f"PID gains: Kp={KP}, Kd={KD}, Ki={KI} "
              f"(Ki is a starting guess, not derived)")
    else:
        print("PID disabled (--no-pid): logging ball position only.")
    if dry_run:
        print("DRY RUN: no serial port opened; angles printed only.")

    print("Running control loop. Ctrl+C to stop.")

    try:
        while True:
            gray = grab_gray(picam2)
            result = find_ball(gray)
            timestamp = time.time()

            if result is not None:
                cx, cy, _ = result
                x_mm, y_mm = px_to_mm(cx, cy, H)
                if use_pid:
                    theta_x, theta_y = controller.update(x_mm, y_mm, timestamp)
                else:
                    theta_x, theta_y = 0.0, 0.0
            else:
                x_mm = y_mm = None
                theta_x, theta_y = 0.0, 0.0
                controller.reset()

            servo_x, servo_y = tilt_to_servo_angles(theta_x, theta_y)
            link.send_angles(servo_x, servo_y)

            if result is not None:
                print(f"ball=({x_mm:+6.1f},{y_mm:+6.1f})mm  "
                      f"tilt=({theta_x:+5.1f},{theta_y:+5.1f})deg  "
                      f"servo=({servo_x:6.2f},{servo_y:6.2f})")
            else:
                print("No ball detected")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        picam2.stop()
        link.close()

def main():
    parser = argparse.ArgumentParser(
        description="Ball-balancing control loop.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't open the serial port; print angles only. "
                             "Use this to test the pipeline without the Nano.")
    parser.add_argument("--no-pid", action="store_true",
                        help="Skip PID; hold tilt at zero and just log ball "
                             "position. Useful for a wiring sanity check.")
    args = parser.parse_args()

    run_loop(dry_run=args.dry_run, use_pid=not args.no_pid)

if __name__ == "__main__":
    main()
