import argparse
import csv
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from config import (
    FULL_W, FULL_H,
    CROP_X_PX, CROP_W, CROP_H,
    TABLE_SIZE_MM,
    HEIGHT_CORRECTION_K,
    BALL_THRESHOLD, BALL_MIN_AREA, BALL_CIRCULARITY, BALL_INVERT,
    HOMOGRAPHY_PATH,
)

REAL_CORNERS = np.array([
    [0.0,           TABLE_SIZE_MM],   # top-left
    [TABLE_SIZE_MM, TABLE_SIZE_MM],   # top-right
    [TABLE_SIZE_MM, 0.0],             # bottom-right
    [0.0,           0.0],             # bottom-left
], dtype=np.float32)

CORNER_LABELS = ["TL", "TR", "BR", "BL"]

def build_camera():
    """Start a Picamera2 stream at the full sensor resolution."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FULL_W, FULL_H), "format": "RGB888"},
    )
    picam2.configure(config)
    picam2.start()
    return picam2


def grab_gray(picam2):

    frame = picam2.capture_array()
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    if CROP_X_PX > 0:
        gray = gray[:, CROP_X_PX:FULL_W - CROP_X_PX]
    return gray

def load_homography(path=HOMOGRAPHY_PATH):

    H = np.load(path).astype(np.float64)
    if CROP_X_PX > 0:
        T = np.array([[1.0, 0.0, float(CROP_X_PX)],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        H = H @ T
    return H

def _threshold_mask(gray, threshold, invert):
    """Blur + threshold. Returns a binary mask (uint8, 0/255)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, threshold, 255, mode)
    return mask


def find_ball(gray,
              threshold=BALL_THRESHOLD,
              min_area=BALL_MIN_AREA,
              circularity=BALL_CIRCULARITY,
              invert=BALL_INVERT):

    mask = _threshold_mask(gray, threshold, invert)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area <= best_area:
            continue
        if circularity > 0.0:
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * np.pi * area / (perim * perim)
            if circ < circularity:
                continue
        best = c
        best_area = area

    if best is None:
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"], best_area

def px_to_mm(px, py, homography):

    pt = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, homography)
    raw_x = float(mapped[0][0][0])
    raw_y = float(mapped[0][0][1])
    cx = raw_x - TABLE_SIZE_MM / 2.0
    cy = raw_y - TABLE_SIZE_MM / 2.0
    return cx * HEIGHT_CORRECTION_K, cy * HEIGHT_CORRECTION_K

def run_center():
    picam2 = build_camera()
    print("Adjust the camera mount until the table's geometric center")
    print("sits on the green crosshair. The orange box is a framing")
    print("reference — the table edges should sit just inside it.")
    print("Press 'q' when satisfied.")

    cx, cy = CROP_W // 2, CROP_H // 2
    size = int(min(CROP_W, CROP_H) * 0.75)
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half

    try:
        while True:
            gray = grab_gray(picam2)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 200, 255), 1)
            cv2.line(display, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
            cv2.line(display, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)
            cv2.putText(display, "center", (cx + 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Camera Centering", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

def run_calibrate():

    picam2 = build_camera()
    print("Capturing a frame for corner selection...")

    # Full-frame grab (no crop) — we need the whole image to click corners.
    frame = picam2.capture_array()
    picam2.stop()
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    clicked = []

    win = "Click calibration square corners: TL, TR, BR, BL"
    cv2.namedWindow(win)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicked) >= 4:
            return
        clicked.append((x, y))
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display, CORNER_LABELS[len(clicked) - 1],
                    (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        if len(clicked) == 4:
            pts = np.array(clicked, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
        cv2.imshow(win, display)

    cv2.setMouseCallback(win, on_click)

    print(f"Click the four corners of the {TABLE_SIZE_MM:.0f}x{TABLE_SIZE_MM:.0f}mm")
    print("calibration square (NOT the physical table edges — the full")
    print("table doesn't fit in frame) IN ORDER: top-left, top-right,")
    print("bottom-right, bottom-left. 'top' = the far edge in the image.")
    print("When all four are placed, press ENTER to confirm, 'r' to")
    print("restart, or 'q' to cancel.")

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
        if key == ord('r'):
            clicked.clear()
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            print("Restarted — click 4 corners again.")
            continue
        if len(clicked) == 4 and key in (13, 10, ord(' ')):
            break

    cv2.destroyAllWindows()

    pixel_corners = np.array(clicked, dtype=np.float32)
    H, _ = cv2.findHomography(pixel_corners, REAL_CORNERS)

    if H is None:
        print("ERROR: homography computation failed. Re-run and click again.")
        return

    np.save(HOMOGRAPHY_PATH, H)
    print(f"\nSaved homography to {HOMOGRAPHY_PATH}")
    print("Pixel corners (full frame):", pixel_corners.tolist())
    print("World corners (mm):", REAL_CORNERS.tolist())
    print(f"Output is center-relative: (0, 0) = center of the "
          f"{TABLE_SIZE_MM:.0f}mm calibration square.")
    print("Note: coordinates outside the calibrated square are")
    print("extrapolated and slightly less accurate near frame edges.")

def run_validate():
    try:
        H = load_homography()
    except FileNotFoundError:
        print(f"ERROR: {HOMOGRAPHY_PATH} not found. "
              f"Run `python vision.py calibrate` first.")
        return

    inv_H = np.linalg.inv(H)
    picam2 = build_camera()
    print("Live ball position in mm (center-relative to the calibration")
    print("square). Blue crosses = where the calibration square corners")
    print("land in the image — they should sit on the physical marks.")
    print("Press 'q' to quit.")

    try:
        while True:
            gray = grab_gray(picam2)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            for wx, wy in REAL_CORNERS:
                pt = np.array([[[wx, wy]]], dtype=np.float32)
                m = cv2.perspectiveTransform(pt, inv_H)
                px, py = int(m[0][0][0]), int(m[0][0][1])
                cv2.drawMarker(display, (px, py), (255, 200, 0),
                               cv2.MARKER_CROSS, 20, 2)

            result = find_ball(gray)
            if result is not None:
                bx, by, _ = result
                x_mm, y_mm = px_to_mm(bx, by, H)
                cv2.circle(display, (int(bx), int(by)), 8, (0, 0, 255), 2)
                cv2.putText(display, f"({x_mm:+.1f}, {y_mm:+.1f}) mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "no ball", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            cv2.imshow("Validate", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

def run_tune():
    picam2 = build_camera()

    win = "tune"
    mask_win = "mask"
    cv2.namedWindow(win)
    cv2.createTrackbar("threshold", win, BALL_THRESHOLD, 255, lambda x: None)
    cv2.createTrackbar("min_area", win, int(BALL_MIN_AREA), 2000, lambda x: None)
    cv2.createTrackbar("circ_x100", win, int(BALL_CIRCULARITY * 100), 100, lambda x: None)
    cv2.createTrackbar("invert", win, int(BALL_INVERT), 1, lambda x: None)

    print(f"Crop active: {CROP_X_PX} px per side "
          f"(cropped frame is {CROP_W}x{CROP_H}).")
    print("Adjust sliders until only the ball is outlined.")
    print("Press 'p' to print tuned values (paste into config.py), "
          "'q' to quit.")

    last = dict(threshold=BALL_THRESHOLD,
                min_area=BALL_MIN_AREA,
                circularity=BALL_CIRCULARITY,
                invert=BALL_INVERT)

    try:
        while True:
            gray = grab_gray(picam2)

            threshold = cv2.getTrackbarPos("threshold", win)
            min_area = cv2.getTrackbarPos("min_area", win)
            circ_x100 = cv2.getTrackbarPos("circ_x100", win)
            invert = bool(cv2.getTrackbarPos("invert", win))
            circularity = circ_x100 / 100.0

            last.update(threshold=threshold, min_area=min_area,
                        circularity=circularity, invert=invert)

            mask = _threshold_mask(gray, threshold, invert)
            result = find_ball(gray, threshold, min_area, circularity, invert)

            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if result is not None:
                cx, cy, area = result
                cv2.circle(display, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.putText(display, f"area={area:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "no detection", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            info = (f"thr={threshold}  area>={min_area}  "
                    f"circ>={circularity:.2f}  invert={int(invert)}")
            cv2.putText(display, info, (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(win, display)
            cv2.imshow(mask_win, mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                print("\n--- paste into config.py ---")
                print(f"BALL_THRESHOLD = {last['threshold']}")
                print(f"BALL_MIN_AREA = {last['min_area']}")
                print(f"BALL_CIRCULARITY = {last['circularity']:.2f}")
                print(f"BALL_INVERT = {last['invert']}")
                print("----------------------------\n")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

def run_record(args):
    try:
        H = load_homography()
    except FileNotFoundError:
        print(f"ERROR: {HOMOGRAPHY_PATH} not found. "
              f"Run `python vision.py calibrate` first.")
        return

    picam2 = build_camera()

    smooth_x, smooth_y = CROP_W / 2.0, CROP_H / 2.0
    lost_frames = 0
    RESET_AFTER_LOST = 10

    csv_file = open(args.log, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "x_px", "y_px", "x_mm", "y_mm", "area_px"])

    print(f"Recording to {args.log}. Press Ctrl+C to stop.")
    if args.smooth:
        print(f"Smoothing ON (alpha={args.smooth_alpha}) — do NOT use this "
              f"for damping-curve fitting.")
    start_time = time.time()

    try:
        while True:
            gray = grab_gray(picam2)
            result = find_ball(gray, args.threshold, args.min_area,
                               args.circularity, args.invert)
            timestamp = time.time() - start_time

            if result is not None:
                cx, cy, area = result
                lost_frames = 0

                if args.smooth:
                    smooth_x = args.smooth_alpha * cx + (1 - args.smooth_alpha) * smooth_x
                    smooth_y = args.smooth_alpha * cy + (1 - args.smooth_alpha) * smooth_y
                    out_x, out_y = smooth_x, smooth_y
                else:
                    out_x, out_y = cx, cy

                x_mm, y_mm = px_to_mm(out_x, out_y, H)
                writer.writerow([f"{timestamp:.5f}",
                                 f"{out_x:.2f}", f"{out_y:.2f}",
                                 f"{x_mm:.2f}", f"{y_mm:.2f}",
                                 f"{area:.1f}"])
            else:
                lost_frames += 1
                if lost_frames >= RESET_AFTER_LOST:
                    smooth_x, smooth_y = CROP_W / 2.0, CROP_H / 2.0
                    lost_frames = 0

            if args.show:
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if result is not None:
                    cv2.circle(display, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.imshow("record", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        picam2.stop()
        csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Vision tools for the ball-balancing table.")
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("center", help="Aim the camera")
    sub.add_parser("calibrate", help="Click corners -> homography.npy")
    sub.add_parser("validate", help="Overlay corners + live mm readout")
    sub.add_parser("tune", help="Trackbar tuner")

    rec = sub.add_parser("record", help="Headless CSV logger")
    rec.add_argument("--log", type=str, default="trajectory.csv",
                     help="CSV output path")
    rec.add_argument("--threshold", type=int, default=BALL_THRESHOLD)
    rec.add_argument("--min-area", type=float, default=BALL_MIN_AREA)
    rec.add_argument("--circularity", type=float, default=BALL_CIRCULARITY)
    rec.add_argument("--invert", action="store_true", default=BALL_INVERT)
    rec.add_argument("--smooth", action="store_true",
                     help="EMA smoothing. OFF by default — leave off for "
                          "damping/trajectory runs, since smoothing distorts "
                          "the decay shape.")
    rec.add_argument("--smooth-alpha", type=float, default=0.3)
    rec.add_argument("--show", action="store_true",
                     help="Show a live preview during recording (costs fps)")

    args = parser.parse_args()

    if args.mode == "center":
        run_center()
    elif args.mode == "calibrate":
        run_calibrate()
    elif args.mode == "validate":
        run_validate()
    elif args.mode == "tune":
        run_tune()
    elif args.mode == "record":
        run_record(args)


if __name__ == "__main__":
    main()
