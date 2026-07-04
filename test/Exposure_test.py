import os
import time
from datetime import datetime

import cv2
import numpy as np

from motion_control import init_motion, shutdown_motion, rotate_camera, set_filter


# ==========================
# USER CONFIG
# ==========================
open_filterwheel_pos = 3
opaque_filterwheel_pos = 7
camera_motor_pos = 1000

exposure_test = [500, 400, 300, 200, 100, 50, 20, 10]

ROOT_DIR = "/home/sciglob/HSI_code/hsi/hsi_captures/live"
MOTOR_PORT = "/dev/ttyUSB0"
FW_PORT = "/dev/ttyACM0"

# Capture / SNR
N_DARK = 10   # opaque_filterwheel_pos
N_AVG  = 10   # open_filterwheel_pos

# Camera
CAM_INDEX = 0
WIDTH = 1280
HEIGHT = 800

# Delay/timing (tune if needed)
SETTLE_AFTER_FILTER_S = 3 #0.8
SETTLE_AFTER_EXPOSURE_S = 3 #0.3
SETTLE_AFTER_MOTOR_S = 3 #0.8


# ==========================
# Helpers
# ==========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_imwrite(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")

def capture_gray(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera.")
    # If camera already outputs grayscale, this still works (3ch -> gray)
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return gray

def average_frames(cap: cv2.VideoCapture, n: int) -> np.ndarray:
    """
    Capture n frames and return uint8 averaged image.
    Uses float32 accumulation to reduce rounding error.
    """
    acc = None
    for i in range(n):
        img = capture_gray(cap).astype(np.float32)
        if acc is None:
            acc = img
        else:
            acc += img
        # tiny pause helps some V4L2 drivers deliver fresh frames
        time.sleep(0.01)
    avg = (acc / float(n))
    # Clip to [0,255] and convert back
    return np.clip(avg, 0, 255).astype(np.uint8)

def set_manual_exposure(cap: cv2.VideoCapture, exposure_value: float) -> None:
    """
    Attempt to force manual exposure then set exposure.
    Note: exact exposure units vary by camera/driver.
    """
    # V4L2 style used on Jetson for many UVC cameras:
    # 0.75 manual, 0.25 auto (common convention)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value))

def flush_frames(cap: cv2.VideoCapture, n: int = 3) -> None:
    for _ in range(n):
        cap.read()
        time.sleep(0.01)


# ==========================
# Main
# ==========================
def main():
    # 1) Create output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT_DIR, f"exposure_test_{ts}")
    ensure_dir(out_dir)
    print(f"[INFO] Saving results to: {out_dir}")

    # 2) Init motion + move to initial positions
    init_motion(MOTOR_PORT, FW_PORT)

    try:
        print(f"[INFO] Moving motor to {camera_motor_pos} ...")
        rotate_camera(camera_motor_pos)
        time.sleep(SETTLE_AFTER_MOTOR_S)

        # 3) Open camera
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera. Check CAM_INDEX or connection.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Make sure we're producing frames
        flush_frames(cap, n=5)

        for exp in exposure_test:
            print(f"\n[TEST] Exposure = {exp}")

            # ---- Bright average ----
            print(f"[INFO] Set filter wheel -> OPEN (pos {open_filterwheel_pos})")
            set_filter(open_filterwheel_pos)
            time.sleep(SETTLE_AFTER_FILTER_S)

            print(f"[INFO] Setting exposure -> {exp}")
            set_manual_exposure(cap, exp)
            time.sleep(SETTLE_AFTER_EXPOSURE_S)
            flush_frames(cap, n=3)

            bright_avg = average_frames(cap, N_AVG)

            bright_name = (
                f"bright_exp{exp}_motor{camera_motor_pos}_fw{open_filterwheel_pos}.png"
            )
            bright_path = os.path.join(out_dir, bright_name)
            safe_imwrite(bright_path, bright_avg)
            print(f"[SAVE] {bright_path}  (avg of {N_AVG})")

            # ---- Dark average ----
            print(f"[INFO] Set filter wheel -> OPAQUE (pos {opaque_filterwheel_pos})")
            set_filter(opaque_filterwheel_pos)
            time.sleep(SETTLE_AFTER_FILTER_S)

            # Keep same exposure for dark frames (important!)
            print(f"[INFO] Dark capture at exposure -> {exp}")
            time.sleep(SETTLE_AFTER_EXPOSURE_S)
            flush_frames(cap, n=3)

            dark_avg = average_frames(cap, N_DARK)

            dark_name = (
                f"dark_exp{exp}_motor{camera_motor_pos}_fw{opaque_filterwheel_pos}.png"
            )
            dark_path = os.path.join(out_dir, dark_name)
            safe_imwrite(dark_path, dark_avg)
            print(f"[SAVE] {dark_path}  (avg of {N_DARK})")

        cap.release()
        print("\n[DONE] Exposure test complete.")

    finally:
        # Always close motion ports
        shutdown_motion()


if __name__ == "__main__":
    main()
