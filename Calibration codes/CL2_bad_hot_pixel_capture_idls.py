import os
import csv
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from ids_peak_common import PixelFormat

from ids_camera import IDSPeakCamera
from motor_control_v2 import MotionSystem
from calibration_common import (
    ensure_dir,
    now_ts,
    clamp,
    r4,
    y_crop,
    save_u16_png,
    average_frames,
    compute_metrics,
    LiveUI,
)

# =========================================================
# USER CONFIG
# =========================================================

BASE_SAVE_DIR = r"HSI_ids\2_hsi_jetson\data\session_4"

# Windows
MOTOR_PORT = "COM23"
FILTERWHEEL_PORT = "COM19"

# Jetson
# MOTOR_PORT = "/dev/ttyUSB0"
# FILTERWHEEL_PORT = "/dev/ttyACM0"

CAMERA_INDEX = 0
IDS_W = 1440
IDS_H = 1080

CROP_Y0 = 150
CROP_Y1 = 800

IDS_CAMERA_PIXEL_FORMAT_ENTRY = "Mono12"
IDS_PIPELINE_OUTPUT_FORMAT = PixelFormat.MONO_12

BIT_DEPTH = 12
MAX_DN = (1 << BIT_DEPTH) - 1

GAIN = 1.0
USE_SOFTWARE_TRIGGER = False

EXPOSURE_START_MS = 1000
EXPOSURE_MIN_MS = 0.02
EXPOSURE_MAX_MS = 2000.0
EXPOSURE_STEP_MS = 5.0

GOOD_MIN_FRAC = 0.60
GOOD_MAX_FRAC = 0.95
SAT_THRESH = int(0.99 * MAX_DN)

GOOD_MIN_DN = int(GOOD_MIN_FRAC * MAX_DN)
GOOD_MAX_DN = int(GOOD_MAX_FRAC * MAX_DN)

USE_PERCENTILE_FOR_EXPOSURE = True
EXPOSURE_CONTROL_PERCENTILE = 99.9
SAT_PIXEL_FRACTION_LIMIT = 1e-5

FILTER_OPEN_POS = 3
FILTER_DARK_POS = 7

# Exposure plan relative to the found good exposure
EXPOSURE_MULTIPLIERS = [0.5, 0.6, 0.8, 0.9, 1.0, 1.1]

# Number of frames to save at each exposure
N_BRIGHT_FRAMES_PER_EXPOSURE = 1
N_DARK_FRAMES_PER_EXPOSURE = 50

SETTLE_TIME_SEC = 0.5
FRAME_SETTLE_DELAY = 0.5
INITIAL_FILTER_SETTLE_SEC = 1

PAUSE_S = 0.01
GRAB_TIMEOUT_MS = 5000

# Optimization behavior
MAX_OPT_ITERS = 20

# =========================================================
# MAIN APP
# =========================================================

class LDLSDarkCaptureApp:
    def __init__(self):
        self.session_dir = os.path.join(
            BASE_SAVE_DIR,
            f"bad_pixel_capture_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.bright_dir = os.path.join(self.session_dir, "bright")
        self.dark_dir = os.path.join(self.session_dir, "dark")
        ensure_dir(self.bright_dir)
        ensure_dir(self.dark_dir)

        self.meta_rows = []

        self.ids = IDSPeakCamera(pipeline_output=IDS_PIPELINE_OUTPUT_FORMAT)
        self.ids.open(index=CAMERA_INDEX)

        self.ids.node_map.FindNode("Width").SetValue(int(IDS_W))
        self.ids.node_map.FindNode("Height").SetValue(int(IDS_H))
        try:
            self.ids.node_map.FindNode("OffsetX").SetValue(0)
            self.ids.node_map.FindNode("OffsetY").SetValue(0)
        except Exception:
            pass

        self.ids.set_pixel_format_entry(IDS_CAMERA_PIXEL_FORMAT_ENTRY)
        self.ids.set_pipeline_output_format(IDS_PIPELINE_OUTPUT_FORMAT)
        self.ids.set_exposure_us(float(EXPOSURE_START_MS * 1000.0))
        self.gain = self.ids.set_gain(float(GAIN), selector="AnalogAll", clamp=True)

        if USE_SOFTWARE_TRIGGER:
            try:
                self.ids.set_trigger_software()
            except Exception:
                self.ids.set_trigger_off()
        else:
            self.ids.set_trigger_off()

        self.ids.start()
        self.exposure_ms = float(EXPOSURE_START_MS)

        self.motion = MotionSystem(
            motor_port=MOTOR_PORT,
            fw_port=FILTERWHEEL_PORT,
            motor_baud=9600,
            fw_baud=4800,
        )
        self.motion.connect()
        self.current_filter_pos = None

        self.ui = LiveUI(h=CROP_Y1 - CROP_Y0, w=IDS_W, max_dn=MAX_DN, pause_s=PAUSE_S)

        print("\n=== Reset filter wheel ===")
        self.motion.reset_fw()

    def get_camera_temperature_c(self) -> float:
        try:
            return float(self.ids.get_temperature_c())
        except Exception:
            return float("nan")
    
    def grab_hsi_frame(self) -> np.ndarray:
        frame = self.ids.grab(timeout_ms=GRAB_TIMEOUT_MS, do_software_trigger=USE_SOFTWARE_TRIGGER)
        return y_crop(frame, CROP_Y0, CROP_Y1)

    def discard_frames(self, n: int = 3):
        for _ in range(n):
            try:
                gray = self.grab_hsi_frame()
                m = compute_metrics(
                    gray,
                    sat_thresh=SAT_THRESH,
                    sat_pixel_fraction_limit=SAT_PIXEL_FRACTION_LIMIT,
                    use_percentile_for_exposure=USE_PERCENTILE_FOR_EXPOSURE,
                    exposure_control_percentile=EXPOSURE_CONTROL_PERCENTILE,
                    good_min_dn=GOOD_MIN_DN,
                    good_max_dn=GOOD_MAX_DN,
                    bad_pixel_mask=None,
                )
                temp_c = self.get_camera_temperature_c()
                m["camera_temp_c"] = temp_c
                self.ui.update(gray, "Settling / discard", self.current_filter_pos, self.exposure_ms, m)
            except Exception:
                pass
            time.sleep(FRAME_SETTLE_DELAY)

    def move_filter(self, pos: int):
        if self.current_filter_pos == pos:
            return
        self.motion.set_filter(int(pos))
        self.current_filter_pos = int(pos)
        time.sleep(SETTLE_TIME_SEC)
        self.discard_frames(2)

    def set_exposure(self, exposure_ms: float):
        exposure_ms = r4(clamp(exposure_ms, EXPOSURE_MIN_MS, EXPOSURE_MAX_MS))
        self.ids.set_exposure_us(exposure_ms * 1000.0)
        self.exposure_ms = exposure_ms
        time.sleep(FRAME_SETTLE_DELAY)
        self.discard_frames(2)

    def capture_average(self, n: int, mode_text: str) -> np.ndarray:
        frames = []
        for i in range(n):
            gray = self.grab_hsi_frame()
            frames.append(gray)

            m = compute_metrics(
                gray,
                sat_thresh=SAT_THRESH,
                sat_pixel_fraction_limit=SAT_PIXEL_FRACTION_LIMIT,
                use_percentile_for_exposure=USE_PERCENTILE_FOR_EXPOSURE,
                exposure_control_percentile=EXPOSURE_CONTROL_PERCENTILE,
                good_min_dn=GOOD_MIN_DN,
                good_max_dn=GOOD_MAX_DN,
                bad_pixel_mask=None,
            )
            self.ui.update(gray, f"{mode_text} ({i+1}/{n})", self.current_filter_pos, self.exposure_ms, m)
            time.sleep(FRAME_SETTLE_DELAY)

        avg = average_frames(frames)

        m_avg = compute_metrics(
            avg,
            sat_thresh=SAT_THRESH,
            sat_pixel_fraction_limit=SAT_PIXEL_FRACTION_LIMIT,
            use_percentile_for_exposure=USE_PERCENTILE_FOR_EXPOSURE,
            exposure_control_percentile=EXPOSURE_CONTROL_PERCENTILE,
            good_min_dn=GOOD_MIN_DN,
            good_max_dn=GOOD_MAX_DN,
            bad_pixel_mask=None,
        )
        self.ui.update(avg, f"{mode_text} AVG", self.current_filter_pos, self.exposure_ms, m_avg)

        return avg

    def optimize_open_filter_exposure(self) -> float:
        """
        Find a good exposure if possible.
        If lamp is dim and even EXPOSURE_MAX_MS is still below GOOD_MIN_DN,
        accept the best unsaturated exposure instead of failing.
        """
        self.move_filter(FILTER_OPEN_POS)
        self.set_exposure(EXPOSURE_START_MS)

        best_unsat_exp = self.exposure_ms
        best_unsat_ctrl = -1.0

        for _ in range(MAX_OPT_ITERS):
            gray = self.capture_average(2, "Optimize")
            m = compute_metrics(
                gray,
                sat_thresh=SAT_THRESH,
                sat_pixel_fraction_limit=SAT_PIXEL_FRACTION_LIMIT,
                use_percentile_for_exposure=USE_PERCENTILE_FOR_EXPOSURE,
                exposure_control_percentile=EXPOSURE_CONTROL_PERCENTILE,
                good_min_dn=GOOD_MIN_DN,
                good_max_dn=GOOD_MAX_DN,
                bad_pixel_mask=None,
            )

            print(
                f"[OPT] exp={self.exposure_ms:.2f} ms | ctrl={m['control_dn']:.1f} | "
                f"raw_max={m['raw_max_dn']} | sat={m['saturated']} | good={m['good']}"
            )

            if not m["saturated"] and m["control_dn"] > best_unsat_ctrl:
                best_unsat_ctrl = m["control_dn"]
                best_unsat_exp = self.exposure_ms

            if m["good"]:
                return self.exposure_ms

            # Too bright or saturated -> step down
            if m["saturated"] or (m["control_dn"] > GOOD_MAX_DN):
                new_exp = max(EXPOSURE_MIN_MS, self.exposure_ms * 0.70)

                # If already near minimum, accept current best unsaturated exposure
                if abs(new_exp - self.exposure_ms) < 1e-9:
                    print(f"[INFO] Using best unsaturated exposure: {best_unsat_exp:.2f} ms")
                    return best_unsat_exp

                self.set_exposure(new_exp)
                continue

            # Too dim -> step up
            if m["too_dim"]:
                if self.exposure_ms >= EXPOSURE_MAX_MS - 1e-9:
                    print(
                        f"[INFO] Could not reach GOOD_MIN_DN even at max exposure {self.exposure_ms:.2f} ms. "
                        f"Using max exposure as fallback."
                    )
                    return self.exposure_ms

                new_exp = min(EXPOSURE_MAX_MS, max(self.exposure_ms + EXPOSURE_STEP_MS, self.exposure_ms * 1.5))
                self.set_exposure(new_exp)
                continue

        print(
            f"[INFO] Optimization ended without hitting 'good' range. "
            f"Using best unsaturated exposure: {best_unsat_exp:.2f} ms"
        )
        return best_unsat_exp

    def build_exposure_list_from_good_exposure(self, good_exposure_ms: float) -> list[float]:
        """
        Fast exposure-list builder.

        It does not re-measure each exposure.
        It assumes that if Step 1 found a safe/good exposure, then lower
        exposures and small nearby exposures are safe enough for dark/bad-pixel capture.
        """
        vals = []

        for mult in EXPOSURE_MULTIPLIERS:
            exp_ms = r4(clamp(good_exposure_ms * mult, EXPOSURE_MIN_MS, EXPOSURE_MAX_MS))
            vals.append(exp_ms)

        vals = sorted(set(vals))

        if not vals:
            raise RuntimeError("No exposure values generated.")

        print(f"[INFO] Fast exposure list from selected exposure {good_exposure_ms:.2f} ms: {vals}")
        return vals

    def save_one_frame(self, img: np.ndarray, out_dir: str, stem: str):
        png_path = os.path.join(out_dir, stem + ".png")
        save_u16_png(png_path, img)

    def capture_and_save_set(self, filter_pos: int, exposure_list: list[float], n_frames: int, frame_type: str, out_dir: str):
        self.move_filter(filter_pos)

        for exp_ms in exposure_list:
            self.set_exposure(exp_ms)

            for i in range(1, n_frames + 1):
                img = self.grab_hsi_frame()
                m = compute_metrics(
                    img,
                    sat_thresh=SAT_THRESH,
                    sat_pixel_fraction_limit=SAT_PIXEL_FRACTION_LIMIT,
                    use_percentile_for_exposure=USE_PERCENTILE_FOR_EXPOSURE,
                    exposure_control_percentile=EXPOSURE_CONTROL_PERCENTILE,
                    good_min_dn=GOOD_MIN_DN,
                    good_max_dn=GOOD_MAX_DN,
                    bad_pixel_mask=None,
                )
                temp_c = self.get_camera_temperature_c()
                m["camera_temp_c"] = temp_c

                self.ui.update(img, f"{frame_type.upper()} {i}/{n_frames}", self.current_filter_pos, self.exposure_ms, m)

                stem = f"{frame_type}_F{filter_pos}_exp_{exp_ms:.2f}ms_{i:03d}_{now_ts()}"
                self.save_one_frame(img, out_dir, stem)

                self.meta_rows.append({
                    "file_name": stem + ".png",
                    "frame_type": frame_type,
                    "camera_temp_c": round(temp_c, 2),
                    "filter_pos": filter_pos,
                    "exposure_ms": exp_ms,
                    "gain": round(self.gain, 3),
                    "raw_max_dn": m["raw_max_dn"],
                    "mean_dn": round(m["mean_dn"], 3),
                    "p99_dn": round(m["p99_dn"], 3),
                    "p999_dn": round(m["p999_dn"], 3),
                    "control_dn": round(m["control_dn"], 3),
                    "sat_fraction": round(m["sat_fraction"], 8),
                    "saturated": bool(m["saturated"]),
                    "good": bool(m["good"]),
                    "timestamp": pd.Timestamp.now().isoformat(),
                })

                print(
                    f"[{frame_type.upper()}] exp={exp_ms:.2f} ms | {i}/{n_frames} | "
                    f"ctrl={m['control_dn']:.1f} | raw_max={m['raw_max_dn']} | sat={m['saturated']}"
                )
                time.sleep(FRAME_SETTLE_DELAY)

    def run(self):
        print(f"Saving to: {self.session_dir}")

        print("\n=== Initial move to open filter with extra delay ===")
        self.move_filter(FILTER_OPEN_POS)
        time.sleep(INITIAL_FILTER_SETTLE_SEC)
        self.discard_frames(10)

        print("\n=== Step 1: Find one usable open-filter exposure ===")
        good_exp_ms = self.optimize_open_filter_exposure()
        print(f"[INFO] Selected open-filter exposure: {good_exp_ms:.2f} ms")

        print("\n=== Step 2: Build multi-exposure list without re-measuring ===")
        exposure_list = self.build_exposure_list_from_good_exposure(good_exp_ms)
        print(f"[INFO] Exposures to use: {exposure_list}")

        print("\n=== Step 3: Capture bright frames on open filter ===")
        self.capture_and_save_set(
            filter_pos=FILTER_OPEN_POS,
            exposure_list=exposure_list,
            n_frames=N_BRIGHT_FRAMES_PER_EXPOSURE,
            frame_type="bright",
            out_dir=self.bright_dir,
        )

        print("\n=== Step 4: Capture dark frames on filter 7 with same exposures ===")
        self.capture_and_save_set(
            filter_pos=FILTER_DARK_POS,
            exposure_list=exposure_list,
            n_frames=N_DARK_FRAMES_PER_EXPOSURE,
            frame_type="dark",
            out_dir=self.dark_dir,
        )

        csv_path = os.path.join(self.session_dir, "metadata.csv")
        if self.meta_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.meta_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.meta_rows)

        with open(os.path.join(self.session_dir, "session_config.json"), "w") as f:
            json.dump({
                "FILTER_OPEN_POS": FILTER_OPEN_POS,
                "FILTER_DARK_POS": FILTER_DARK_POS,
                "EXPOSURE_START_MS": EXPOSURE_START_MS,
                "selected_exposure_ms": good_exp_ms,
                "exposure_list_ms": exposure_list,
                "N_BRIGHT_FRAMES_PER_EXPOSURE": N_BRIGHT_FRAMES_PER_EXPOSURE,
                "N_DARK_FRAMES_PER_EXPOSURE": N_DARK_FRAMES_PER_EXPOSURE,
                "crop_y0": CROP_Y0,
                "crop_y1": CROP_Y1,
                "gain": self.gain,
            }, f, indent=2)

        print(f"\n[DONE] Saved session to: {self.session_dir}")

    def cleanup(self):
        try:
            self.ui.close()
        except Exception:
            pass
        try:
            self.ids.stop()
        except Exception:
            pass
        try:
            self.ids.close()
        except Exception:
            pass
        try:
            self.motion.close()
        except Exception:
            pass


if __name__ == "__main__":
    app = None
    try:
        app = LDLSDarkCaptureApp()
        app.run()
    finally:
        if app is not None:
            app.cleanup()
