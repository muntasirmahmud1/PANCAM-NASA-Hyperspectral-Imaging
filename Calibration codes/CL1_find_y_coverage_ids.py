import time
import numpy as np
import matplotlib.pyplot as plt

from ids_peak_common import PixelFormat

from calibration_common import (
    CalibrationCameraSystem,
    LiveUI,
    compute_metrics,
    r4,
)


# =========================================================
# USER SETTINGS
# =========================================================

# # Windows
# MOTOR_PORT = "COM23"
# FILTERWHEEL_PORT = "COM19"

# Jetson
MOTOR_PORT = "/dev/ttyUSB0"
FILTERWHEEL_PORT = "/dev/ttyACM0"

CAMERA_INDEX = 0

IDS_W = 1440
IDS_H = 1080

# For this script, use full frame first so we can detect coverage
CROP_Y0 = 0
CROP_Y1 = IDS_H

PIXEL_FORMAT_ENTRY = "Mono12"
PIPELINE_OUTPUT_FORMAT = PixelFormat.MONO_12

USE_SOFTWARE_TRIGGER = False
GAIN = 1.0

BIT_DEPTH = 12
MAX_DN = (1 << BIT_DEPTH) - 1

# Exposure search
EXPOSURE_START_MS = 500
EXPOSURE_MAX_MS = 2000.0
EXPOSURE_MULTIPLIER = 1.15
N_FRAMES_PER_EXPOSURE = 3

# Stop when image is bright enough but not saturated
TARGET_P99_FRAC = 0.90
SAT_THRESH = int(0.99 * MAX_DN)
SAT_FRAC_LIMIT = 1e-5

# Row coverage detection
SMOOTH_KERNEL = 15          # moving-average kernel over row profile
ROW_THRESHOLD_FRAC = 0.3 #0.15   # threshold relative to row-profile dynamic range
MIN_RUN_ROWS = 20           # minimum continuous illuminated run to accept

# Visualization
PAUSE_S = 0.01


# =========================================================
# HELPERS (script-specific)
# =========================================================

def moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()

    k = int(k)
    if k % 2 == 0:
        k += 1

    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, kernel, mode="valid")


def longest_true_run(mask: np.ndarray):
    """
    Return (start_idx, end_idx) inclusive for the longest True run.
    If no True values, return None.
    """
    best_start = None
    best_end = None
    best_len = 0

    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            end = i - 1
            run_len = end - start + 1
            if run_len > best_len:
                best_len = run_len
                best_start, best_end = start, end
            start = None

    if start is not None:
        end = len(mask) - 1
        run_len = end - start + 1
        if run_len > best_len:
            best_len = run_len
            best_start, best_end = start, end

    if best_len == 0:
        return None

    return best_start, best_end


def find_covered_rows(gray: np.ndarray):
    """
    Compute row coverage using row-mean profile across X.
    """
    row_profile = gray.mean(axis=1).astype(np.float64)
    row_profile_smooth = moving_average_1d(row_profile, SMOOTH_KERNEL)

    prof_min = float(np.min(row_profile_smooth))
    prof_max = float(np.max(row_profile_smooth))
    dyn = prof_max - prof_min

    if dyn <= 0:
        return {
            "row_profile": row_profile,
            "row_profile_smooth": row_profile_smooth,
            "threshold": prof_max,
            "covered_mask": np.zeros_like(row_profile, dtype=bool),
            "y0": None,
            "y1": None,
        }

    threshold = prof_min + ROW_THRESHOLD_FRAC * dyn
    covered_mask = row_profile_smooth >= threshold

    run = longest_true_run(covered_mask)

    y0 = None
    y1 = None
    if run is not None:
        y0, y1 = run
        if (y1 - y0 + 1) < MIN_RUN_ROWS:
            y0, y1 = None, None

    return {
        "row_profile": row_profile,
        "row_profile_smooth": row_profile_smooth,
        "threshold": threshold,
        "covered_mask": covered_mask,
        "y0": y0,
        "y1": y1,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    cam_sys = None
    ui = None

    try:
        cam_sys = CalibrationCameraSystem(
            camera_index=CAMERA_INDEX,
            ids_w=IDS_W,
            ids_h=IDS_H,
            crop_y0=CROP_Y0,
            crop_y1=CROP_Y1,
            pixel_format_entry=PIXEL_FORMAT_ENTRY,
            pipeline_output_format=PIPELINE_OUTPUT_FORMAT,
            motor_port=MOTOR_PORT,
            filterwheel_port=FILTERWHEEL_PORT,
            gain=GAIN,
            exposure_start_ms=EXPOSURE_START_MS,
            use_software_trigger=USE_SOFTWARE_TRIGGER,
            frame_settle_delay=0.05,
            filter_settle_sec=0.15,
            initial_filter_settle_sec=2.0,
        )

        ui = LiveUI(h=CROP_Y1 - CROP_Y0, w=IDS_W, max_dn=MAX_DN, pause_s=PAUSE_S)

        exposure_ms = float(EXPOSURE_START_MS)
        best_gray = None
        best_metrics = None

        print("Starting exposure search...")
        print(f"Camera gain set to {cam_sys.gain:.2f}")

        while True:
            cam_sys.set_exposure(exposure_ms)

            frames = []
            for _ in range(N_FRAMES_PER_EXPOSURE):
                gray = cam_sys.grab_frame()
                frames.append(gray)

                m = compute_metrics(
                    gray,
                    sat_thresh=SAT_THRESH,
                    sat_pixel_fraction_limit=SAT_FRAC_LIMIT,
                    use_percentile_for_exposure=True,
                    exposure_control_percentile=99.9,
                    good_min_dn=None,
                    good_max_dn=None,
                    bad_pixel_mask=None,
                )
                ui.update(gray, "Exposure search", cam_sys.current_filter_pos, cam_sys.exposure_ms, m)
                time.sleep(0.03)

            gray_avg = np.round(np.mean(np.stack(frames, axis=0).astype(np.float32), axis=0)).astype(np.uint16)

            m = compute_metrics(
                gray_avg,
                sat_thresh=SAT_THRESH,
                sat_pixel_fraction_limit=SAT_FRAC_LIMIT,
                use_percentile_for_exposure=True,
                exposure_control_percentile=99.9,
                good_min_dn=None,
                good_max_dn=None,
                bad_pixel_mask=None,
            )

            coverage = find_covered_rows(gray_avg)
            ui.update(gray_avg, "Coverage detection", cam_sys.current_filter_pos, cam_sys.exposure_ms, m)

            print(
                f"Exp={cam_sys.exposure_ms:.2f} ms | "
                f"raw_max={m['raw_max_dn']} | "
                f"p99={m['p99_dn']:.1f} | "
                f"p99.9={m['p999_dn']:.1f} | "
                f"sat={m['saturated']}"
            )

            best_gray = gray_avg.copy()
            best_metrics = m.copy()

            # Stop if good brightness reached without saturation
            if (not m["saturated"]) and (m["p99_dn"] >= TARGET_P99_FRAC * MAX_DN):
                print("Reached target brightness for coverage detection.")
                break

            # Stop if saturation or exposure limit reached
            if m["saturated"] or exposure_ms >= EXPOSURE_MAX_MS:
                print("Stopping due to saturation or exposure limit.")
                break

            exposure_ms = min(EXPOSURE_MAX_MS, r4(exposure_ms * EXPOSURE_MULTIPLIER))

        # Final coverage analysis
        coverage = find_covered_rows(best_gray)
        y0 = coverage["y0"]
        y1 = coverage["y1"]

        print("\n===== Y-axis coverage result =====")
        if y0 is None or y1 is None:
            print("Could not confidently determine covered Y range.")
        else:
            print(f"Covered rows: {y0} to {y1}")
            print("Suggested crop:")
            print(f"  CROP_Y0 = {y0}")
            print(f"  CROP_Y1 = {y1 + 1}")

            if y0 > 0:
                print(f"Top rows to crop away    : 0 to {y0 - 1}")
            else:
                print("Top rows to crop away    : none")

            if y1 < IDS_H - 1:
                print(f"Bottom rows to crop away : {y1 + 1} to {IDS_H - 1}")
            else:
                print("Bottom rows to crop away : none")

        # Plot final image with detected coverage
        plt.figure(figsize=(12, 5))
        plt.imshow(best_gray, cmap="gray", aspect="auto", origin="upper")
        if y0 is not None and y1 is not None:
            plt.axhline(y0, color="lime", linestyle="--", linewidth=1.5, label=f"y0={y0}")
            plt.axhline(y1, color="cyan", linestyle="--", linewidth=1.5, label=f"y1={y1}")
            plt.legend()
        plt.title("Final HSI image with detected covered Y-range")
        plt.xlabel("Spectral pixel (X)")
        plt.ylabel("Spatial pixel (Y)")
        plt.tight_layout()
        plt.show()

        # Plot row profile
        y = np.arange(best_gray.shape[0])

        plt.figure(figsize=(12, 5))
        plt.plot(y, coverage["row_profile"], label="Row mean", linewidth=1.0)
        plt.plot(y, coverage["row_profile_smooth"], label="Smoothed row mean", linewidth=2.0)
        plt.axhline(coverage["threshold"], linestyle="--", label=f"Threshold={coverage['threshold']:.2f}")
        if y0 is not None and y1 is not None:
            plt.axvline(y0, color="lime", linestyle="--", label=f"y0={y0}")
            plt.axvline(y1, color="cyan", linestyle="--", label=f"y1={y1}")
        plt.title("Row profile used to detect illuminated Y coverage")
        plt.xlabel("Y row")
        plt.ylabel("Mean DN across X")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\nClose the figures when done.")
        plt.show(block=True)

    finally:
        try:
            if ui is not None:
                ui.close()
        except Exception:
            pass

        try:
            if cam_sys is not None:
                cam_sys.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()