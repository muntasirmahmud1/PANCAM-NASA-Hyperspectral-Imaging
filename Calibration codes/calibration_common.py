import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ids_peak_common import PixelFormat
from ids_camera import IDSPeakCamera
from motor_control_v2 import MotionSystem


def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clamp(x, xmin, xmax):
    return max(xmin, min(xmax, x))


def r4(x):
    return round(float(x), 4)


def y_crop(gray: np.ndarray, crop_y0: int, crop_y1: int) -> np.ndarray:
    return gray[crop_y0:crop_y1, :]


def save_u16_png(path: str, img_u16: np.ndarray):
    ok = cv2.imwrite(path, img_u16.astype(np.uint16))
    if not ok:
        raise IOError(f"Failed to save PNG: {path}")


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0].copy()
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.round(np.mean(stack, axis=0)).astype(np.uint16)


def dark_correct_frame(bright: np.ndarray, dark: np.ndarray | None) -> np.ndarray:
    if dark is None:
        return bright.astype(np.float32)
    out = bright.astype(np.float32) - dark.astype(np.float32)
    out[out < 0] = 0
    return out


def compute_metrics(
    gray: np.ndarray,
    sat_thresh: int,
    sat_pixel_fraction_limit: float,
    use_percentile_for_exposure: bool = True,
    exposure_control_percentile: float = 99.9,
    good_min_dn: int | None = None,
    good_max_dn: int | None = None,
    bad_pixel_mask: np.ndarray | None = None,
) -> dict:
    arr = gray.astype(np.uint16)

    if bad_pixel_mask is not None:
        valid = ~bad_pixel_mask
        valid_pixels = arr[valid]
    else:
        valid_pixels = arr.ravel()

    if valid_pixels.size == 0:
        raise ValueError("No valid pixels left after masking.")

    raw_max_dn = int(np.max(arr))
    masked_max_dn = int(np.max(valid_pixels))
    mean_dn = float(np.mean(valid_pixels))
    p99 = float(np.percentile(valid_pixels, 99.0))
    p999 = float(np.percentile(valid_pixels, 99.9))

    if use_percentile_for_exposure:
        control_dn = float(np.percentile(valid_pixels, exposure_control_percentile))
    else:
        control_dn = float(masked_max_dn)

    sat_fraction = float(np.mean(valid_pixels >= sat_thresh))
    saturated = sat_fraction > sat_pixel_fraction_limit

    too_dim = False
    good = False
    if good_min_dn is not None and good_max_dn is not None:
        too_dim = control_dn < good_min_dn
        good = (good_min_dn <= control_dn <= good_max_dn) and (not saturated)

    return {
        "raw_max_dn": raw_max_dn,
        "masked_max_dn": masked_max_dn,
        "mean_dn": mean_dn,
        "p99_dn": p99,
        "p999_dn": p999,
        "control_dn": control_dn,
        "sat_fraction": sat_fraction,
        "saturated": saturated,
        "too_dim": too_dim,
        "good": good,
    }


def dark_correct_frame(bright: np.ndarray, dark: np.ndarray | None) -> np.ndarray:
    if dark is None:
        return bright.astype(np.float32)
    out = bright.astype(np.float32) - dark.astype(np.float32)
    out[out < 0] = 0
    return out


def apply_bad_pixel_mask_to_rows_mean(
    img: np.ndarray,
    row_start: int,
    row_end: int,
    bad_pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Average selected rows per column, optionally ignoring bad pixels.
    Returns 1D spectrum-like vector of length W.
    """
    sub = img[row_start:row_end, :]

    if bad_pixel_mask is None:
        return np.mean(sub, axis=0).astype(np.float32)

    sub_mask = bad_pixel_mask[row_start:row_end, :]
    out = np.zeros(sub.shape[1], dtype=np.float32)

    for x in range(sub.shape[1]):
        valid = ~sub_mask[:, x]
        if np.any(valid):
            out[x] = np.mean(sub[valid, x])
        else:
            out[x] = np.nan

    return out


def save_npy_and_png(img: np.ndarray, out_dir: str, stem: str, save_png: bool = True):
    np.save(os.path.join(out_dir, stem + ".npy"), img)
    if save_png:
        save_u16_png(os.path.join(out_dir, stem + ".png"), img)


def build_capture_metadata_row(
    *,
    file_name: str,
    frame_type: str,
    filter_pos: int,
    filter_name: str,
    matching_filter_pos: int | None,
    matching_filter_name: str | None,
    exposure_ms: float,
    gain: float,
    metrics: dict,
    use_y_crop: bool,
    crop_y0: int,
    crop_y1: int,
    ids_width: int,
    ids_height: int,
    bit_depth: int,
    max_dn: int,
    capture_group: str,
    candidate_for_nonlinearity: bool,
    candidate_for_flat_field: bool,
    candidate_for_relative_sensitivity: bool,
    is_common_exposure: bool,
    note: str = "",
) -> dict:
    return {
        "file_name": file_name,
        "frame_type": frame_type,
        "filter_pos": filter_pos,
        "filter_name": filter_name,
        "matching_filter_pos": matching_filter_pos,
        "matching_filter_name": matching_filter_name,
        "exposure_ms": round(float(exposure_ms), 4),
        "gain": round(float(gain), 4),
        "camera_temp_c": round(float(metrics.get("camera_temp_c", float("nan"))), 4),
        "raw_max_dn": int(metrics["raw_max_dn"]),
        "masked_max_dn": int(metrics.get("masked_max_dn", metrics["raw_max_dn"])),
        "mean_dn": round(float(metrics["mean_dn"]), 4),
        "p99_dn": round(float(metrics["p99_dn"]), 4),
        "p999_dn": round(float(metrics["p999_dn"]), 4),
        "control_dn": round(float(metrics["control_dn"]), 4),
        "sat_fraction": round(float(metrics["sat_fraction"]), 6),
        "saturated": bool(metrics["saturated"]),
        "good": bool(metrics["good"]),
        "use_y_crop": bool(use_y_crop),
        "crop_y0": int(crop_y0),
        "crop_y1": int(crop_y1),
        "ids_width": int(ids_width),
        "ids_height": int(ids_height),
        "bit_depth": int(bit_depth),
        "max_dn": int(max_dn),
        "capture_group": capture_group,
        "candidate_for_nonlinearity": bool(candidate_for_nonlinearity),
        "candidate_for_flat_field": bool(candidate_for_flat_field),
        "candidate_for_relative_sensitivity": bool(candidate_for_relative_sensitivity),
        "is_common_exposure": bool(is_common_exposure),
        "note": str(note),
        "timestamp": pd.Timestamp.now().isoformat(),
    }


def choose_best_exposure_from_metrics(exposure_metric_rows: list[dict], target_dn: float):
    """
    exposure_metric_rows example:
        [{"exposure_ms": 800.0, "control_dn": 540.0, "saturated": False}, ...]
    Returns the row whose control_dn is closest to target_dn among unsaturated rows.
    """
    valid = [r for r in exposure_metric_rows if not r["saturated"]]
    if not valid:
        return None
    return min(valid, key=lambda r: abs(r["control_dn"] - target_dn))

class LiveUI:
    def __init__(self, h: int, w: int, max_dn: int, pause_s: float = 0.01):
        self.max_dn = max_dn
        self.pause_s = pause_s

        plt.ion()
        self.fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = self.fig.add_gridspec(2, 2)

        self.ax_img = self.fig.add_subplot(gs[:, 0])
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        self.ax_hist = self.fig.add_subplot(gs[1, 1])

        self.im = self.ax_img.imshow(
            np.zeros((h, w), dtype=np.uint16),
            cmap="turbo",
            vmin=0,
            vmax=max_dn,
            aspect="auto",
            origin="upper",
        )
        self.ax_img.set_title("HSI Live")
        self.ax_img.set_xlabel("Spectral pixel")
        self.ax_img.set_ylabel("Spatial pixel")

        self.spec_x = np.arange(w)
        (self.line_spec,) = self.ax_spec.plot([], [], lw=1.4)
        self.ax_spec.set_xlim(0, w - 1)
        self.ax_spec.set_ylim(0, max_dn)
        self.ax_spec.grid(True, alpha=0.3)
        self.ax_spec.set_title("Mean Spectrum")
        self.ax_spec.set_xlabel("Spectral pixel")
        self.ax_spec.set_ylabel("DN")

        self.status_text = self.ax_spec.text(
            0.5, 0.95,
            "",
            transform=self.ax_spec.transAxes,
            ha="center",
            va="top",
            color="red",
            fontsize=11,
            fontweight="bold",
        )

    def update(
        self,
        gray: np.ndarray,
        mode_text: str,
        filter_pos: int | None,
        exposure_ms: float,
        metrics: dict,
    ):
        self.im.set_data(gray)

        mean_spec = gray.mean(axis=0, dtype=np.float32)
        self.line_spec.set_data(self.spec_x, mean_spec)

        filt_txt = "NA" if filter_pos is None else str(filter_pos)
        temp_c = metrics.get("camera_temp_c", float("nan"))

        self.ax_img.set_title(
            f"{mode_text}\n"
            f"Filter={filt_txt}  Exp={exposure_ms:.2f} ms  "
            f"Temp={temp_c:.2f} C  "
            f"RawMax={metrics['raw_max_dn']}  Ctrl={metrics['control_dn']:.1f}"
        )

        self.ax_hist.clear()
        vals = gray.ravel()
        self.ax_hist.hist(vals, bins=200, log=True)
        self.ax_hist.set_title("Histogram")
        self.ax_hist.set_xlabel("DN")
        self.ax_hist.set_ylabel("Count")

        if metrics["saturated"]:
            self.status_text.set_text(
                f"WARNING sat_frac={metrics['sat_fraction']:.6f}  ctrl={metrics['control_dn']:.1f}"
            )
        else:
            self.status_text.set_text("")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.pause_s)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


class CalibrationCameraSystem:
    def __init__(
        self,
        *,
        camera_index: int,
        ids_w: int,
        ids_h: int,
        crop_y0: int,
        crop_y1: int,
        pixel_format_entry: str,
        pipeline_output_format: PixelFormat,
        motor_port: str,
        filterwheel_port: str,
        gain: float,
        exposure_start_ms: float,
        use_software_trigger: bool,
        frame_settle_delay: float = 0.05,
        filter_settle_sec: float = 0.15,
        initial_filter_settle_sec: float = 0.5,
    ):
        self.ids_w = ids_w
        self.ids_h = ids_h
        self.crop_y0 = crop_y0
        self.crop_y1 = crop_y1
        self.frame_settle_delay = frame_settle_delay
        self.filter_settle_sec = filter_settle_sec
        self.initial_filter_settle_sec = initial_filter_settle_sec
        self.use_software_trigger = use_software_trigger

        self.ids = IDSPeakCamera(pipeline_output=pipeline_output_format)
        self.ids.open(index=camera_index)

        self.ids.node_map.FindNode("Width").SetValue(int(ids_w))
        self.ids.node_map.FindNode("Height").SetValue(int(ids_h))
        try:
            self.ids.node_map.FindNode("OffsetX").SetValue(0)
            self.ids.node_map.FindNode("OffsetY").SetValue(0)
        except Exception:
            pass

        self.ids.set_pixel_format_entry(pixel_format_entry)
        self.ids.set_pipeline_output_format(pipeline_output_format)
        self.ids.set_exposure_us(float(exposure_start_ms * 1000.0))
        self.gain = self.ids.set_gain(float(gain), selector="AnalogAll", clamp=True)

        if use_software_trigger:
            try:
                self.ids.set_trigger_software()
            except Exception:
                self.ids.set_trigger_off()
        else:
            self.ids.set_trigger_off()

        self.ids.start()
        self.exposure_ms = float(exposure_start_ms)

        self.motion = MotionSystem(
            motor_port=motor_port,
            fw_port=filterwheel_port,
            motor_baud=9600,
            fw_baud=4800,
        )
        self.motion.connect()
        self.current_filter_pos = None

    def grab_frame(self) -> np.ndarray:
        frame = self.ids.grab(timeout_ms=3000, do_software_trigger=self.use_software_trigger)
        return y_crop(frame, self.crop_y0, self.crop_y1)

    def discard_frames(self, n: int = 3):
        for _ in range(n):
            try:
                _ = self.grab_frame()
            except Exception:
                pass
            time.sleep(self.frame_settle_delay)

    def set_exposure(self, exposure_ms: float):
        exposure_ms = r4(exposure_ms)
        self.ids.set_exposure_us(exposure_ms * 1000.0)
        self.exposure_ms = exposure_ms
        time.sleep(self.frame_settle_delay)
        self.discard_frames(2)

    def move_filter(self, pos: int):
        if self.current_filter_pos == pos:
            return
        self.motion.set_filter(int(pos))
        self.current_filter_pos = int(pos)
        time.sleep(self.filter_settle_sec)
        self.discard_frames(2)

    def move_filter_initial(self, pos: int):
        self.motion.set_filter(int(pos))
        self.current_filter_pos = int(pos)
        time.sleep(self.initial_filter_settle_sec)
        self.discard_frames(2)

    def capture_frames(self, n: int) -> list[np.ndarray]:
        frames = []
        for _ in range(n):
            frames.append(self.grab_frame())
            time.sleep(self.frame_settle_delay)
        return frames

    def capture_average(self, n: int) -> np.ndarray:
        return average_frames(self.capture_frames(n))

    def close(self):
        try:
            self.ids.stop()
        except Exception:
            pass
        try:
            self.ids.close()
        except Exception:
            pass
        try:
            self.motion.disconnect()
        except Exception:
            pass
