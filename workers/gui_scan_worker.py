from __future__ import annotations

import time
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np

try:
    import pandas as pd
    import pvlib
except Exception:
    pd = None
    pvlib = None

from PySide6.QtCore import QObject, Signal, Slot

from pancam_scan_common_v2 import (
    build_range_list,
    crop_y,
    average_frames,
    compute_metrics,
    normalize_for_display,
    dark_correct_frame,
)


class GuiAzScanWorker(QObject):
    log = Signal(str)
    frame_ready = Signal(object)
    spectrum_ready = Signal(object)
    reconstruction_ready = Signal(object)
    status_ready = Signal(dict)
    session_ready = Signal(object)
    finished = Signal(bool, str)

    def __init__(
        self,
        *,
        scan_name: str,
        devices: dict,
        of_config: dict,
        cf_loaded: bool,
        calibration_dir: Path,
        output_day_dir: Path,
        view_to_motor_azimuth,
        view_to_motor_zenith,
        view_az_start_deg: float,
        view_az_end_deg: float,
        view_az_step_deg: float,
        view_zn_fixed_deg: float,
        view_zn_ref_deg: float,
        exposure_ms: float,
        exposure_min_ms: float,
        exposure_max_ms: float,
        gain: float,
        filter_start_pos: int,
        filter_scan_default: int,
        filter_dark_pos: int,
        filter_sequence: list[int],
        crop_y0: int,
        crop_y1: int,
        n_captures: int,
        camera_rot_steps: int,
        bit_depth: int = 12,
        good_min_frac: float = 0.60,
        good_max_frac: float = 0.98,
        exposure_control_percentile: float = 99.9,
        sat_pixel_fraction_limit: float = 1e-5,
        use_percentile_for_exposure: bool = True,
        save_raw_png: bool = True,
        save_raw_npy: bool = False,
        settle_time_sec: float = 0.10,
        frame_settle_delay: float = 0.01,
    ):
        super().__init__()

        self.scan_name = scan_name
        self.devices = devices
        self.of_config = of_config
        self.cf_loaded = cf_loaded
        self.calibration_dir = Path(calibration_dir)
        self.output_day_dir = Path(output_day_dir)

        self.view_to_motor_azimuth = view_to_motor_azimuth
        self.view_to_motor_zenith = view_to_motor_zenith

        self.view_az_start_deg = float(view_az_start_deg)
        self.view_az_end_deg = float(view_az_end_deg)
        self.view_az_step_deg = float(view_az_step_deg)
        self.view_zn_fixed_deg = float(view_zn_fixed_deg)
        self.view_zn_ref_deg = float(view_zn_ref_deg)

        self.exposure_ms = float(exposure_ms)
        self.exposure_min_ms = float(exposure_min_ms)
        self.exposure_max_ms = float(exposure_max_ms)
        self.gain = float(gain)

        self.filter_start_pos = int(filter_start_pos)
        self.filter_scan_default = int(filter_scan_default)
        self.filter_dark_pos = int(filter_dark_pos)
        self.filter_sequence = [int(x) for x in filter_sequence]

        self.crop_y0 = int(crop_y0)
        self.crop_y1 = int(crop_y1)
        self.n_captures = int(n_captures)
        self.camera_rot_steps = int(camera_rot_steps)

        self.bit_depth = int(bit_depth)
        self.max_dn = (1 << self.bit_depth) - 1

        self.good_min_frac = float(good_min_frac)
        self.good_max_frac = float(good_max_frac)
        self.exposure_control_percentile = float(exposure_control_percentile)
        self.sat_pixel_fraction_limit = float(sat_pixel_fraction_limit)
        self.use_percentile_for_exposure = bool(use_percentile_for_exposure)

        self.save_raw_png = bool(save_raw_png)
        self.save_raw_npy = bool(save_raw_npy)

        self.settle_time_sec = float(settle_time_sec)
        self.frame_settle_delay = float(frame_settle_delay)

        self.stop_requested = False
        self.current_filter_pos = None
        self.current_view_az_deg = None
        self.current_view_zn_deg = None

        self.recon_columns = []
        self.scan_frames_for_final_recon = []
        self.scan_setting_keys_used = set()
        self.scan_dark_map = {}
        self.meta_rows = []

        self.bad_pixel_mask = self._load_bad_pixel_mask()

    # ----------------------------------------------------
    # Basic helpers
    # ----------------------------------------------------

    def request_stop(self):
        self.stop_requested = True

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _now_iso(self):
        return datetime.now().isoformat(timespec="seconds")

    def _now_utc_iso(self):
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


    def _solar_angles_now(self):
        if pd is None or pvlib is None:
            return np.nan, np.nan

        try:
            lat = float(self.of_config.get("LATITUDE_DEG"))
            lon = float(self.of_config.get("LONGITUDE_DEG"))
            alt = float(self.of_config.get("ALTITUDE_M", 0.0))

            now_utc = pd.Timestamp.now(tz="UTC")

            solpos = pvlib.solarposition.get_solarposition(
                time=pd.DatetimeIndex([now_utc]),
                latitude=lat,
                longitude=lon,
                altitude=alt,
                method="nrel_numpy",
            )

            solar_az = float(solpos["azimuth"].iloc[0])
            solar_zn = float(solpos["apparent_zenith"].iloc[0])

            return solar_az, solar_zn

        except Exception:
            return np.nan, np.nan

    def _bool_of(self, key, default=False):
        return str(self.of_config.get(key, str(default))).strip().lower() in ["true", "1", "yes", "y"]

    def _make_setting_key(self, filter_pos, exposure_ms):
        return int(filter_pos), round(float(exposure_ms), 2)

    def _make_session_dirs(self):
        session_dir = self.output_day_dir / f"{self.scan_name}_{self._timestamp()}"
        self.raw_dir = session_dir / "raw"
        self.ref_dir = session_dir / "references"
        self.dark_dir = session_dir / "darks"
        self.products_dir = session_dir / "products"

        for d in [self.raw_dir, self.ref_dir, self.dark_dir, self.products_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return session_dir

    def _load_bad_pixel_mask(self):
        use_if_available = str(
            self.of_config.get("USE_BAD_PIXEL_MASK_IF_AVAILABLE", "True")
        ).lower() in ["true", "1", "yes"]

        if not use_if_available:
            return None

        mask_name = self.of_config.get("BAD_PIXEL_MASK_FILE", "bad_pixel_mask.npy")
        mask_path = self.calibration_dir / mask_name

        if not (self.cf_loaded and mask_path.exists()):
            return None

        try:
            mask = np.load(mask_path).astype(bool)
            expected_shape = (
                self.crop_y1 - self.crop_y0,
                int(self.of_config.get("SENSOR_WIDTH_PIXELS", 1440)),
            )

            if mask.shape != expected_shape:
                self.log.emit(f"Bad pixel mask ignored: shape={mask.shape}, expected={expected_shape}")
                return None

            self.log.emit(f"Bad pixel mask loaded: {mask_path}")
            return mask

        except Exception as e:
            self.log.emit(f"Bad pixel mask load failed: {e}")
            return None

    # ----------------------------------------------------
    # Hardware helpers
    # ----------------------------------------------------

    def _set_exposure(self, exp_ms):
        cam = self.devices["hsi"]
        exp_ms = max(self.exposure_min_ms, min(self.exposure_max_ms, float(exp_ms)))
        cam.set_exposure_us(exp_ms * 1000.0)
        self.exposure_ms = float(exp_ms)
        time.sleep(self.frame_settle_delay)

    def _set_filter(self, pos):
        motion = self.devices["motion"]
        pos = int(pos)

        if self.current_filter_pos == pos:
            return

        motion.set_filter(pos)
        self.current_filter_pos = pos
        time.sleep(self.settle_time_sec)

    def _move_to_view(self, view_az, view_zn):
        motion = self.devices["motion"]

        motor_zn = self.view_to_motor_zenith(view_zn)
        motor_az = self.view_to_motor_azimuth(view_az)

        motion.move_zenith(float(motor_zn))
        time.sleep(self.settle_time_sec)

        motion.move_azimuth(float(motor_az))
        time.sleep(self.settle_time_sec)

        self.current_view_az_deg = float(view_az) % 360.0
        self.current_view_zn_deg = float(view_zn)

        self.status_ready.emit(
            {
                "view_az": self.current_view_az_deg,
                "view_zn": self.current_view_zn_deg,
                "exposure_ms": self.exposure_ms,
                "filter_pos": self.current_filter_pos if self.current_filter_pos is not None else -1,
                "sat": "No",
                "wv_shift": "No",
            }
        )

    def _move_camera_rotation(self):
        try:
            self.devices["motion"].move_camera_rotation(int(self.camera_rot_steps))
            time.sleep(self.settle_time_sec)
        except Exception as e:
            self.log.emit(f"Camera internal motor warning: {e}")

    def _grab_one(self):
        cam = self.devices["hsi"]
        frame = cam.grab(timeout_ms=4000, do_software_trigger=False)
        return crop_y(frame, self.crop_y0, self.crop_y1)

    def _discard_frames(self, n=1):
        for _ in range(n):
            if self.stop_requested:
                return
            try:
                _ = self._grab_one()
            except Exception:
                pass
            time.sleep(self.frame_settle_delay)

    def _capture_average(self, label="capture"):
        frames = []

        for i in range(self.n_captures):
            if self.stop_requested:
                break

            gray = self._grab_one()
            frames.append(gray)

            self.frame_ready.emit(gray)
            self.spectrum_ready.emit(gray.mean(axis=0, dtype=np.float32))

            time.sleep(self.frame_settle_delay)

        if not frames:
            raise RuntimeError(f"No frames captured for {label}")

        return average_frames(frames)

    # ----------------------------------------------------
    # Metrics / save / reconstruction
    # ----------------------------------------------------

    def _metrics(self, img):
        return compute_metrics(
            gray=img,
            bit_depth=self.bit_depth,
            good_min_frac=self.good_min_frac,
            good_max_frac=self.good_max_frac,
            sat_pixel_fraction_limit=self.sat_pixel_fraction_limit,
            exposure_control_percentile=self.exposure_control_percentile,
            use_percentile_for_exposure=self.use_percentile_for_exposure,
            bad_pixel_mask=self.bad_pixel_mask,
        )

    def _save_frame(self, out_dir, stem, img):
        if self.save_raw_png:
            cv2.imwrite(str(out_dir / f"{stem}.png"), img.astype(np.uint16))

        if self.save_raw_npy:
            np.save(str(out_dir / f"{stem}.npy"), img.astype(np.uint16))

    def _append_meta(self, row):
        self.meta_rows.append(row)

    def _metadata_row(self, *, frame_type, stem, metrics, note=""):
        solar_az, solar_zn = self._solar_angles_now()

        return {
            "utc_timestamp": self._now_utc_iso(),
            "file_name": stem,
            "measurement_type": frame_type,
            "solar_azimuth_deg": None if not np.isfinite(solar_az) else round(float(solar_az), 6),
            "solar_zenith_deg": None if not np.isfinite(solar_zn) else round(float(solar_zn), 6),
            "view_azimuth_deg": None if self.current_view_az_deg is None else round(float(self.current_view_az_deg), 4),
            "view_zenith_deg": None if self.current_view_zn_deg is None else round(float(self.current_view_zn_deg), 4),
            "camera_rot_steps": int(self.camera_rot_steps),
            "filter_pos": self.current_filter_pos,
            "exposure_ms": round(float(self.exposure_ms), 4),
            "gain": round(float(self.gain), 4),
            "masked_max_dn": int(metrics["masked_max_dn"]),
            "mean_dn": round(float(metrics["mean_dn"]), 4),
            "p999_dn": round(float(metrics["p999_dn"]), 4),
            "sat_percent": round(float(metrics["sat_percent"]), 3),
            "saturated": bool(metrics["saturated"]),
            "too_dim": bool(metrics["too_dim"]),
            "too_bright": bool(metrics["too_bright"]),
            "good": bool(metrics["good"]),
            "control_dn": round(float(metrics["control_dn"]), 4),
            "note": note,
        }

    def _add_reconstruction_column(self, img):
        col = img.mean(axis=1, dtype=np.float32)

        if self._bool_of("USE_EXPOSURE_NORMALIZED_RECON", True):
            col = col / max(float(self.exposure_ms), 1e-6)

        self.recon_columns.append(col.reshape(-1, 1))
        recon = np.hstack(self.recon_columns)

        if self._bool_of("USE_DISPLAY_NORMALIZED_RECON", True):
            p_low = float(self.of_config.get("RECON_DISPLAY_P_LOW", 1))
            p_high = float(self.of_config.get("RECON_DISPLAY_P_HIGH", 99))
            recon_ui = normalize_for_display(recon, p_low=p_low, p_high=p_high)
        else:
            recon_ui = recon

        self.reconstruction_ready.emit(recon_ui)


    # ----------------------------------------------------
    # Full scan workflow
    # ----------------------------------------------------

    def _capture_labeled_frame(self, out_dir, label, frame_type, note=""):
        self._discard_frames(1)
        img = self._capture_average(label)
        m = self._metrics(img)

        stem = f"{label}_{self._timestamp()}"
        self._save_frame(out_dir, stem, img)

        self.frame_ready.emit(img)
        self.spectrum_ready.emit(img.mean(axis=0, dtype=np.float32))

        row = self._metadata_row(frame_type=frame_type, stem=stem, metrics=m, note=note)
        self._append_meta(row)

        return img, m

    def _capture_scan_darks_for_used_settings(self):
        self.log.emit("Capturing scan darks for used settings...")

        for filt_used, exp_used in sorted(self.scan_setting_keys_used):
            if self.stop_requested:
                return

            self._set_filter(self.filter_dark_pos)
            self._set_exposure(exp_used)

            dark_img = self._capture_average(f"scan_dark_F{filt_used}_{exp_used:.2f}ms")
            m = self._metrics(dark_img)

            key = self._make_setting_key(filt_used, exp_used)
            self.scan_dark_map[key] = dark_img.copy()

            stem = f"scan_dark_F{filt_used}_exp_{exp_used:.2f}ms_{self._timestamp()}".replace(" ", "_")
            self._save_frame(self.dark_dir, stem, dark_img)

            self._append_meta(
                self._metadata_row(
                    frame_type="scan_dark",
                    stem=stem,
                    metrics=m,
                    note=f"dark matched to scan filter={filt_used}, exposure={exp_used:.2f} ms",
                )
            )

    def _build_final_dark_corrected_reconstruction(self):
        if not self.scan_frames_for_final_recon:
            return

        recon_cols = []

        for item in self.scan_frames_for_final_recon:
            img = item["img"]
            filt = item["filter_pos"]
            exp_ms = item["exposure_ms"]

            key = self._make_setting_key(filt, exp_ms)
            dark = self.scan_dark_map.get(key, None)

            corrected = dark_correct_frame(img, dark)

            col = corrected.mean(axis=1, dtype=np.float32)

            if self._bool_of("USE_EXPOSURE_NORMALIZED_RECON", True):
                col = col / max(float(exp_ms), 1e-6)

            recon_cols.append(col.reshape(-1, 1))

        recon_final = np.hstack(recon_cols)

        if self._bool_of("USE_DISPLAY_NORMALIZED_RECON", True):
            p_low = float(self.of_config.get("RECON_DISPLAY_P_LOW", 1))
            p_high = float(self.of_config.get("RECON_DISPLAY_P_HIGH", 99))
            recon_display = normalize_for_display(recon_final, p_low=p_low, p_high=p_high)
        else:
            recon_display = recon_final

        out_png = self.products_dir / "final_dark_corrected_reconstruction.png"

        img8 = np.clip(recon_display * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_png), img8)

        self.reconstruction_ready.emit(recon_display)
        self.log.emit(f"Final dark-corrected reconstruction saved: {out_png.name}")

    def _write_outputs(self, session_dir):
        meta_path = session_dir / "metadata.csv"
        if self.meta_rows:
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.meta_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.meta_rows)

        config_path = session_dir / "session_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scan_name": self.scan_name,
                    "view_az_start_deg": self.view_az_start_deg,
                    "view_az_end_deg": self.view_az_end_deg,
                    "view_az_step_deg": self.view_az_step_deg,
                    "view_zn_fixed_deg": self.view_zn_fixed_deg,
                    "view_zn_ref_deg": self.view_zn_ref_deg,
                    "crop_y0": self.crop_y0,
                    "crop_y1": self.crop_y1,
                    "filter_sequence": self.filter_sequence,
                    "save_raw_png": self.save_raw_png,
                    "save_raw_npy": self.save_raw_npy,
                },
                f,
                indent=2,
            )

    @Slot()
    def run(self):
        try:
            if self.devices.get("hsi") is None:
                raise RuntimeError("HSI camera is not connected.")
            if self.devices.get("motion") is None:
                raise RuntimeError("Motion system is not connected.")

            session_dir = self._make_session_dirs()
            self.log.emit(f"Starting full scan workflow: {self.scan_name}")
            self.log.emit(f"Saving scan to: {session_dir}")

            # 1. Move camera internal
            self._move_camera_rotation()

            # 2. Starting zenith reference
            self.log.emit("Moving to starting zenith reference pose...")
            self._move_to_view(self.view_az_start_deg, self.view_zn_ref_deg)

            self.log.emit("Optimizing START reference exposure/filter...")
            ref_filter, ref_exp, _ = self._optimize_filter_and_exposure(
                start_filter=self.filter_start_pos,
                start_exposure_ms=self.exposure_ms,
                mode_text="START REF",
            )

            self._set_filter(ref_filter)
            self._set_exposure(ref_exp)

            self.log.emit("Capturing START bright reference...")
            self._capture_labeled_frame(
                self.ref_dir,
                "reference_start_bright",
                "reference_start_bright",
                note="starting bright reference before scan",
            )

            self.log.emit("Capturing START dark reference...")
            self._set_filter(self.filter_dark_pos)
            self._capture_labeled_frame(
                self.dark_dir,
                "reference_start_dark",
                "reference_start_dark",
                note="dark matched to starting bright reference",
            )

            if self.stop_requested:
                self.finished.emit(False, "Scan stopped before scan loop.")
                return

            # 3. Move to scan start and optimize scan
            self.log.emit("Moving to scan start pose...")
            self._move_to_view(self.view_az_start_deg, self.view_zn_fixed_deg)

            self.log.emit("Optimizing scan exposure/filter at first scan position...")
            scan_filter, scan_exp, _ = self._optimize_filter_and_exposure(
                start_filter=self.filter_scan_default,
                start_exposure_ms=ref_exp,
                mode_text="SCAN START",
            )

            self._set_filter(scan_filter)
            self._set_exposure(scan_exp)

            # 4. Scan loop
            az_list = build_range_list(
                self.view_az_start_deg,
                self.view_az_end_deg,
                self.view_az_step_deg,
            )

            self.log.emit(f"Beginning azimuth scan with {len(az_list)} steps...")

            for i, view_az in enumerate(az_list, start=1):
                if self.stop_requested:
                    self.finished.emit(False, "Scan stopped by user.")
                    return

                self.log.emit(
                    f"[SCAN] {i}/{len(az_list)} View Az={view_az:.3f}°, "
                    f"View Zn={self.view_zn_fixed_deg:.3f}°"
                )

                self._move_to_view(view_az, self.view_zn_fixed_deg)

                img = self._capture_average(f"{self.scan_name} step {i}")
                m = self._metrics(img)

                too_bright = m["saturated"] or m["too_bright"]
                too_dim = m["too_dim"]

                brightest_possible = (
                    self.current_filter_pos == self.filter_sequence[0]
                    and self.exposure_ms >= self.exposure_max_ms - 1e-6
                )

                darkest_possible = (
                    self.current_filter_pos == self.filter_sequence[-1]
                    and self.exposure_ms <= self.exposure_min_ms + 1e-6
                )

                should_reopt = (
                    too_bright
                    or (too_dim and not brightest_possible)
                )

                if should_reopt and not darkest_possible:
                    self.log.emit(
                        f"[SCAN] Re-optimizing at View Az={view_az:.3f} | "
                        f"sat={m['saturated']} too_dim={too_dim} too_bright={m['too_bright']}"
                    )

                    new_filter, new_exp, _ = self._optimize_filter_and_exposure(
                        start_filter=self.current_filter_pos,
                        start_exposure_ms=self.exposure_ms,
                        mode_text=f"REOPT AZ={view_az:.2f}",
                        max_iters_per_filter=12,
                    )

                    self._set_filter(new_filter)
                    self._set_exposure(new_exp)

                    img = self._capture_average(f"{self.scan_name} step {i} after reopt")
                    m = self._metrics(img)

                elif too_dim and brightest_possible:
                    self.log.emit(
                        "[SCAN] Too dim at brightest setting. Skipping re-optimization."
                    )

                elif too_bright and darkest_possible:
                    self.log.emit(
                        "[SCAN] Saturated/too bright at darkest setting. Skipping re-optimization."
                    )

                key = self._make_setting_key(self.current_filter_pos, self.exposure_ms)
                self.scan_setting_keys_used.add(key)

                stem = f"{self.scan_name}_{i:04d}_viewaz_{view_az:.2f}_{self._timestamp()}"
                self._save_frame(self.raw_dir, stem, img)

                self._append_meta(
                    self._metadata_row(
                        frame_type=self.scan_name,
                        stem=stem,
                        metrics=m,
                        note="azimuth scan frame",
                    )
                )

                self.scan_frames_for_final_recon.append(
                    {
                        "img": img.copy(),
                        "filter_pos": int(self.current_filter_pos),
                        "exposure_ms": round(float(self.exposure_ms), 2),
                    }
                )

                self.frame_ready.emit(img)
                self.spectrum_ready.emit(img.mean(axis=0, dtype=np.float32))
                self._add_reconstruction_column(img)

                self.status_ready.emit(
                    {
                        "view_az": float(view_az),
                        "view_zn": float(self.view_zn_fixed_deg),
                        "exposure_ms": float(self.exposure_ms),
                        "filter_pos": int(self.current_filter_pos),
                        "sat": "Yes" if m["saturated"] else "No",
                        "wv_shift": "No",
                    }
                )

            # 5. Scan darks and final reconstruction
            self._capture_scan_darks_for_used_settings()
            self._build_final_dark_corrected_reconstruction()

            # 6. Ending reference
            self.log.emit("Moving to ending zenith reference pose...")
            self._move_to_view(self.view_az_end_deg, self.view_zn_ref_deg)

            self.log.emit("Optimizing END reference exposure/filter...")
            end_filter, end_exp, _ = self._optimize_filter_and_exposure(
                start_filter=ref_filter,
                start_exposure_ms=ref_exp,
                mode_text="END REF",
            )

            self._set_filter(end_filter)
            self._set_exposure(end_exp)

            self.log.emit("Capturing END bright reference...")
            self._capture_labeled_frame(
                self.ref_dir,
                "reference_end_bright",
                "reference_end_bright",
                note="ending bright reference after scan",
            )

            self.log.emit("Capturing END dark reference...")
            self._set_filter(self.filter_dark_pos)
            self._capture_labeled_frame(
                self.dark_dir,
                "reference_end_dark",
                "reference_end_dark",
                note="dark matched to ending bright reference",
            )

            self._write_outputs(session_dir)

            self.session_ready.emit({
                "session_dir": str(session_dir),
                "scan_name": self.scan_name,
            })

            self.finished.emit(True, f"Full scan finished: {self.scan_name}")

        except Exception as e:
            self.finished.emit(False, f"Scan failed: {e}")
