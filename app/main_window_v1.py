from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import glob

from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QSizePolicy,
)
from PySide6.QtCore import QTimer, Qt, QSignalBlocker, QThread
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QFont, QColor

from PANCAM_ui_index import Ui_MainWindow
from workers.gui_scan_worker import GuiAzScanWorker

import numpy as np

from ids_peak_common import PixelFormat
from ids_camera import IDSPeakCamera
from pancam_scan_common_v2 import crop_y
from color_camera import ColorCamera
from motor_control_v2 import MotionSystem


# Optional packages
try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import requests
except Exception:
    requests = None

try:
    import pandas as pd
    import pvlib
except Exception:
    pd = None
    pvlib = None


# ============================================================
# DEFAULT PANCAM STATE
# ============================================================

HOME_VIEW_AZ_DEG = 90.0
HOME_VIEW_ZN_DEG = 90.0
DEFAULT_EXPOSURE_MS = 500.0
DEFAULT_FILTER_POS = 3
CAMERA_INTERNAL_START_STEPS = -875

STATUS_GRAY = "background-color: rgb(192, 191, 188);"
STATUS_GREEN = "background-color: #34D481; color: white; font-weight: bold;"

LIVE_CAMERA_TIMER_MS = 700

# ============================================================
# WEATHER CODE MAP FOR OPEN-METEO
# ============================================================

WEATHER_CODE_MAP = {
    0: "Sunny",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Cloudy",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy showers",
    95: "Thunderstorm",
}


# ============================================================
# MAIN GUI CLASS
# ============================================================

class PANCAM_UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("PANCAM")

        self.base_dir = Path(__file__).resolve().parents[1]
        self.operation_dir = self.base_dir / "Operation File"
        self.calibration_dir = self.base_dir / "Calibration File"
        self.data_root = self.base_dir / "data"
        self.schedule_dir = self.base_dir / "Schedule"
        self.log_file_path = None
        self.setup_daily_log_file()

        self.of_config = {}
        self.cf_loaded = False

        self.connected = {
            "hsi": False,
            "rgb": False,
            "tracker": False,
            "fw": False,
            "imu": False,
            "temp": False,
        }

        self.devices = {
            "hsi": None,
            "rgb": None,
            "tracker": None,
            "fw": None,
            "motion": None,
        }

        self.current_view_az_deg = HOME_VIEW_AZ_DEG
        self.current_view_zn_deg = HOME_VIEW_ZN_DEG

        # Internal motor angles are still needed for motor commands/debugging
        self.current_az_motor_deg = self.view_to_motor_azimuth(self.current_view_az_deg)
        self.current_zn_motor_deg = self.view_to_motor_zenith(self.current_view_zn_deg)
        self.current_exp_ms = DEFAULT_EXPOSURE_MS
        self.current_fw_pos = DEFAULT_FILTER_POS
        self.current_camera_steps = CAMERA_INTERNAL_START_STEPS

        self.latest_hsi_frame = None
        self.latest_reconstruction = None
        self.current_camera_temp_c = float("nan")
        self.current_sat = "No"
        self.current_wv_shift = "No"

        self.current_sky_target = "sun"
        self.current_target_az_deg = None
        self.current_target_zn_deg = None

        self.skyfield_ts = None
        self.skyfield_eph = None

        self._setup_start_page()
        self._setup_log_area()
        self._setup_button_connections()
        self._populate_startup_files()
        self.refresh_device_ports()
        self._setup_initial_dashboard_state()
        self._setup_dashboard_controls()

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_header_date_time_only)
        self.clock_timer.start(1000)

        self.sky_target_timer = QTimer(self)
        self.sky_target_timer.timeout.connect(self.update_sky_target_header)
        self.sky_target_timer.start(30 * 1000)

        self.update_header_date_time_only()
        self.update_sky_target_header()

        self.weather_timer = QTimer(self)
        self.weather_timer.timeout.connect(self.update_weather)
        self.weather_timer.start(10 * 60 * 1000)

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_views)
        self._hsi_busy = False
        self.camera_timer.start(LIVE_CAMERA_TIMER_MS)

        self.rgb_scan_timer = QTimer(self)
        self.rgb_scan_timer.timeout.connect(self.show_rgb_frame)

    # ========================================================
    # STARTUP SETUP
    # ========================================================

    def _setup_start_page(self):
        self.icon_only_widget.setHidden(True)

        self.stackedWidget.setCurrentIndex(4)
        self.settings_1.setChecked(True)
        self.settings_2.setChecked(True)

    def _setup_log_area(self):
        self.log_text = self.plainTextEdit_log
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(5000)

        self.log("PANCAM software started.")


    def _setup_button_connections(self):
        # Sidebar navigation
        self.dashboard_1.clicked.connect(self.switch_to_dashboard_page)
        self.dashboard_2.clicked.connect(self.switch_to_dashboard_page)

        self.analysis_1.clicked.connect(self.switch_to_analysis_page)
        self.analysis_2.clicked.connect(self.switch_to_analysis_page)

        self.lab_1.clicked.connect(self.switch_to_lab_page)
        self.lab_2.clicked.connect(self.switch_to_lab_page)

        self.calibration_1.clicked.connect(self.switch_to_calibration_page)
        self.calibration_2.clicked.connect(self.switch_to_calibration_page)

        self.settings_1.clicked.connect(self.switch_to_settings_page)
        self.settings_2.clicked.connect(self.switch_to_settings_page)

        # Settings page buttons
        self.pushButton_connect_all.clicked.connect(self.connect_all_devices)
        self.pushButton_disconnect_all.clicked.connect(self.disconnect_all_devices)
        self.pushButton_reset_all.clicked.connect(self.reset_all_devices_placeholder)

        self.pushButton_browse_of.clicked.connect(self.browse_operation_file)
        self.pushButton_load_of.clicked.connect(self.load_operation_file)

        self.pushButton_browse_cf.clicked.connect(self.browse_calibration_file_or_folder)
        self.pushButton_load_cf.clicked.connect(self.load_calibration_files)

    def _setup_initial_dashboard_state(self):
        self.set_status_disconnected()

        self.label_hsi_camera.setAlignment(Qt.AlignCenter)
        self.label_mean_spectrum.setAlignment(Qt.AlignCenter)
        self.label_rgb_camera.setAlignment(Qt.AlignCenter)
        self.label_reconstructed_image.setAlignment(Qt.AlignCenter)
        self.label_no2_profile.setAlignment(Qt.AlignCenter)

        self.label_hsi_camera.setText("HSI camera not connected")
        self.label_mean_spectrum.setText("Mean spectrum not available")
        self.label_rgb_camera.setText("RGB camera not connected")
        self.label_reconstructed_image.setText("Reconstructed image not available")
        self.label_no2_profile.setText("NO₂ retrieval not available")

        for label in [
            self.label_hsi_camera,
            self.label_mean_spectrum,
            self.label_rgb_camera,
            self.label_reconstructed_image,
            self.label_no2_profile,
        ]:
            label.setAlignment(Qt.AlignCenter)
            label.setScaledContents(False)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            label.setMinimumSize(50, 50)

    # ========================================================
    # PAGE SWITCHING
    # ========================================================

    def switch_to_dashboard_page(self):
        self.stackedWidget.setCurrentIndex(0)

    def switch_to_analysis_page(self):
        self.stackedWidget.setCurrentIndex(1)

    def switch_to_lab_page(self):
        self.stackedWidget.setCurrentIndex(2)

    def switch_to_calibration_page(self):
        self.stackedWidget.setCurrentIndex(3)

    def switch_to_settings_page(self):
        self.stackedWidget.setCurrentIndex(4)

    # ========================================================
    # FILE INITIALIZATION
    # ========================================================

    def _populate_startup_files(self):
        self.operation_dir.mkdir(exist_ok=True)
        self.calibration_dir.mkdir(exist_ok=True)

        of_files = sorted(self.operation_dir.glob("*.txt"))
        if of_files:
            self.lineEdit_load_of.setText(str(of_files[0]))
        else:
            self.lineEdit_load_of.setText(str(self.operation_dir))

        self.lineEdit_load_cf.setText(str(self.calibration_dir))

    # ========================================================
    # DEVICE PORT DETECTION
    # ========================================================

    def refresh_device_ports(self):
        serial_ports = self.get_serial_ports()
        hsi_ports = self.get_ids_camera_ports()
        rgb_ports = self.get_rgb_camera_ports()

        self.fill_combo(self.comboBox_connect_hsi_camera, hsi_ports)
        self.fill_combo(self.comboBox_connect_rgb_camera, rgb_ports)
        self.fill_combo(self.comboBox_connect_tracker, serial_ports)
        self.fill_combo(self.comboBox_connect_fw, serial_ports)

        # Future devices
        self.comboBox_connect_imu.clear()
        self.comboBox_connect_imu.addItem("Future")
        self.comboBox_connect_imu.setEnabled(False)

        self.comboBox_temperature_controller.clear()
        self.comboBox_temperature_controller.addItem("Future")
        self.comboBox_temperature_controller.setEnabled(False)

        self.log("Device port list refreshed.")
        self.auto_select_known_ports()

    def get_serial_ports(self):
        ports = [("Select", None)]

        # Method 1: pyserial
        if list_ports is not None:
            for p in list_ports.comports():
                label = f"{p.device} | {p.description}"
                ports.append((label, p.device))

        # Method 2: Linux fallback
        for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial/by-id/*"]:
            for dev in sorted(glob.glob(pattern)):
                if not any(data == dev for _, data in ports):
                    ports.append((dev, dev))

        # Method 3: Windows fallback
        for i in range(1, 31):
            dev = f"COM{i}"
            if not any(data == dev for _, data in ports):
                ports.append((dev, dev))

        return ports
    
    
    def auto_select_known_ports(self):
        # HSI camera: IDS SDK placeholder for now
        if self.comboBox_connect_hsi_camera.count() > 1:
            self.comboBox_connect_hsi_camera.setCurrentIndex(1)

        # RGB camera: first /dev/video* camera
        for i in range(self.comboBox_connect_rgb_camera.count()):
            data = self.comboBox_connect_rgb_camera.itemData(i)
            if data and str(data).startswith("/dev/video"):
                self.comboBox_connect_rgb_camera.setCurrentIndex(i)
                break

        # Tracker: prefer ttyUSB
        for i in range(self.comboBox_connect_tracker.count()):
            data = str(self.comboBox_connect_tracker.itemData(i))
            text = self.comboBox_connect_tracker.itemText(i).lower()
            if "ttyusb" in data.lower() or "ttyusb" in text:
                self.comboBox_connect_tracker.setCurrentIndex(i)
                break

        # Filterwheel: prefer ttyACM
        for i in range(self.comboBox_connect_fw.count()):
            data = str(self.comboBox_connect_fw.itemData(i))
            text = self.comboBox_connect_fw.itemText(i).lower()
            if "ttyacm" in data.lower() or "ttyacm" in text:
                self.comboBox_connect_fw.setCurrentIndex(i)
                break


    def get_ids_camera_ports(self):
        ports = [("Select", None)]

        try:
            cam = IDSPeakCamera(pipeline_output=PixelFormat.MONO_12)
            infos = cam.list_cameras()

            for info in infos:
                label = f"IDS {info.model_name} | SN:{info.serial_number} | index:{info.index}"
                ports.append((label, {"index": info.index, "serial": info.serial_number}))

            cam.close_library()

        except Exception as e:
            ports.append((f"IDS not found: {e}", None))

        return ports


    def get_rgb_camera_ports(self):
        ports = [("Select", None)]

        # Linux/Jetson video devices
        for path in sorted(Path("/dev").glob("video*")):
            ports.append((str(path), str(path)))

        # Windows fallback
        if sys.platform.startswith("win"):
            for idx in range(6):
                ports.append((f"Camera index {idx}", idx))

        return ports

    def fill_combo(self, combo, items):
        combo.blockSignals(True)
        combo.clear()
        for label, data in items:
            combo.addItem(label, data)
        combo.blockSignals(False)
        self.set_combo_black(combo)

    # ========================================================
    # CONNECTION LOGIC
    # ========================================================

    def connect_all_devices(self):
        self.log("Connecting selected devices...")

        self.connected["hsi"] = self.connect_ids_hsi_camera()
        self.connected["rgb"] = self.connect_rgb_camera()
        motion_ok = self.connect_motion_system()
        self.connected["tracker"] = motion_ok
        self.connected["fw"] = motion_ok

        self.update_connection_combo_colors()

        any_connected = any([
            self.connected["hsi"],
            self.connected["rgb"],
            self.connected["tracker"],
            self.connected["fw"],
        ])

        if any_connected:
            self.set_status_connected()
            self.update_dashboard_connected_placeholders()
            self.log("At least one device connected successfully.")
            self.initialize_connected_devices_to_home()
        else:
            self.set_status_disconnected()
            self.log("No device connected. Check selected ports.")


    def initialize_connected_devices_to_home(self):
        self.log("Initializing PANCAM: home view position, default exposure, filterwheel...")

        # 1. Set exposure
        if self.connected.get("hsi", False):
            self.set_hsi_exposure_ms(DEFAULT_EXPOSURE_MS)

        # 2. Reset / initialize filterwheel if available
        motion = self.get_motion()
        if motion is not None:
            try:
                if hasattr(motion, "initialize_filterwheel"):
                    motion.initialize_filterwheel(start_pos=DEFAULT_FILTER_POS)
                    self.current_fw_pos = DEFAULT_FILTER_POS
                    self.log(f"Filterwheel initialized to position {DEFAULT_FILTER_POS}.")
                else:
                    self.set_filterwheel_position(DEFAULT_FILTER_POS)
            except Exception as e:
                self.log(f"Filterwheel initialization failed: {e}")

        # 3. Move to home view position: View Az=90, View Zn=90
        if self.connected.get("tracker", False):
            self.go_home_view_position()

        # 4. Sync GUI values
        with QSignalBlocker(self.doubleSpinBox_change_exp):
            self.doubleSpinBox_change_exp.setValue(DEFAULT_EXPOSURE_MS)

        with QSignalBlocker(self.spinBox_change_fw):
            self.spinBox_change_fw.setValue(DEFAULT_FILTER_POS)

        self.update_current_status_text()
        
    def connect_ids_hsi_camera(self):
        selected = self.comboBox_connect_hsi_camera.currentData()

        if selected is None:
            self.log("HSI IDS camera not selected.")
            return False

        try:
            width = int(self.of_config.get("SENSOR_WIDTH_PIXELS", 1440))
            height = int(self.of_config.get("SENSOR_HEIGHT_PIXELS", 1080))
            pixel_format = self.of_config.get("PIXEL_FORMAT", "Mono12")

            exposure_ms = float(self.of_config.get("EXPOSURE_START_MS", 500.0))
            gain = float(self.of_config.get("GAIN_START", 1.0))

            cam = IDSPeakCamera(pipeline_output=PixelFormat.MONO_12)

            serial = selected.get("serial")
            index = int(selected.get("index", 0))

            if serial:
                cam.open(serial=serial)
            else:
                cam.open(index=index)

            cam.node_map.FindNode("Width").SetValue(width)
            cam.node_map.FindNode("Height").SetValue(height)

            try:
                cam.node_map.FindNode("OffsetX").SetValue(0)
                cam.node_map.FindNode("OffsetY").SetValue(0)
            except Exception:
                pass

            cam.set_pixel_format_entry(pixel_format)
            cam.set_pipeline_output_format(PixelFormat.MONO_12)

            cam.set_exposure_us(exposure_ms * 1000.0)
            self.current_exp_ms = exposure_ms

            self.hsi_gain = cam.set_gain(gain, selector="AnalogAll", clamp=True)

            cam.set_trigger_off()
            cam.start()

            self.devices["hsi"] = cam
            self.log(f"HSI IDS camera connected: index={index}, serial={serial}")

            return True

        except Exception as e:
            self.log(f"HSI IDS camera connection failed: {e}")
            return False
        
    def connect_rgb_camera(self):
        selected = self.comboBox_connect_rgb_camera.currentData()

        if selected is None:
            self.log("RGB camera not selected.")
            return False

        try:
            cam = ColorCamera(
                src=selected,
                width=640,
                height=480,
                retry_sec=5.0,
            )

            if not cam.is_ok():
                self.log(f"RGB camera failed to open: {selected}")
                return False

            self.devices["rgb"] = cam
            self.log(f"RGB camera connected: {selected}")
            return True

        except Exception as e:
            self.log(f"RGB camera connection failed: {e}")
            return False


    def disconnect_all_devices(self):
        self.log("Disconnecting all devices...")

        # Stop live update first
        self.camera_timer.stop()

        # HSI
        try:
            dev = self.devices.get("hsi")
            if dev is not None:
                dev.stop()
                dev.close()
                dev.close_library()
                self.log("HSI disconnected.")
        except Exception as e:
            self.log(f"HSI disconnect error: {e}")

        # RGB
        try:
            dev = self.devices.get("rgb")
            if dev is not None:
                dev.release()
                self.log("RGB disconnected.")
        except Exception as e:
            self.log(f"RGB disconnect error: {e}")

        # Motion system: tracker + filterwheel
        try:
            dev = self.devices.get("motion")
            if dev is not None:
                dev.close()
                self.log("MOTION/TRACKER/FILTERWHEEL disconnected.")
        except Exception as e:
            self.log(f"Motion disconnect error: {e}")

        # Clear devices
        for key in self.devices:
            self.devices[key] = None

        # Clear connection state
        for key in self.connected:
            self.connected[key] = False

        self.update_connection_combo_colors()
        self.set_status_disconnected()
        self._setup_initial_dashboard_state()

        # Restart timer for future reconnection
        self._hsi_busy = False
        self.camera_timer.start(LIVE_CAMERA_TIMER_MS)

    def reset_all_devices_placeholder(self):
        self.log("Restart requested. Placeholder only. Relay power reset will be added later.")

    def update_connection_combo_colors(self):
        mapping = {
            "hsi": self.comboBox_connect_hsi_camera,
            "rgb": self.comboBox_connect_rgb_camera,
            "tracker": self.comboBox_connect_tracker,
            "fw": self.comboBox_connect_fw,
        }

        for key, combo in mapping.items():
            if self.connected.get(key, False):
                self.set_combo_green(combo)
            else:
                self.set_combo_black(combo)

    # ========================================================
    # OPERATION FILE
    # ========================================================

    def get_of_float(self, key, default):
        try:
            return float(self.of_config.get(key, default))
        except Exception:
            return float(default)

    def browse_operation_file(self):
        self.pause_live_view()

        try:
            dialog = QFileDialog(self)
            dialog.setWindowTitle("Select PANCAM Operation File")
            dialog.setDirectory(str(self.operation_dir))
            dialog.setNameFilter("Operation Files (*.txt);;All Files (*)")
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)

            if dialog.exec():
                selected = dialog.selectedFiles()
                if selected:
                    file_path = selected[0]
                    self.lineEdit_load_of.setText(file_path)
                    self.set_lineedit_black(self.lineEdit_load_of)
                    self.log(f"Selected OF: {file_path}")

        finally:
            self.resume_live_view()

    def browse_schedule_folder(self):
        self.pause_live_view()

        try:
            self.schedule_dir.mkdir(exist_ok=True)

            dialog = QFileDialog(self)
            dialog.setWindowTitle("Select Schedule Folder")
            dialog.setDirectory(str(self.schedule_dir))
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setOption(QFileDialog.ShowDirsOnly, True)
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)

            if dialog.exec():
                selected = dialog.selectedFiles()
                if selected:
                    folder = selected[0]
                    self.lineEdit_schedule.setText(folder)
                    self.log(f"Schedule folder selected: {folder}")

        finally:
            self.resume_live_view()

    def load_operation_file(self):
        path = Path(self.lineEdit_load_of.text().strip())

        if not path.is_file():
            self.log(f"Operation file not found: {path}")
            self.set_lineedit_black(self.lineEdit_load_of)
            return

        try:
            self.of_config = self.parse_key_value_file(path)
            self.fill_header_from_of()
            # self.update_header_dynamic_values()
            self.update_header_date_time_only()
            self.update_weather()
            self.set_lineedit_green(self.lineEdit_load_of)
            self.log(f"Operation file loaded: {path.name}")

        except Exception as e:
            self.log(f"Failed to load operation file: {e}")
            self.set_lineedit_black(self.lineEdit_load_of)

    def parse_key_value_file(self, path):
        config = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if "#" in line:
                    line = line.split("#", 1)[0].strip()

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                config[key] = value

        return config

    def fill_header_from_of(self):
        cfg = self.of_config

        name = cfg.get("INSTRUMENT_NAME", cfg.get("INSTRUMENT_TYPE", "PANCAM"))
        site = cfg.get("SITE_NAME", cfg.get("LOCATION", "Unknown"))
        lat = cfg.get("LATITUDE_DEG", "")
        lon = cfg.get("LONGITUDE_DEG", "")

        self.h_name_fill.setText(str(name))
        self.h_location_fill.setText(str(site))
        self.h_latitude_fill.setText(str(lat))
        self.h_longtitude_fill.setText(str(lon))

    def pause_live_view(self):
        if hasattr(self, "camera_timer") and self.camera_timer.isActive():
            self.camera_timer.stop()
            self._live_view_was_running = True
        else:
            self._live_view_was_running = False


    def resume_live_view(self):
        if getattr(self, "_live_view_was_running", False):
            self.camera_timer.start(LIVE_CAMERA_TIMER_MS)
            self._live_view_was_running = False

    # ========================================================
    # CALIBRATION FILE
    # ========================================================

    def browse_calibration_file_or_folder(self):
        self.pause_live_view()

        try:
            dialog = QFileDialog(self)
            dialog.setWindowTitle("Select PANCAM Calibration Folder")
            dialog.setDirectory(str(self.calibration_dir))
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setOption(QFileDialog.ShowDirsOnly, True)
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)

            if dialog.exec():
                selected = dialog.selectedFiles()
                if selected:
                    folder = selected[0]
                    self.lineEdit_load_cf.setText(folder)
                    self.set_lineedit_black(self.lineEdit_load_cf)
                    self.log(f"Selected CF folder: {folder}")

        finally:
            self.resume_live_view()

    def load_calibration_files(self):
        path = Path(self.lineEdit_load_cf.text().strip())

        if not path.exists():
            self.log(f"Calibration path not found: {path}")
            self.set_lineedit_black(self.lineEdit_load_cf)
            return

        bad_pixel = path / "bad_pixel_mask.npy"
        smile = path / "smile_shift_matrix.npy"

        found_files = []
        if bad_pixel.exists():
            found_files.append("bad_pixel_mask.npy")
        if smile.exists():
            found_files.append("smile_shift_matrix.npy")

        if found_files:
            self.cf_loaded = True
            self.set_lineedit_green(self.lineEdit_load_cf)
            self.log("Calibration files loaded: " + ", ".join(found_files))
        else:
            self.cf_loaded = False
            self.set_lineedit_black(self.lineEdit_load_cf)
            self.log("No calibration files found yet.")

    # ========================================================
    # HEADER: DATE, TIME, SOLAR, WEATHER
    # ========================================================

    def update_header_date_time_only(self):
        tz_name = self.of_config.get("TIMEZONE", "America/New_York")

        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("America/New_York")

        now = datetime.now(tz)

        self.h_date_fill.setText(now.strftime("%Y-%m-%d"))
        self.h_time_fill.setText(now.strftime("%H:%M:%S"))

    def update_sky_target_header(self):
        try:
            solar_az, solar_zn = self.compute_solar_position_numeric()

            if solar_zn <= 90.0:
                self.current_sky_target = "sun"
                self.current_target_az_deg = solar_az
                self.current_target_zn_deg = solar_zn

                self.h_solar_az.setText("Solar Az:")
                self.h_solar_zn.setText("Solar Zn:")
                self.h_solar_az_fill.setText(f"{solar_az:.2f}°")
                self.h_solar_zn_fill.setText(f"{solar_zn:.2f}°")
                self.pushButton_sun_search.setText("Sun Search")

            else:
                lunar_az, lunar_zn = self.compute_lunar_position_numeric()

                self.current_sky_target = "moon"
                self.current_target_az_deg = lunar_az
                self.current_target_zn_deg = lunar_zn

                self.h_solar_az.setText("Lunar Az:")
                self.h_solar_zn.setText("Lunar Zn:")
                self.h_solar_az_fill.setText(f"{lunar_az:.2f}°")
                self.h_solar_zn_fill.setText(f"{lunar_zn:.2f}°")
                self.pushButton_sun_search.setText("Moon Search")

        except Exception as e:
            self.log(f"Sky target update failed: {e}")
            self.h_solar_az_fill.setText("N/A")
            self.h_solar_zn_fill.setText("N/A")

    def compute_solar_position(self, now=None):
        try:
            import pandas as pd
            import pvlib

            lat = float(self.of_config.get("LATITUDE_DEG"))
            lon = float(self.of_config.get("LONGITUDE_DEG"))
            alt = float(self.of_config.get("ALTITUDE_M", 0.0))
            tz_name = self.of_config.get("TIMEZONE", "America/New_York")

            now_local = pd.Timestamp.now(tz=tz_name)
            now_utc = now_local.tz_convert("UTC")

            solpos = pvlib.solarposition.get_solarposition(
                time=pd.DatetimeIndex([now_utc]),
                latitude=lat,
                longitude=lon,
                altitude=alt,
            )

            solar_zn = float(solpos["zenith"].iloc[0])
            solar_az = float(solpos["azimuth"].iloc[0])

            return f"{solar_az:.2f}°", f"{solar_zn:.2f}°"

        except Exception as e:
            self.log(f"pvlib solar position failed, using fallback: {e}")
            return self.compute_solar_position_fallback()

    def compute_solar_position_numeric(self):
        if pd is None or pvlib is None:
            az_txt, zn_txt = self.compute_solar_position_fallback()
            return float(az_txt.replace("°", "")), float(zn_txt.replace("°", ""))

        lat = float(self.of_config.get("LATITUDE_DEG"))
        lon = float(self.of_config.get("LONGITUDE_DEG"))
        alt = float(self.of_config.get("ALTITUDE_M", 0.0))
        tz_name = self.of_config.get("TIMEZONE", "America/New_York")

        now_local = pd.Timestamp.now(tz=tz_name)
        now_utc = now_local.tz_convert("UTC")

        solpos = pvlib.solarposition.get_solarposition(
            time=pd.DatetimeIndex([now_utc]),
            latitude=lat,
            longitude=lon,
            altitude=alt,
        )

        solar_az = float(solpos["azimuth"].iloc[0])
        solar_zn = float(solpos["zenith"].iloc[0])

        return solar_az, solar_zn

    def compute_solar_position_fallback(self):
        try:
            lat = float(self.of_config.get("LATITUDE_DEG"))
            lon = float(self.of_config.get("LONGITUDE_DEG"))
            tz_name = self.of_config.get("TIMEZONE", "America/New_York")

            local_now = datetime.now(ZoneInfo(tz_name))
            utc_now = local_now.astimezone(ZoneInfo("UTC"))

            day_of_year = utc_now.timetuple().tm_yday
            hour = utc_now.hour + utc_now.minute / 60 + utc_now.second / 3600

            gamma = 2.0 * math.pi / 365.0 * (
                day_of_year - 1 + (hour - 12) / 24
            )

            decl = (
                0.006918
                - 0.399912 * math.cos(gamma)
                + 0.070257 * math.sin(gamma)
                - 0.006758 * math.cos(2 * gamma)
                + 0.000907 * math.sin(2 * gamma)
                - 0.002697 * math.cos(3 * gamma)
                + 0.00148 * math.sin(3 * gamma)
            )

            eq_time = 229.18 * (
                0.000075
                + 0.001868 * math.cos(gamma)
                - 0.032077 * math.sin(gamma)
                - 0.014615 * math.cos(2 * gamma)
                - 0.040849 * math.sin(2 * gamma)
            )

            local_minutes = local_now.hour * 60 + local_now.minute + local_now.second / 60
            timezone_offset = local_now.utcoffset().total_seconds() / 3600

            true_solar_time = (
                local_minutes
                + eq_time
                + 4 * lon
                - 60 * timezone_offset
            ) % 1440

            hour_angle = true_solar_time / 4 - 180

            ha_rad = math.radians(hour_angle)
            lat_rad = math.radians(lat)

            cos_zenith = (
                math.sin(lat_rad) * math.sin(decl)
                + math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad)
            )
            cos_zenith = max(-1.0, min(1.0, cos_zenith))

            solar_zn = math.degrees(math.acos(cos_zenith))

            az_rad = math.atan2(
                math.sin(ha_rad),
                math.cos(ha_rad) * math.sin(lat_rad)
                - math.tan(decl) * math.cos(lat_rad),
            )

            solar_az = (math.degrees(az_rad) + 180) % 360

            return f"{solar_az:.2f}°", f"{solar_zn:.2f}°"

        except Exception as e:
            self.log(f"Fallback solar position failed: {e}")
            return "N/A", "N/A"

    def ensure_skyfield_loaded(self):
        if self.skyfield_ts is None or self.skyfield_eph is None:
            from skyfield.api import load

            self.skyfield_ts = load.timescale()
            self.skyfield_eph = load("de421.bsp")


    def compute_lunar_position_numeric(self):
        from skyfield.api import wgs84

        self.ensure_skyfield_loaded()

        lat = float(self.of_config.get("LATITUDE_DEG"))
        lon = float(self.of_config.get("LONGITUDE_DEG"))
        alt_m = float(self.of_config.get("ALTITUDE_M", 0.0))

        t = self.skyfield_ts.now()

        observer = wgs84.latlon(
            latitude_degrees=lat,
            longitude_degrees=lon,
            elevation_m=alt_m,
        )

        astrometric = (
            self.skyfield_eph["earth"] + observer
        ).at(t).observe(
            self.skyfield_eph["moon"]
        )

        alt, az, distance = astrometric.apparent().altaz()

        lunar_az = float(az.degrees)
        lunar_zn = 90.0 - float(alt.degrees)

        return lunar_az, lunar_zn

    def update_weather(self):
        if not self.of_config:
            return

        if requests is None:
            self.h_temperature_fill.setText("N/A")
            self.h_humidity_fill.setText("N/A")
            self.h_pressure_fill.setText("N/A")
            self.h_type_fill.setText("N/A")
            return

        try:
            lat = float(self.of_config.get("LATITUDE_DEG"))
            lon = float(self.of_config.get("LONGITUDE_DEG"))

            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,relative_humidity_2m,surface_pressure,weather_code"
                "&timezone=auto"
            )

            r = requests.get(url, timeout=3)
            r.raise_for_status()
            data = r.json()["current"]

            temp = data.get("temperature_2m")
            hum = data.get("relative_humidity_2m")
            pressure = data.get("surface_pressure")
            code = data.get("weather_code")

            self.h_temperature_fill.setText(f"{temp:.1f} °C" if temp is not None else "N/A")
            self.h_humidity_fill.setText(f"{hum:.0f} %" if hum is not None else "N/A")
            self.h_pressure_fill.setText(f"{pressure:.1f} hPa" if pressure is not None else "N/A")
            self.h_type_fill.setText(WEATHER_CODE_MAP.get(code, f"Code {code}"))

            self.log("Weather updated.")

        except Exception as e:
            self.h_temperature_fill.setText("N/A")
            self.h_humidity_fill.setText("N/A")
            self.h_pressure_fill.setText("N/A")
            self.h_type_fill.setText("N/A")
            self.log(f"Weather update failed: {e}")

    # ========================================================
    # DASHBOARD STATUS
    # ========================================================

    def set_status_disconnected(self):
        self.label_current_status.setStyleSheet(STATUS_GRAY)
        self.label_current_status.setText(
            "View Az:  | View Zn:  | Exp time:  | FW:  | Sat:  | Wv shift: "
        )

    def set_status_connected(self):
        self.label_current_status.setStyleSheet(STATUS_GREEN)
        self.update_current_status_text()

    def update_camera_temperature(self):
        cam = self.devices.get("hsi", None)

        if cam is None:
            self.current_camera_temp_c = float("nan")
            return

        try:
            self.current_camera_temp_c = float(cam.get_temperature_c())
        except Exception:
            self.current_camera_temp_c = float("nan")

    def update_current_status_text(self):
        if np.isnan(self.current_camera_temp_c):
            temp_txt = "N/A"
        else:
            temp_txt = f"{self.current_camera_temp_c:.1f}°C"

        self.label_current_status.setText(
            f"View Az: {self.current_view_az_deg:.1f}°  | "
            f"View Zn: {self.current_view_zn_deg:.1f}°  | "
            f"Exp time: {self.current_exp_ms:.2f} ms  | "
            f"FW: {self.current_fw_pos}  | "
            f"Temp: {temp_txt}  | "
            f"Sat: {self.current_sat}  | "
            f"Wv shift: {self.current_wv_shift}"
            )

    def update_dashboard_connected_placeholders(self):
        if self.connected["hsi"]:
            self.label_hsi_camera.setText("HSI camera connected")
            self.label_mean_spectrum.setText("Mean spectrum display ready")

        if self.connected["rgb"]:
            self.label_rgb_camera.setText("RGB camera connected")

        self.label_reconstructed_image.setText("Reconstructed image display ready")
        self.label_no2_profile.setText("NO₂ profile retrieval display placeholder")

