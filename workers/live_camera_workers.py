from __future__ import annotations

import time
import numpy as np

from PySide6.QtCore import QObject, Signal, Slot

from pancam_scan_common_v2 import crop_y


class LiveRGBWorker(QObject):
    frame_ready = Signal(object)
    log = Signal(str)
    finished = Signal()

    def __init__(self, rgb_camera, interval_ms=100):
        super().__init__()
        self.rgb_camera = rgb_camera
        self.interval_ms = int(interval_ms)
        self.running = False

    @Slot()
    def run(self):
        self.running = True

        while self.running:
            try:
                frame = self.rgb_camera.read()
                if frame is not None:
                    self.frame_ready.emit(frame)
            except Exception as e:
                self.log.emit(f"RGB worker error: {e}")

            time.sleep(self.interval_ms / 1000.0)

        self.finished.emit()

    @Slot()
    def stop(self):
        self.running = False


class LiveHSIWorker(QObject):
    frame_ready = Signal(object)
    spectrum_ready = Signal(object)
    temperature_ready = Signal(float)
    log = Signal(str)
    finished = Signal()

    def __init__(
        self,
        hsi_camera,
        crop_y0=290,
        crop_y1=990,
        interval_ms=500,
        spectrum_every_n=2,
    ):
        super().__init__()

        self.hsi_camera = hsi_camera
        self.crop_y0 = int(crop_y0)
        self.crop_y1 = int(crop_y1)
        self.interval_ms = int(interval_ms)
        self.spectrum_every_n = max(1, int(spectrum_every_n))

        self.running = False
        self.counter = 0

    @Slot()
    def run(self):
        self.running = True

        while self.running:
            try:
                frame = self.hsi_camera.grab(
                    timeout_ms=4000,
                    do_software_trigger=False,
                )

                frame = crop_y(frame, self.crop_y0, self.crop_y1)

                self.frame_ready.emit(frame)

                self.counter += 1
                if self.counter % self.spectrum_every_n == 0:
                    mean_spec = frame.mean(axis=0, dtype=np.float32)
                    self.spectrum_ready.emit(mean_spec)

                try:
                    temp_c = float(self.hsi_camera.get_temperature_c())
                    self.temperature_ready.emit(temp_c)
                except Exception:
                    pass

            except Exception as e:
                msg = str(e)
                if "TIMEOUT" not in msg and "timed out" not in msg:
                    self.log.emit(f"HSI worker error: {e}")

            time.sleep(self.interval_ms / 1000.0)

        self.finished.emit()

    @Slot()
    def stop(self):
        self.running = False
