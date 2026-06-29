from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import numpy as np

MOLEC_CM2_PER_DU = 2.687e16

WL_MIN = 435.0
WL_MAX = 490.0
POLY_ORDER = 4

USE_NO2 = True
USE_O3 = True
USE_O4 = True
USE_H2O = True
USE_CHOCHO = False
RING_MODE = "solar"

FIT_WAVELENGTH_CORRECTION = True
USE_SHIFT = True
SHIFT_MIN_NM = -0.50
SHIFT_MAX_NM = 0.50
SHIFT_STEP_NM = 0.01


def _get_backend(use_gpu=True):
    if not use_gpu:
        return np, False

    try:
        import cupy as cp
        ok = cp.cuda.runtime.getDeviceCount() > 0
        return (cp if ok else np), ok
    except Exception:
        return np, False


def _parse_utc(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _time_diff_seconds(a, b):
    da = _parse_utc(a)
    db = _parse_utc(b)

    if da is None or db is None:
        return 1e99

    return abs((da - db).total_seconds())


def _load_first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return np.loadtxt(
                p,
                comments=(";", "#", "*"),
                usecols=(0, 1),
            )

    raise FileNotFoundError("Missing cross-section file:\n" + "\n".join(str(p) for p in paths))


def load_cross_sections(xs_dir):
    xs_dir = Path(xs_dir)
    xs = {}

    xs["NO2"] = np.loadtxt(
        xs_dir / "NO2_220K_PANCAM_430_460nm_convolved.xs",
        comments=(";", "#", "*"),
        usecols=(0, 1),
    )

    xs["O3"] = np.loadtxt(
        xs_dir / "O3_223K_PANCAM_430_460nm_convolved.xs",
        comments=(";", "#", "*"),
        usecols=(0, 1),
    )

    xs["O4"] = np.loadtxt(
        xs_dir / "O4_293K_PANCAM_430_460nm_convolved.xs",
        comments=(";", "#", "*"),
        usecols=(0, 1),
    )

    xs["H2O"] = np.loadtxt(
        xs_dir / "H2O_PANCAM_430_460nm_convolved.xs",
        comments=(";", "#", "*"),
        usecols=(0, 1),
    )

    if USE_CHOCHO:
        xs["CHOCHO"] = _load_first_existing([
            xs_dir / "CHOCHO_296K_PANCAM_430_460nm_convolved.xs",
            xs_dir / "CHOCHO_296K_PANCAM_convolved_430_460nm.txt",
        ])

    ring_solar = _load_first_existing([
        xs_dir / "Ring_solar_PANCAM_430_460nm_convolved.xs",
        xs_dir / "Ring_solar_PANCAM_convolved_430_460nm.txt",
    ])

    ring_raman = _load_first_existing([
        xs_dir / "Ring_raman_PANCAM_430_460nm_convolved.xs",
        xs_dir / "Ring_raman_PANCAM_convolved_430_460nm.txt",
    ])

    if RING_MODE.lower() == "solar":
        xs["Ring"] = ring_solar
    elif RING_MODE.lower() == "raman":
        xs["Ring"] = ring_raman
    else:
        common_wl = ring_solar[:, 0]
        raman_interp = np.interp(common_wl, ring_raman[:, 0], ring_raman[:, 1])
        xs["Ring"] = np.column_stack([common_wl, raman_interp / np.maximum(ring_solar[:, 1], 1e-30)])

    return xs


def remove_polynomial(y, wl, order=4):
    x = (wl - np.mean(wl)) / (wl.max() - wl.min())
    P = np.column_stack([x**p for p in range(order + 1)])
    valid = np.isfinite(y)
    coef, *_ = np.linalg.lstsq(P[valid], y[valid], rcond=None)
    return y - P @ coef
