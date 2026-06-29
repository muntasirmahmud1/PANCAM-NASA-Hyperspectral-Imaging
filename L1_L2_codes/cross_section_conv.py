from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from L0_to_L1_functions import read_general_cf

# =========================================================
# PATHS
# =========================================================
XS_DIR = Path("Cross sections")
CAL_DIR = Path("Calibration File")
GENERAL_CF_FILE = CAL_DIR / "general_calibration_file.txt"

OUT_DIR_FULL = XS_DIR / "PANCAM_convolved_full_grid"
OUT_DIR_430_460 = XS_DIR / "PANCAM_convolved_435_490nm"

OUT_DIR_FULL.mkdir(parents=True, exist_ok=True)
OUT_DIR_430_460.mkdir(parents=True, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
WL_MIN = 435.0
WL_MAX = 490.0

N_SPEC = 1440

# =========================================================
# LOAD PANCAM WAVELENGTH AND RESOLUTION CALIBRATION
# =========================================================
cf = read_general_cf(GENERAL_CF_FILE)

dispersion_poly = np.array(
    [float(x) for x in cf["DISPERSION_POLY"].split()],
    dtype=float
)

resolution_poly = np.array(
    [float(x) for x in cf["RESOLUTION_POLY"].split()],
    dtype=float
)

pix = np.arange(N_SPEC)
wavelength = np.polyval(dispersion_poly, pix)

fit_mask = (wavelength >= WL_MIN) & (wavelength <= WL_MAX)
wavelength_430_460 = wavelength[fit_mask]

print("Dispersion polynomial:", dispersion_poly)
print("Resolution polynomial:", resolution_poly)
print("Full wavelength range:", wavelength.min(), "to", wavelength.max(), "nm")
print("430–460 grid points:", wavelength_430_460.size)

# =========================================================
# INPUT CROSS-SECTIONS
# =========================================================
raw_xs_files = {
    "NO2_220K": XS_DIR / "no2_VANDAELE_1998_220K.xs",
    "O3_223K": XS_DIR / "O3223_Serdyuchenko(2014)_223K_213-1100nm(2013 version).xs",
    "O4_293K": XS_DIR / "O4_ThalmanVolkamer(2013)_293K_335.749-600.802nm.txt",
    "H2O": XS_DIR / "h2o_polyanski_20151202_vac.xs",
    "CHOCHO_296K": XS_DIR / "CHOCHO_Volkamer(2005)_296K_250.031-526.168nm(convol.0.01nm).txt",
    "Ring_solar": XS_DIR / "ring_sao2010_hr_norm_solar.xs",
    "Ring_raman": XS_DIR / "ringraman_sao2010_hr_norm_raman.xs",
}

# =========================================================
# FUNCTIONS
# =========================================================
def pancam_fwhm_nm(wl_nm):
    """
    PANCAM spectral FWHM in nm.
    The resolution polynomial is evaluated using wavelength in microns.
    """
    wl_um = wl_nm / 1000.0
    return np.polyval(resolution_poly, wl_um)


def load_two_column_xs(path):
    arr = np.loadtxt(
        path,
        comments=("*", ";", "#"),
        usecols=(0, 1)
    )

    wl = arr[:, 0].astype(float)
    val = arr[:, 1].astype(float)

    good = np.isfinite(wl) & np.isfinite(val)
    wl = wl[good]
    val = val[good]

    order = np.argsort(wl)

    return wl[order], val[order]


def gaussian_convolve_to_target_grid(wl_hr, xs_hr, wl_target):
    """
    Convolve high-resolution cross-section to PANCAM target grid
    using wavelength-dependent Gaussian FWHM.
    """
    xs_conv = np.full_like(wl_target, np.nan, dtype=float)

    for i, wl0 in enumerate(wl_target):
        fwhm = pancam_fwhm_nm(wl0)
        sigma = fwhm / 2.354820045

        use = np.abs(wl_hr - wl0) <= 5.0 * sigma

        if np.sum(use) < 3:
            continue

        weights = np.exp(
            -0.5 * ((wl_hr[use] - wl0) / sigma) ** 2
        )

        weights = weights / np.sum(weights)

        xs_conv[i] = np.sum(xs_hr[use] * weights)

    return xs_conv


def save_two_column_file(path, wl, val, source_path, grid_note):
    header = (
        "\n"
        "PANCAM convolved cross section file\n"
        f"High resolution file : {source_path}\n"
        f"Grid : {grid_note}\n"
        "Shift applied : 0 nm\n"
        "Convolution type : Standard convolution\n"
        "Slit function type : Gaussian\n"
        f"FWHM polynomial : {' '.join([f'{v:.8e}' for v in resolution_poly])}\n"
        "Column 1 : PANCAM calibrated wavelength [nm]\n"
        "Column 2 : convolved cross section / spectrum\n"
    )

    np.savetxt(
        path,
        np.column_stack([wl, val]),
        fmt="%.14e %.14e",
        header=header,
        comments=";"
    )


def make_ring_ratio(raman_conv, solar_conv):
    return raman_conv / np.maximum(solar_conv, 1e-30)


# =========================================================
# CONVOLVE AND SAVE
# =========================================================
prepared_full = {}
prepared_430_460 = {}

for name, path in raw_xs_files.items():

    print("Processing:", name)

    wl_hr, xs_hr = load_two_column_xs(path)

    margin = 5.0

    # -----------------------------
    # Full PANCAM grid convolution
    # -----------------------------
    use_full = (
        (wl_hr >= wavelength.min() - margin)
        &
        (wl_hr <= wavelength.max() + margin)
    )

    xs_conv_full = gaussian_convolve_to_target_grid(
        wl_hr[use_full],
        xs_hr[use_full],
        wavelength
    )

    prepared_full[name] = xs_conv_full

    out_full = OUT_DIR_FULL / f"{name}_PANCAM_full_grid_convolved.xs"

    save_two_column_file(
        out_full,
        wavelength,
        xs_conv_full,
        path,
        grid_note="Full PANCAM wavelength grid"
    )

    print("Saved full:", out_full.name)

    # -----------------------------
    # 430–460 nm convolution
    # -----------------------------
    use_fit = (
        (wl_hr >= WL_MIN - margin)
        &
        (wl_hr <= WL_MAX + margin)
    )

    xs_conv_430_460 = gaussian_convolve_to_target_grid(
        wl_hr[use_fit],
        xs_hr[use_fit],
        wavelength_430_460
    )

    prepared_430_460[name] = xs_conv_430_460

    out_430_460 = OUT_DIR_430_460 / f"{name}_PANCAM_430_460nm_convolved.xs"

    save_two_column_file(
        out_430_460,
        wavelength_430_460,
        xs_conv_430_460,
        path,
        grid_note="PANCAM wavelength grid restricted to 430–460 nm"
    )

    print("Saved 430–460:", out_430_460.name)


# =========================================================
# RING RATIO FILES
# =========================================================
ring_ratio_full = make_ring_ratio(
    prepared_full["Ring_raman"],
    prepared_full["Ring_solar"]
)

ring_ratio_430_460 = make_ring_ratio(
    prepared_430_460["Ring_raman"],
    prepared_430_460["Ring_solar"]
)

ring_full_file = OUT_DIR_FULL / "Ring_ratio_Raman_over_Solar_PANCAM_full_grid_convolved.xs"
ring_fit_file = OUT_DIR_430_460 / "Ring_ratio_Raman_over_Solar_PANCAM_430_460nm_convolved.xs"

save_two_column_file(
    ring_full_file,
    wavelength,
    ring_ratio_full,
    "ringraman_sao2010_hr_norm_raman.xs / ring_sao2010_hr_norm_solar.xs",
    grid_note="Full PANCAM wavelength grid"
)

save_two_column_file(
    ring_fit_file,
    wavelength_430_460,
    ring_ratio_430_460,
    "ringraman_sao2010_hr_norm_raman.xs / ring_sao2010_hr_norm_solar.xs",
    grid_note="PANCAM wavelength grid restricted to 430–460 nm"
)

print("Saved:", ring_full_file.name)
print("Saved:", ring_fit_file.name)

# =========================================================
# QUICK PLOTS
# =========================================================

# Plot normalized differential shapes for 430–460 nm files
plt.figure(figsize=(14, 6))

for name in ["NO2_220K", "O3_223K", "O4_293K", "H2O", "CHOCHO_296K"]:

    y = prepared_430_460[name].copy()
    y = y - np.nanmean(y)

    max_abs = np.nanmax(np.abs(y))

    if max_abs > 0:
        y = y / max_abs

    plt.plot(wavelength_430_460, y, label=name, linewidth=2)

plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized differential shape")
plt.title("PANCAM-convolved cross sections: 430–460 nm")
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()


# Plot NO2 actual convolved cross section
plt.figure(figsize=(12, 4))

plt.plot(
    wavelength_430_460,
    prepared_430_460["NO2_220K"],
    "o-",
    linewidth=2
)

plt.xlabel("Wavelength [nm]")
plt.ylabel("NO2 cross section")
plt.title("PANCAM-convolved NO2 cross section: 430–460 nm")
plt.grid(True)
plt.tight_layout()


# Plot Ring ratio
plt.figure(figsize=(14, 4))

plt.plot(
    wavelength_430_460,
    ring_ratio_430_460,
    linewidth=2
)

plt.xlabel("Wavelength [nm]")
plt.ylabel("Ring ratio")
plt.title("PANCAM Ring = Raman / Solar: 430–460 nm")
plt.grid(True)
plt.tight_layout()


# Plot PANCAM FWHM
plt.figure(figsize=(10, 4))

plt.plot(
    wavelength_430_460,
    pancam_fwhm_nm(wavelength_430_460),
    linewidth=2
)

plt.xlabel("Wavelength [nm]")
plt.ylabel("FWHM [nm]")
plt.title("PANCAM FWHM used for convolution")
plt.grid(True)
plt.tight_layout()

plt.show()

print("Done.")
print("Full-grid files saved to:", OUT_DIR_FULL)
print("430–460 nm files saved to:", OUT_DIR_430_460)
