# # compare_l1_vs_reference.py

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

# =========================================================
# INPUTS
# =========================================================

QDOAS_DIR = Path("data/20260603/manual_az_scan_20260603_084109_638/L1_QDOAS")

BIN_TO_PLOT = 2
SPECTRUM_INDEX = 300

# =========================================================
# FILES
# =========================================================

spe_file = QDOAS_DIR / f"PANCAM_20260603_bin{BIN_TO_PLOT:02d}.spe"
ref_file = QDOAS_DIR / f"PANCAM_20260603_bin{BIN_TO_PLOT:02d}.spe_ref"
clb_file = QDOAS_DIR / "PANCAM_20260603.clb"

# =========================================================
# LOAD
# =========================================================

wavelength = np.loadtxt(clb_file)

spe_data = np.loadtxt(spe_file, dtype=str)
ref_data = np.loadtxt(ref_file)

print("spe_data shape:", spe_data.shape)
print("ref_data shape:", ref_data.shape)

# =========================================================
# SELECT ONE SPECTRUM
# =========================================================

row = spe_data[SPECTRUM_INDEX]

scan_spec = row[5:].astype(float)

# reference file:
# col0 = wavelength
# col1 = intensity

ref_spec = ref_data[:, 1]


scan_spec = savgol_filter(scan_spec, 31, 2)
ref_spec = savgol_filter(ref_spec, 31, 2)
# =========================================================
# NORMALIZE
# =========================================================

scan_norm = scan_spec / np.nanmax(scan_spec)
ref_norm = ref_spec / np.nanmax(ref_spec)

# =========================================================
# RATIO
# =========================================================

ratio = scan_spec / np.maximum(ref_spec, 1e-30)

dod = -np.log(np.maximum( ratio, 1e-30))

# =========================================================
# PLOT
# =========================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 14),sharex=True)

# ---------------------------------------------------------
# scan and reference
# ---------------------------------------------------------

axes[0].plot(wavelength, scan_norm, label="Scan")

axes[0].plot(wavelength, ref_norm,label="Reference")
axes[0].set_title(f"Bin {BIN_TO_PLOT}, spectrum {SPECTRUM_INDEX}")

axes[0].set_ylabel("Normalized")
axes[0].legend()
axes[0].grid(True)

# ---------------------------------------------------------
# difference
# ---------------------------------------------------------

axes[1].plot(wavelength, scan_norm - ref_norm)
axes[1].set_ylabel("Scan - Ref")
axes[1].grid(True)

# ---------------------------------------------------------
# ratio
# ---------------------------------------------------------
axes[2].plot(wavelength, ratio)
axes[2].set_ylabel("Scan / Ref")
axes[2].grid(True)

# ---------------------------------------------------------
# differential optical depth
# ---------------------------------------------------------

axes[3].plot(wavelength, dod)
axes[3].set_ylabel("-ln(Scan/Ref)")
axes[3].set_xlabel("Wavelength [nm]")
axes[3].grid(True)
plt.tight_layout()




file_path = "data/20260603/Pandora2s1_GreenbeltMD_20260603_L1_smca1c20d20240823p1-8 - Copy.txt"
base_name = os.path.basename(file_path)

# Read the entire file
with open(file_path, 'r', encoding='latin1') as file:
    lines = file.readlines()

# Extract the line containing the nominal wavelengths
nominal_wavelengths_line = lines[22].strip()  # Line 23 in the file (index 22)

# Split the line into individual wavelengths and convert them to float
wavelength_p2 = [float(value) for value in nominal_wavelengths_line.split(': ')[1].split()]

# Skip the initial 89 rows
data_lines = lines[89:]

# Process each line to split into rows and columns
data_list = []
for line in data_lines:
    # Strip the newline character and split by tabs
    rows = line.strip().split('\t')
    for row in rows:
        # Split each row by spaces
        columns = row.split()
        data_list.append(columns)

# Convert the list of lists into a DataFrame
data1 = pd.DataFrame(data_list)
# Convert the necessary columns to numeric before saving
data1.iloc[:, 2:6205] = data1.iloc[:, 2:6205].apply(pd.to_numeric, errors='coerce')

p2_reference = data1.iloc[36, 61:2109].astype(float).values
p2_spectrum  = data1.iloc[39, 61:2109].astype(float).values
wavelength_p2 = np.asarray(wavelength_p2, dtype=float)

valid = (
    np.isfinite(wavelength_p2)
    & np.isfinite(p2_reference)
    & np.isfinite(p2_spectrum)
    & (p2_reference > 0)
    & (p2_spectrum > 0)
)

wl = wavelength_p2[valid]
ref = p2_reference[valid]
scan = p2_spectrum[valid]

ref_norm = ref / np.nanmax(ref)
scan_norm = scan / np.nanmax(scan)

ratio = scan / ref
dod = -np.log(ratio)

print("valid points:", valid.sum(), "out of", len(valid))
print("ratio min:", np.nanmin(ratio))
print("ratio max:", np.nanmax(ratio))
print("ratio median:", np.nanmedian(ratio))
print("dod min:", np.nanmin(dod))
print("dod max:", np.nanmax(dod))

fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

axes[0].plot(wl, scan_norm, label="Pandora scan", linewidth=2)
axes[0].plot(wl, ref_norm, label="Pandora reference", linewidth=2)
axes[0].set_ylabel("Normalized")
axes[0].set_title("Pandora reference vs scan")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(wl, scan_norm - ref_norm, linewidth=2)
axes[1].set_ylabel("Scan - Ref")
axes[1].grid(True)

axes[2].plot(wl, ratio, linewidth=2)
axes[2].set_ylabel("Scan / Ref")
axes[2].grid(True)

axes[3].plot(wl, dod, linewidth=2)
axes[3].set_ylabel("-ln(Scan/Ref)")
axes[3].set_xlabel("Wavelength [nm]")
axes[3].grid(True)

plt.xlim(390, 525)
plt.tight_layout()
plt.show()