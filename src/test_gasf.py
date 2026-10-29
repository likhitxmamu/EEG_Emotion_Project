# ============================================================
# EEG EMOTION PROJECT
# GASF RGB TEST - VERSION 2
#
# PURPOSE:
# Generate ONE clean RGB GASF image for visual testing.
#
# DO NOT GENERATE THE FULL DATASET YET.
# DO NOT TRAIN THE CNN YET.
# ============================================================

import os
import numpy as np
from scipy.io import loadmat
from scipy import signal
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data/output/fear/1_20160518_trial2.mat"
)

OUTPUT_DIR = "data/processed/test_gasf"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# EEG channel to test
CHANNEL_NUMBER = 1

# Final CNN image size
IMAGE_SIZE = 224

# DEAP sampling frequency
FS = 128

# EEG frequency range
LOWCUT = 1.0
HIGHCUT = 40.0

# Power-line interference
NOTCH_FREQ = 50.0

# ------------------------------------------------------------
# IMPORTANT TUNING PARAMETERS
# ------------------------------------------------------------

# More smoothing = fewer sharp grid lines
SIGNAL_SMOOTHING = 2.0

# Small GASF smoothing
GASF_SMOOTHING = 0.6

# Do NOT stretch the extreme 0-100% range.
# This keeps strong boundaries from becoming excessive.
LOW_PERCENTILE = 5.0
HIGH_PERCENTILE = 95.0

# Slight gamma correction
GAMMA = 1.0

# RGB colour map
COLORMAP = "jet"


# ============================================================
# LOAD EEG
# ============================================================

def load_eeg(file_path):

    print("\n" + "=" * 70)
    print("LOADING EEG")
    print("=" * 70)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"MAT file not found:\n{file_path}"
        )

    mat = loadmat(file_path)

    print("\nMATLAB variables:")

    for key, value in mat.items():

        if not key.startswith("__"):

            if isinstance(value, np.ndarray):

                print(
                    f"{key}: "
                    f"shape={value.shape}, "
                    f"dtype={value.dtype}"
                )

    if "EEG" not in mat:

        raise KeyError(
            "Variable 'EEG' was not found."
        )

    eeg = np.asarray(
        mat["EEG"],
        dtype=np.float64
    )

    print(
        f"\nOriginal EEG shape: {eeg.shape}"
    )

    if eeg.ndim != 2:

        raise ValueError(
            "EEG data must be a 2D array."
        )

    # Make shape = channels x samples
    if eeg.shape[0] > eeg.shape[1]:

        eeg = eeg.T

    print(
        f"Final EEG shape: {eeg.shape}"
    )

    return eeg


# ============================================================
# EXTRACT CHANNEL
# ============================================================

def get_channel(eeg, channel_number):

    print("\n" + "=" * 70)
    print("EXTRACTING CHANNEL")
    print("=" * 70)

    index = channel_number - 1

    if index < 0 or index >= eeg.shape[0]:

        raise ValueError(
            f"Invalid channel number: {channel_number}"
        )

    x = eeg[index].copy()

    print(
        f"Channel: {channel_number}"
    )

    print(
        f"Samples: {len(x)}"
    )

    return x


# ============================================================
# DETREND
# ============================================================

def detrend(x):

    print("\nRemoving linear trend...")

    return signal.detrend(
        x,
        type="linear"
    )


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass(x):

    print(
        f"Bandpass filtering: "
        f"{LOWCUT} - {HIGHCUT} Hz"
    )

    nyquist = FS / 2.0

    high = min(
        HIGHCUT,
        nyquist - 1
    )

    sos = signal.butter(
        4,
        [LOWCUT, high],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    return signal.sosfiltfilt(
        sos,
        x
    )


# ============================================================
# NOTCH FILTER
# ============================================================

def notch(x):

    print(
        f"Applying notch filter: "
        f"{NOTCH_FREQ} Hz"
    )

    # 50 Hz is below Nyquist for 128 Hz
    if NOTCH_FREQ >= FS / 2:

        print(
            "Notch skipped."
        )

        return x

    b, a = signal.iirnotch(
        w0=NOTCH_FREQ,
        Q=30,
        fs=FS
    )

    return signal.filtfilt(
        b,
        a,
        x
    )


# ============================================================
# SMOOTH EEG
# ============================================================

def smooth_eeg(x):

    print(
        f"EEG smoothing sigma: "
        f"{SIGNAL_SMOOTHING}"
    )

    return gaussian_filter(
        x,
        sigma=SIGNAL_SMOOTHING
    )


# ============================================================
# ROBUST NORMALIZATION
# ============================================================

def normalize(x):

    print("\nRobust normalization...")

    # Remove DC offset
    x = x - np.median(x)

    # Median absolute deviation
    mad = np.median(
        np.abs(x)
    )

    if mad < 1e-12:

        std = np.std(x)

        if std < 1e-12:

            raise ValueError(
                "EEG signal has almost no variation."
            )

        x = x / std

    else:

        x = x / (
            1.4826 * mad
        )

    # Remove extreme outliers
    x = np.clip(
        x,
        -3.0,
        3.0
    )

    # Scale to [-1, 1]
    maximum = np.max(
        np.abs(x)
    )

    if maximum > 0:

        x = x / maximum

    print(
        f"Normalized range: "
        f"{x.min():.4f} to {x.max():.4f}"
    )

    return x


# ============================================================
# RESAMPLE
# ============================================================

def resample(x):

    print(
        f"\nResampling to "
        f"{IMAGE_SIZE} samples..."
    )

    return signal.resample(
        x,
        IMAGE_SIZE
    )


# ============================================================
# CREATE GASF
# ============================================================

def create_gasf(x):

    print("\n" + "=" * 70)
    print("CREATING GASF")
    print("=" * 70)

    x = np.clip(
        x,
        -1.0,
        1.0
    )

    # Angular encoding
    phi = np.arccos(x)

    # GASF:
    #
    # G(i,j) = cos(phi_i + phi_j)
    #

    gasf = np.cos(
        phi[:, None] +
        phi[None, :]
    )

    print(
        f"GASF shape: {gasf.shape}"
    )

    print(
        f"Minimum: {gasf.min():.6f}"
    )

    print(
        f"Maximum: {gasf.max():.6f}"
    )

    print(
        f"Mean: {gasf.mean():.6f}"
    )

    return gasf


# ============================================================
# GASF SMOOTHING
# ============================================================

def smooth_gasf(gasf):

    print(
        f"\nGASF smoothing sigma: "
        f"{GASF_SMOOTHING}"
    )

    gasf = gaussian_filter(
        gasf,
        sigma=GASF_SMOOTHING
    )

    return np.clip(
        gasf,
        -1.0,
        1.0
    )


# ============================================================
# SOFT CONTRAST NORMALIZATION
# ============================================================

def normalize_gasf_for_color(gasf):

    print("\n" + "=" * 70)
    print("SOFT CONTRAST NORMALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Instead of forcing the complete range to [0,1],
    # use the central 90% of values.
    #
    # This reduces the extremely strong grid boundaries.
    # --------------------------------------------------------

    low = np.percentile(
        gasf,
        LOW_PERCENTILE
    )

    high = np.percentile(
        gasf,
        HIGH_PERCENTILE
    )

    print(
        f"Low percentile: "
        f"{LOW_PERCENTILE}% = {low:.6f}"
    )

    print(
        f"High percentile: "
        f"{HIGH_PERCENTILE}% = {high:.6f}"
    )

    if high <= low:

        low = gasf.min()
        high = gasf.max()

    img = (
        gasf - low
    ) / (
        high - low + 1e-12
    )

    img = np.clip(
        img,
        0.0,
        1.0
    )

    # Gamma = 1 means no artificial brightness distortion.
    if GAMMA != 1.0:

        img = np.power(
            img,
            GAMMA
        )

    return img


# ============================================================
# RGB CONVERSION
# ============================================================

def convert_to_rgb(gasf):

    print("\n" + "=" * 70)
    print("CONVERTING GASF TO RGB")
    print("=" * 70)

    normalized = normalize_gasf_for_color(
        gasf
    )

    cmap = plt.get_cmap(
        COLORMAP
    )

    rgba = cmap(
        normalized
    )

    rgb = (
        rgba[:, :, :3] * 255
    ).astype(
        np.uint8
    )

    print(
        f"RGB shape: {rgb.shape}"
    )

    print(
        f"RGB dtype: {rgb.dtype}"
    )

    return rgb


# ============================================================
# SAVE
# ============================================================

def save_rgb(rgb):

    output_file = os.path.join(
        OUTPUT_DIR,
        "TEST_fear_channel_01_GASF_RGB_V2.png"
    )

    image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    image.save(
        output_file,
        format="PNG"
    )

    print("\n" + "=" * 70)
    print("IMAGE SAVED")
    print("=" * 70)

    print(
        output_file
    )

    print(
        f"Size: "
        f"{image.width} x {image.height}"
    )

    print(
        "Mode: RGB"
    )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("EEG → FILTER → GASF → RGB")
    print("TEST VERSION 2")
    print("=" * 70)

    # 1. Load
    eeg = load_eeg(
        INPUT_FILE
    )

    # 2. Channel
    x = get_channel(
        eeg,
        CHANNEL_NUMBER
    )

    # 3. Remove trend
    x = detrend(
        x
    )

    # 4. Bandpass
    x = bandpass(
        x
    )

    # 5. Notch
    x = notch(
        x
    )

    # 6. Smooth
    x = smooth_eeg(
        x
    )

    # 7. Normalize
    x = normalize(
        x
    )

    # 8. Resample
    x = resample(
        x
    )

    # 9. GASF
    gasf = create_gasf(
        x
    )

    # 10. Smooth GASF
    gasf = smooth_gasf(
        gasf
    )

    # 11. RGB
    rgb = convert_to_rgb(
        gasf
    )

    # 12. Save
    output = save_rgb(
        rgb
    )

    print("\n")
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    print(
        "\nOnly ONE image was generated."
    )

    print(
        "The full dataset has NOT been modified."
    )

    print(
        "Do NOT train the CNN yet."
    )

    print(
        f"\nOpen:\n{output}"
    )

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()