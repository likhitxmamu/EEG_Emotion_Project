# ============================================================
# TEST STFT EEG SPECTROGRAM - HIGH RESOLUTION
# ============================================================
#
# PURPOSE:
#   Generate ONE clean STFT spectrogram for visual inspection.
#
# IMPORTANT:
#   - Existing dataset is NOT modified.
#   - CNN training is NOT started.
#   - Only ONE image is generated.
#   - Searches the entire project for EEG source files.
#
# OUTPUT:
#   data/processed/test_stft/TEST_EEG_STFT_RGB.png
#
# ============================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import loadmat


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

OUTPUT_DIR = os.path.join(
    DATA_ROOT,
    "processed",
    "test_stft"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "TEST_EEG_STFT_RGB.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EEG / STFT SETTINGS
# ============================================================

# Sampling frequency of the EEG data
FS = 200

# Frequency range that we want to display
FREQ_MIN = 1
FREQ_MAX = 45

# STFT window
N_PER_SEG = 256

# 75% overlap
N_OVERLAP = 192

# FFT resolution
N_FFT = 512

# Final image resolution
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# Channel to use if a multi-channel EEG matrix is found
CHANNEL_INDEX = 0


# ============================================================
# FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".mat",
    ".csv",
    ".npy",
    ".npz"
)


# ============================================================
# FOLDERS TO IGNORE
# ============================================================

# We don't want to accidentally read our generated PNG files
# or the CNN dataset as EEG input.

IGNORED_DIR_NAMES = {
    "spectrograms",
    "cnn_dataset",
    "test_gasf",
    "test_stft",
    "__pycache__",
    ".git",
    ".venv"
}


# ============================================================
# FIND EEG SOURCE FILE
# ============================================================

def find_eeg_files():

    print("\nSearching entire project for EEG source files...")
    print("Project root:")
    print(PROJECT_ROOT)

    found_files = []

    for root, dirs, files in os.walk(PROJECT_ROOT):

        # Prevent searching generated/output folders
        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIR_NAMES
        ]

        for filename in files:

            full_path = os.path.join(root, filename)

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in SUPPORTED_EXTENSIONS:
                found_files.append(full_path)

    # Remove duplicates
    found_files = sorted(list(set(found_files)))

    print("\nSupported files found:")

    if len(found_files) == 0:
        print("NO EEG SOURCE FILES FOUND.")

    else:
        for i, file_path in enumerate(found_files, start=1):
            print(f"{i}. {file_path}")

    return found_files


# ============================================================
# LOAD NUMERIC ARRAYS FROM MAT FILE
# ============================================================

def extract_numeric_arrays_from_mat(file_path):

    print("\nReading MAT file:")
    print(file_path)

    try:

        mat_data = loadmat(
            file_path,
            squeeze_me=True,
            struct_as_record=False
        )

    except Exception as e:

        print("Could not read MAT file:")
        print(e)

        return []

    arrays = []

    for key, value in mat_data.items():

        # Ignore MATLAB metadata
        if key.startswith("__"):
            continue

        try:

            arr = np.asarray(value)

        except Exception:
            continue

        # Must be numeric
        if not np.issubdtype(arr.dtype, np.number):
            continue

        # Remove NaN/Inf where possible
        try:

            if not np.isfinite(arr).any():
                continue

        except Exception:
            continue

        arrays.append(
            (key, arr)
        )

    return arrays


# ============================================================
# EXTRACT SIGNAL FROM ARRAY
# ============================================================

def extract_signal_from_array(arr):

    arr = np.asarray(arr)

    # --------------------------------------------------------
    # CASE 1:
    # Already a one-dimensional EEG signal
    # --------------------------------------------------------

    if arr.ndim == 1:

        signal_data = arr.astype(np.float64)

        return signal_data


    # --------------------------------------------------------
    # CASE 2:
    # Single-row or single-column signal
    # --------------------------------------------------------

    if arr.ndim == 2:

        rows, cols = arr.shape

        # One row
        if rows == 1:

            return arr[0, :].astype(np.float64)

        # One column
        if cols == 1:

            return arr[:, 0].astype(np.float64)

        # ----------------------------------------------------
        # Multi-channel EEG
        # ----------------------------------------------------
        #
        # Usually:
        #   rows = channels
        #   cols = samples
        #
        # If the first dimension is relatively small and the
        # second dimension is much larger, treat rows as
        # channels.
        # ----------------------------------------------------

        if rows <= 256 and cols > rows:

            channel = min(
                CHANNEL_INDEX,
                rows - 1
            )

            return arr[channel, :].astype(np.float64)

        # Alternative orientation:
        # samples x channels

        if cols <= 256 and rows > cols:

            channel = min(
                CHANNEL_INDEX,
                cols - 1
            )

            return arr[:, channel].astype(np.float64)

        # If unclear, use the longest dimension
        # and take the first available signal.

        if cols >= rows:

            return arr[0, :].astype(np.float64)

        else:

            return arr[:, 0].astype(np.float64)


    # --------------------------------------------------------
    # CASE 3:
    # Higher-dimensional EEG data
    # --------------------------------------------------------

    if arr.ndim >= 3:

        shape = arr.shape

        print(
            f"Multi-dimensional EEG array detected: {shape}"
        )

        # Look for a dimension that resembles channels.
        #
        # Common EEG channel counts:
        # 8, 14, 16, 32, 40, 62, 64, 128, 256

        possible_channel_dims = [
            i for i, size in enumerate(shape)
            if size in (
                8, 14, 16, 18, 19, 20,
                32, 40, 62, 63, 64,
                128, 256
            )
        ]

        if possible_channel_dims:

            channel_dim = possible_channel_dims[0]

            # Move channel dimension to front
            moved = np.moveaxis(
                arr,
                channel_dim,
                0
            )

            channel_count = moved.shape[0]

            channel = min(
                CHANNEL_INDEX,
                channel_count - 1
            )

            channel_data = moved[channel]

            # Flatten remaining dimensions
            channel_data = np.asarray(
                channel_data
            ).squeeze()

            # If still multidimensional, take the longest
            # dimension as the temporal dimension.

            if channel_data.ndim > 1:

                dimensions = channel_data.shape

                time_dim = np.argmax(
                    dimensions
                )

                channel_data = np.moveaxis(
                    channel_data,
                    time_dim,
                    0
                )

                channel_data = channel_data.reshape(
                    channel_data.shape[0],
                    -1
                )[:, 0]

            return channel_data.astype(np.float64)

        # ----------------------------------------------------
        # No obvious channel dimension.
        # Find the largest numeric slice.
        # ----------------------------------------------------

        flat = arr.reshape(-1)

        return flat.astype(np.float64)


    return None


# ============================================================
# LOAD EEG SIGNAL
# ============================================================

def load_eeg_signal():

    files = find_eeg_files()

    if len(files) == 0:

        raise FileNotFoundError(
            "\n"
            "NO EEG SOURCE FILE WAS FOUND.\n\n"
            "The script searched the complete project:\n"
            f"{PROJECT_ROOT}\n\n"
            "Supported files:\n"
            "  .mat\n"
            "  .csv\n"
            "  .npy\n"
            "  .npz\n"
        )

    # --------------------------------------------------------
    # Try files one by one
    # --------------------------------------------------------

    for file_path in files:

        extension = os.path.splitext(
            file_path
        )[1].lower()

        print("\n----------------------------------------")
        print("Trying:")
        print(file_path)
        print("----------------------------------------")

        # ====================================================
        # MAT
        # ====================================================

        if extension == ".mat":

            arrays = extract_numeric_arrays_from_mat(
                file_path
            )

            # Prefer large arrays because EEG recordings
            # normally contain many samples.

            arrays = sorted(
                arrays,
                key=lambda x: x[1].size,
                reverse=True
            )

            for name, arr in arrays:

                print(
                    f"MAT variable: {name} | "
                    f"shape={arr.shape}"
                )

                try:

                    eeg_signal = extract_signal_from_array(
                        arr
                    )

                except Exception as e:

                    print(
                        f"Could not extract {name}: {e}"
                    )

                    continue

                if eeg_signal is not None:

                    if len(eeg_signal) >= N_PER_SEG * 2:

                        print(
                            "\nSUCCESS!"
                        )

                        print(
                            f"Variable: {name}"
                        )

                        print(
                            f"Signal samples: "
                            f"{len(eeg_signal)}"
                        )

                        return eeg_signal, file_path, name


        # ====================================================
        # CSV
        # ====================================================

        elif extension == ".csv":

            try:

                arr = np.genfromtxt(
                    file_path,
                    delimiter=",",
                    dtype=np.float64
                )

                eeg_signal = extract_signal_from_array(
                    arr
                )

                if eeg_signal is not None:

                    if len(eeg_signal) >= N_PER_SEG * 2:

                        print("\nSUCCESS!")

                        return (
                            eeg_signal,
                            file_path,
                            "CSV"
                        )

            except Exception as e:

                print(
                    f"CSV error: {e}"
                )


        # ====================================================
        # NPY
        # ====================================================

        elif extension == ".npy":

            try:

                arr = np.load(
                    file_path,
                    allow_pickle=False
                )

                eeg_signal = extract_signal_from_array(
                    arr
                )

                if eeg_signal is not None:

                    if len(eeg_signal) >= N_PER_SEG * 2:

                        print("\nSUCCESS!")

                        return (
                            eeg_signal,
                            file_path,
                            "NPY"
                        )

            except Exception as e:

                print(
                    f"NPY error: {e}"
                )


        # ====================================================
        # NPZ
        # ====================================================

        elif extension == ".npz":

            try:

                data = np.load(
                    file_path,
                    allow_pickle=False
                )

                keys = list(data.keys())

                for key in keys:

                    arr = data[key]

                    print(
                        f"NPZ variable: {key} | "
                        f"shape={arr.shape}"
                    )

                    eeg_signal = extract_signal_from_array(
                        arr
                    )

                    if eeg_signal is not None:

                        if len(eeg_signal) >= N_PER_SEG * 2:

                            print("\nSUCCESS!")

                            return (
                                eeg_signal,
                                file_path,
                                key
                            )

            except Exception as e:

                print(
                    f"NPZ error: {e}"
                )

    raise RuntimeError(
        "\n"
        "EEG files were found, but a usable EEG signal "
        "could not be extracted."
    )


# ============================================================
# CLEAN EEG SIGNAL
# ============================================================

def clean_eeg_signal(eeg):

    eeg = np.asarray(
        eeg,
        dtype=np.float64
    )

    # Remove NaN / infinity
    eeg = np.nan_to_num(
        eeg,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Remove DC offset
    eeg = eeg - np.mean(eeg)

    # --------------------------------------------------------
    # Band-pass filter
    # 1-45 Hz
    # --------------------------------------------------------

    low = FREQ_MIN
    high = FREQ_MAX

    nyquist = FS / 2

    low_norm = low / nyquist
    high_norm = high / nyquist

    b, a = signal.butter(
        4,
        [low_norm, high_norm],
        btype="bandpass"
    )

    eeg = signal.filtfilt(
        b,
        a,
        eeg
    )

    # --------------------------------------------------------
    # Remove remaining DC / slow drift
    # --------------------------------------------------------

    eeg = signal.detrend(
        eeg,
        type="linear"
    )

    return eeg


# ============================================================
# CREATE STFT
# ============================================================

def create_stft(eeg):

    print("\nCreating STFT...")

    frequencies, times, Zxx = signal.stft(
        eeg,
        fs=FS,
        window="hann",
        nperseg=N_PER_SEG,
        noverlap=N_OVERLAP,
        nfft=N_FFT,
        detrend="constant",
        boundary=None,
        padded=False
    )

    # Magnitude
    magnitude = np.abs(Zxx)

    # Convert to power
    power = magnitude ** 2

    # Convert to decibels
    power_db = 10 * np.log10(
        power + 1e-12
    )

    # --------------------------------------------------------
    # Keep only requested frequency range
    # --------------------------------------------------------

    frequency_mask = (
        (frequencies >= FREQ_MIN)
        &
        (frequencies <= FREQ_MAX)
    )

    frequencies = frequencies[
        frequency_mask
    ]

    power_db = power_db[
        frequency_mask,
        :
    ]

    # --------------------------------------------------------
    # Robust contrast normalization
    #
    # This prevents a few extreme values from making the
    # entire image almost one color.
    # --------------------------------------------------------

    lower = np.percentile(
        power_db,
        5
    )

    upper = np.percentile(
        power_db,
        98
    )

    power_db = np.clip(
        power_db,
        lower,
        upper
    )

    print(
        f"Frequency range: "
        f"{frequencies[0]:.2f} - "
        f"{frequencies[-1]:.2f} Hz"
    )

    print(
        f"STFT matrix: "
        f"{power_db.shape}"
    )

    return frequencies, times, power_db


# ============================================================
# CREATE CLEAN IMAGE
# ============================================================

def save_spectrogram(
    frequencies,
    times,
    power_db
):

    print("\nCreating high-resolution image...")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No axes
    # No grid
    # No labels
    # No title
    # No borders
    # No interpolation artifacts
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(10.24, 10.24),
        dpi=100,
        frameon=False
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    # --------------------------------------------------------
    # Spectrogram
    #
    # "turbo" gives good separation between power levels
    # while retaining the visual character of the previous
    # image.
    # --------------------------------------------------------

    ax.pcolormesh(
        times,
        frequencies,
        power_db,
        shading="gouraud",
        cmap="turbo",
        antialiased=True
    )

    # Remove everything except image
    ax.set_axis_off()

    ax.set_xlim(
        times[0],
        times[-1]
    )

    ax.set_ylim(
        FREQ_MIN,
        FREQ_MAX
    )

    # Remove margins
    ax.margins(
        x=0,
        y=0
    )

    # --------------------------------------------------------
    # Save EXACTLY 1024 x 1024
    # --------------------------------------------------------

    fig.savefig(
        OUTPUT_FILE,
        dpi=100,
        bbox_inches=None,
        pad_inches=0,
        facecolor="black",
        edgecolor="none"
    )

    plt.close(fig)

    print("\nIMAGE SAVED")
    print("=" * 60)
    print(f"File : {OUTPUT_FILE}")
    print("Size : 1024 x 1024")
    print("Mode : RGB")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("HIGH-RESOLUTION EEG STFT TEST")
    print("=" * 60)

    print("\nIMPORTANT:")
    print("Only ONE image will be generated.")
    print("Existing datasets will NOT be modified.")
    print("CNN training will NOT start.")

    print("\nSettings:")
    print(f"Sampling frequency : {FS} Hz")
    print(
        f"Frequency range    : "
        f"{FREQ_MIN}-{FREQ_MAX} Hz"
    )
    print(
        f"STFT window        : "
        f"{N_PER_SEG} samples"
    )
    print(
        f"STFT overlap       : "
        f"{N_OVERLAP} samples"
    )
    print(
        f"FFT resolution     : "
        f"{N_FFT}"
    )
    print(
        f"Output resolution  : "
        f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
    )

    # --------------------------------------------------------
    # Load EEG
    # --------------------------------------------------------

    eeg, source_file, variable_name = load_eeg_signal()

    print("\n")
    print("=" * 60)
    print("SOURCE EEG FOUND")
    print("=" * 60)

    print(
        f"File     : {source_file}"
    )

    print(
        f"Variable : {variable_name}"
    )

    print(
        f"Samples  : {len(eeg)}"
    )

    print(
        f"Duration : {len(eeg) / FS:.2f} seconds"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    eeg = clean_eeg_signal(
        eeg
    )

    print("\nEEG preprocessing complete.")

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    frequencies, times, power_db = create_stft(
        eeg
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_spectrogram(
        frequencies,
        times,
        power_db
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    print("\nONLY ONE IMAGE WAS GENERATED.")
    print("THE EXISTING DATASET WAS NOT MODIFIED.")
    print("CNN TRAINING WAS NOT STARTED.")

    print("\nOpen this image and inspect it:")
    print(
        "data/processed/test_stft/"
        "TEST_EEG_STFT_RGB.png"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()