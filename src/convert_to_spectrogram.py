# ============================================================
# EEG EMOTION PROJECT
# EEG -> STFT SPECTROGRAM DATASET GENERATOR
# ============================================================
#
# PURPOSE:
#   Convert EEG recordings into MANY STFT spectrogram images.
#
# SOURCE:
#   data/output/
#       fear/
#       happy/
#       neutral/
#       sad/
#
# OUTPUT (new representation; original is read-only):
#   data/processed/stft_spectrograms_global/
#       fear/
#       happy/
#       neutral/
#       sad/
#
# DATASET:
#   Target = 1024 images
#   256 images per emotion
#
# IMPORTANT:
#   One EEG recording is split into MANY overlapping windows.
#   Each window produces one STFT image.
#
#   Original EEG files are NEVER modified.
#
# ============================================================


import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import loadmat


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data"
)

# EEG SOURCE DATA
SOURCE_ROOT = os.path.join(
    DATA_ROOT,
    "output"
)

# ORIGINAL STFT DATASET (READ-ONLY)
ORIGINAL_OUTPUT_ROOT = os.path.join(
    DATA_ROOT,
    "processed",
    "stft_spectrograms"
)

# NEW GLOBALLY SCALED STFT DATASET
OUTPUT_ROOT = os.path.join(
    DATA_ROOT,
    "processed",
    "stft_spectrograms_global"
)

TRAINING_DATASET_ROOT = os.path.join(
    DATA_ROOT,
    "processed",
    "crnn_dataset"
)

GLOBAL_SCALE_REPORT = os.path.join(
    DATA_ROOT,
    "diagnostics",
    "global_stft_experiment",
    "stft_scale.json"
)

SEED = 42
TRIAL_FILENAME_PATTERN = re.compile(
    r"^(?P<trial>.+_trial\d+)_window_\d+_STFT_RGB$",
    re.IGNORECASE
)

np.random.seed(SEED)


# ============================================================
# EEG SETTINGS
# ============================================================

# EEG sampling frequency
FS = 200

# Frequency range used for EEG
FREQ_MIN = 1
FREQ_MAX = 45

# ------------------------------------------------------------
# WINDOW SETTINGS
# ------------------------------------------------------------
#
# Each EEG recording is divided into windows.
#
# 2 seconds = 400 samples at 200 Hz.
#
# 75% overlap means:
#
# Window 1: 0.0 - 2.0 sec
# Window 2: 0.5 - 2.5 sec
# Window 3: 1.0 - 3.0 sec
# Window 4: 1.5 - 3.5 sec
# etc.
#
# This creates MANY images from one EEG recording.
# ------------------------------------------------------------

WINDOW_SECONDS = 2.0

WINDOW_SAMPLES = int(
    FS * WINDOW_SECONDS
)

OVERLAP_RATIO = 0.75

STEP_SAMPLES = int(
    WINDOW_SAMPLES * (1 - OVERLAP_RATIO)
)


# ============================================================
# STFT SETTINGS
# ============================================================

N_PER_SEG = 256

N_OVERLAP = 192

N_FFT = 512


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_WIDTH = 1024

IMAGE_HEIGHT = 1024

IMAGE_DPI = 100


# ============================================================
# EEG CHANNEL
# ============================================================

# Channel 0 is used.
#
# If your EEG contains multiple channels, this can later
# be changed after we verify the actual dataset structure.

CHANNEL_INDEX = 0


# ============================================================
# EMOTIONS
# ============================================================

EMOTIONS = [
    "fear",
    "happy",
    "neutral",
    "sad"
]


# ============================================================
# TARGET DATASET SIZE
# ============================================================

TARGET_IMAGES_PER_EMOTION = 256

TARGET_TOTAL_IMAGES = (
    TARGET_IMAGES_PER_EMOTION
    * len(EMOTIONS)
)


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".mat",
    ".csv",
    ".npy",
    ".npz"
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders():

    for emotion in EMOTIONS:

        folder = os.path.join(
            OUTPUT_ROOT,
            emotion
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# ============================================================
# PREPARE NEW STFT DATASET
# ============================================================

def prepare_new_stft_output():

    print("\n")
    print("=" * 75)
    print("PREPARING NEW GLOBAL-SCALE STFT OUTPUT")
    print("=" * 75)

    output_path = os.path.abspath(OUTPUT_ROOT)
    original_path = os.path.abspath(ORIGINAL_OUTPUT_ROOT)
    if output_path == original_path or output_path.startswith(original_path + os.sep):
        raise RuntimeError(
            "Refusing to clear output because it is the original STFT dataset: "
            f"{original_path}"
        )
    if os.path.islink(output_path):
        raise RuntimeError(f"Refusing to clear a symlink output path: {output_path}")

    print(f"New output path:          {output_path}")
    print(f"Original path preserved:  {original_path}")

    if os.path.lexists(OUTPUT_ROOT):
        raise FileExistsError(
            "Refusing to overwrite or delete an existing dataset. "
            f"Choose a new generated output path: {OUTPUT_ROOT}"
        )

    create_output_folders()

    print("New output folders created; no existing dataset was removed.")


# ============================================================
# FIND ALL EEG FILES
# ============================================================

def find_eeg_files():

    print("\n")
    print("=" * 75)
    print("SEARCHING FOR EEG SOURCE FILES")
    print("=" * 75)

    print(
        f"\nSource folder:\n{SOURCE_ROOT}"
    )

    all_files = []

    for emotion in EMOTIONS:

        emotion_folder = os.path.join(
            SOURCE_ROOT,
            emotion
        )

        if not os.path.isdir(
            emotion_folder
        ):

            print(
                f"\nWARNING: Missing folder:"
            )

            print(
                emotion_folder
            )

            continue

        emotion_files = []

        for root, dirs, files in os.walk(
            emotion_folder
        ):

            for filename in files:

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension not in SUPPORTED_EXTENSIONS:
                    continue

                full_path = os.path.join(
                    root,
                    filename
                )

                emotion_files.append(
                    full_path
                )

        emotion_files = sorted(
            list(
                set(
                    emotion_files
                )
            )
        )

        print(
            f"\n{emotion.upper()}: "
            f"{len(emotion_files)} EEG files"
        )

        all_files.extend(
            emotion_files
        )

    all_files = sorted(
        list(
            set(
                all_files
            )
        )
    )

    print("\n")
    print("=" * 75)

    print(
        f"TOTAL EEG SOURCE FILES: "
        f"{len(all_files)}"
    )

    print("=" * 75)

    return all_files


# ============================================================
# DETERMINE EMOTION
# ============================================================

def get_emotion_from_path(
    file_path
):

    normalized_path = os.path.normpath(
        file_path
    )

    path_parts = normalized_path.split(
        os.sep
    )

    for emotion in EMOTIONS:

        if emotion in path_parts:

            return emotion

    return None


# ============================================================
# LOAD MATLAB ARRAYS
# ============================================================

def extract_numeric_arrays_from_mat(
    file_path
):

    try:

        mat_data = loadmat(
            file_path,
            squeeze_me=True,
            struct_as_record=False
        )

    except Exception as e:

        print(
            f"MAT read error: {e}"
        )

        return []

    arrays = []

    for key, value in mat_data.items():

        if key.startswith(
            "__"
        ):

            continue

        try:

            arr = np.asarray(
                value
            )

        except Exception:

            continue

        if not np.issubdtype(
            arr.dtype,
            np.number
        ):

            continue

        try:

            if not np.isfinite(
                arr
            ).any():

                continue

        except Exception:

            continue

        arrays.append(
            (
                key,
                arr
            )
        )

    return arrays


# ============================================================
# EXTRACT EEG SIGNAL
# ============================================================

def extract_signal_from_array(
    arr
):

    arr = np.asarray(
        arr
    )

    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    if arr.ndim == 1:

        return arr.astype(
            np.float64
        )


    # --------------------------------------------------------
    # 2D
    # --------------------------------------------------------

    if arr.ndim == 2:

        rows, cols = arr.shape

        # One row
        if rows == 1:

            return arr[0, :].astype(
                np.float64
            )

        # One column
        if cols == 1:

            return arr[:, 0].astype(
                np.float64
            )

        # Channels x samples
        if rows <= 256 and cols > rows:

            channel = min(
                CHANNEL_INDEX,
                rows - 1
            )

            return arr[
                channel,
                :
            ].astype(
                np.float64
            )

        # Samples x channels
        if cols <= 256 and rows > cols:

            channel = min(
                CHANNEL_INDEX,
                cols - 1
            )

            return arr[
                :,
                channel
            ].astype(
                np.float64
            )

        # Unknown orientation
        if cols >= rows:

            return arr[
                0,
                :
            ].astype(
                np.float64
            )

        return arr[
            :,
            0
        ].astype(
            np.float64
        )


    # --------------------------------------------------------
    # 3D or higher
    # --------------------------------------------------------

    if arr.ndim >= 3:

        shape = arr.shape

        print(
            f"Multi-dimensional EEG shape: "
            f"{shape}"
        )

        possible_channel_dims = [
            i
            for i, size in enumerate(shape)
            if size in (
                8,
                14,
                16,
                18,
                19,
                20,
                32,
                40,
                62,
                63,
                64,
                128,
                256
            )
        ]

        if possible_channel_dims:

            channel_dim = (
                possible_channel_dims[0]
            )

            moved = np.moveaxis(
                arr,
                channel_dim,
                0
            )

            channel_count = (
                moved.shape[0]
            )

            channel = min(
                CHANNEL_INDEX,
                channel_count - 1
            )

            channel_data = moved[
                channel
            ]

            channel_data = np.asarray(
                channel_data
            ).squeeze()

            if channel_data.ndim > 1:

                dimensions = (
                    channel_data.shape
                )

                time_dim = np.argmax(
                    dimensions
                )

                channel_data = np.moveaxis(
                    channel_data,
                    time_dim,
                    0
                )

                channel_data = (
                    channel_data.reshape(
                        channel_data.shape[0],
                        -1
                    )[:, 0]
                )

            return channel_data.astype(
                np.float64
            )

        # Fallback
        flat = arr.reshape(
            -1
        )

        return flat.astype(
            np.float64
        )

    return None


# ============================================================
# LOAD ONE EEG FILE
# ============================================================

def load_one_eeg_file(
    file_path
):

    extension = os.path.splitext(
        file_path
    )[1].lower()


    # ========================================================
    # MAT
    # ========================================================

    if extension == ".mat":

        arrays = (
            extract_numeric_arrays_from_mat(
                file_path
            )
        )

        # Largest numeric arrays first
        arrays = sorted(
            arrays,
            key=lambda x: x[1].size,
            reverse=True
        )

        for variable_name, arr in arrays:

            try:

                eeg = extract_signal_from_array(
                    arr
                )

            except Exception:

                continue

            if eeg is None:

                continue

            if len(eeg) < WINDOW_SAMPLES:

                continue

            return (
                eeg,
                variable_name
            )

        return None, None


    # ========================================================
    # CSV
    # ========================================================

    if extension == ".csv":

        try:

            arr = np.genfromtxt(
                file_path,
                delimiter=",",
                dtype=np.float64
            )

            eeg = extract_signal_from_array(
                arr
            )

            if (
                eeg is not None
                and len(eeg) >= WINDOW_SAMPLES
            ):

                return (
                    eeg,
                    "CSV"
                )

        except Exception as e:

            print(
                f"CSV error: {e}"
            )

        return None, None


    # ========================================================
    # NPY
    # ========================================================

    if extension == ".npy":

        try:

            arr = np.load(
                file_path,
                allow_pickle=False
            )

            eeg = extract_signal_from_array(
                arr
            )

            if (
                eeg is not None
                and len(eeg) >= WINDOW_SAMPLES
            ):

                return (
                    eeg,
                    "NPY"
                )

        except Exception as e:

            print(
                f"NPY error: {e}"
            )

        return None, None


    # ========================================================
    # NPZ
    # ========================================================

    if extension == ".npz":

        try:

            data = np.load(
                file_path,
                allow_pickle=False
            )

            keys = sorted(
                data.keys(),
                key=lambda k: data[k].size,
                reverse=True
            )

            for key in keys:

                arr = data[key]

                eeg = extract_signal_from_array(
                    arr
                )

                if (
                    eeg is not None
                    and len(eeg) >= WINDOW_SAMPLES
                ):

                    return (
                        eeg,
                        key
                    )

        except Exception as e:

            print(
                f"NPZ error: {e}"
            )

        return None, None


    return None, None


# ============================================================
# CLEAN EEG SIGNAL
# ============================================================

def clean_eeg_signal(
    eeg
):

    eeg = np.asarray(
        eeg,
        dtype=np.float64
    )

    # Remove NaN and infinity
    eeg = np.nan_to_num(
        eeg,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Remove DC
    eeg = (
        eeg
        - np.mean(eeg)
    )

    # --------------------------------------------------------
    # Band-pass filter 1-45 Hz
    # --------------------------------------------------------

    nyquist = FS / 2

    low_norm = (
        FREQ_MIN / nyquist
    )

    high_norm = (
        FREQ_MAX / nyquist
    )

    b, a = signal.butter(
        4,
        [
            low_norm,
            high_norm
        ],
        btype="bandpass"
    )

    # Need enough samples for filtfilt
    if len(eeg) > 30:

        eeg = signal.filtfilt(
            b,
            a,
            eeg
        )

    # Remove linear trend
    eeg = signal.detrend(
        eeg,
        type="linear"
    )

    return eeg


# ============================================================
# SPLIT EEG INTO WINDOWS
# ============================================================

def create_eeg_windows(
    eeg
):

    windows = []

    total_samples = len(
        eeg
    )

    # --------------------------------------------------------
    # Generate overlapping windows
    # --------------------------------------------------------

    start = 0

    while (
        start + WINDOW_SAMPLES
        <= total_samples
    ):

        end = (
            start
            + WINDOW_SAMPLES
        )

        window = eeg[
            start:end
        ]

        windows.append(
            (
                start,
                end,
                window
            )
        )

        start += STEP_SAMPLES

    return windows


# ============================================================
# CREATE STFT
# ============================================================

def create_stft(
    eeg
):

    frequencies, times, Zxx = (
        signal.stft(
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
    )

    # Magnitude
    magnitude = np.abs(
        Zxx
    )

    # Power
    power = (
        magnitude ** 2
    )

    # dB
    power_db = (
        10
        * np.log10(
            power + 1e-12
        )
    )

    # Frequency filtering
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

    return (
        frequencies,
        times,
        power_db
    )


# ============================================================
# SAVE STFT IMAGE
# ============================================================

def save_spectrogram(
    frequencies,
    times,
    power_db,
    output_file,
    global_db_min,
    global_db_max
):

    fig = plt.figure(
        figsize=(
            IMAGE_WIDTH / IMAGE_DPI,
            IMAGE_HEIGHT / IMAGE_DPI
        ),
        dpi=IMAGE_DPI,
        frameon=False
    )

    ax = fig.add_axes(
        [
            0,
            0,
            1,
            1
        ]
    )

    ax.pcolormesh(
        times,
        frequencies,
        power_db,
        shading="gouraud",
        cmap="turbo",
        antialiased=True,
        vmin=global_db_min,
        vmax=global_db_max
    )

    ax.set_axis_off()

    ax.set_xlim(
        times[0],
        times[-1]
    )

    ax.set_ylim(
        FREQ_MIN,
        FREQ_MAX
    )

    ax.margins(
        x=0,
        y=0
    )

    fig.savefig(
        output_file,
        dpi=IMAGE_DPI,
        bbox_inches=None,
        pad_inches=0,
        facecolor="black",
        edgecolor="none"
    )

    plt.close(
        fig
    )


# ============================================================
# GENERATE ALL WINDOWS FOR ONE EMOTION
# ============================================================

def collect_windows_for_emotion(
    emotion_files,
    emotion
):

    print("\n")
    print("=" * 75)

    print(
        f"COLLECTING EEG WINDOWS: "
        f"{emotion.upper()}"
    )

    print("=" * 75)

    all_windows = []

    for file_index, file_path in enumerate(
        emotion_files,
        start=1
    ):

        print("\n")
        print(
            f"[{emotion.upper()}] "
            f"FILE {file_index}/{len(emotion_files)}"
        )

        print(
            os.path.basename(
                file_path
            )
        )

        eeg, variable_name = (
            load_one_eeg_file(
                file_path
            )
        )

        if eeg is None:

            print(
                "FAILED: Could not load EEG."
            )

            continue

        print(
            f"Variable: {variable_name}"
        )

        print(
            f"Samples: {len(eeg)}"
        )

        print(
            f"Duration: "
            f"{len(eeg) / FS:.2f} seconds"
        )

        # Clean entire recording first
        eeg = clean_eeg_signal(
            eeg
        )

        # Create windows
        windows = create_eeg_windows(
            eeg
        )

        print(
            f"Windows generated: "
            f"{len(windows)}"
        )

        for window_number, (
            start,
            end,
            window
        ) in enumerate(
            windows,
            start=1
        ):

            all_windows.append(
                {
                    "file_path": file_path,
                    "emotion": emotion,
                    "window_number": window_number,
                    "start": start,
                    "end": end,
                    "signal": window
                }
            )

    print("\n")
    print(
        f"TOTAL WINDOWS FOR "
        f"{emotion.upper()}: "
        f"{len(all_windows)}"
    )

    return all_windows


# ============================================================
# SELECT DATASET WINDOWS
# ============================================================

def select_windows(
    windows,
    target_count
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We do NOT duplicate images.
    #
    # If enough real windows exist, we select exactly the
    # required number.
    #
    # The selection is deterministic so the same data gives
    # the same dataset each time.
    # --------------------------------------------------------

    if len(windows) < target_count:

        print("\n")
        print(
            "WARNING:"
        )

        print(
            f"Only {len(windows)} windows available."
        )

        print(
            f"Cannot create {target_count} "
            f"unique images without additional "
            f"EEG data."
        )

        return windows

    # Deterministic selection
    indices = np.linspace(
        0,
        len(windows) - 1,
        target_count,
        dtype=int
    )

    selected = [
        windows[index]
        for index in indices
    ]

    return selected


def get_trial_key_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = TRIAL_FILENAME_PATTERN.fullmatch(stem)
    if match is not None:
        return match.group("trial")
    raw_trial_match = re.fullmatch(r"(?P<trial>.+_trial\d+)", stem, re.IGNORECASE)
    if raw_trial_match is not None:
        return raw_trial_match.group("trial")
    raise ValueError(f"Cannot parse trial key from filename: {filename}")


def load_training_trial_keys():
    """Read trial assignment from the existing CRNN training split only."""
    keys = set()
    for emotion in EMOTIONS:
        emotion_folder = os.path.join(TRAINING_DATASET_ROOT, "train", emotion)
        if not os.path.isdir(emotion_folder):
            raise FileNotFoundError(
                "The existing trial-level training split is required to derive "
                f"the global STFT scale: {emotion_folder}"
            )
        for filename in sorted(os.listdir(emotion_folder)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                keys.add((emotion, get_trial_key_from_filename(filename)))
    if not keys:
        raise ValueError("No trial assignments found in the existing training split.")
    return keys


def compute_global_training_scale(selected_by_emotion, training_trial_keys):
    """Calculate one shared 1st–99th percentile dB range from training trials."""
    training_values = []
    training_counts = {emotion: 0 for emotion in EMOTIONS}

    for emotion, selected_windows in selected_by_emotion.items():
        for item in selected_windows:
            trial_key = get_trial_key_from_filename(item["file_path"])
            if (emotion, trial_key) not in training_trial_keys:
                continue
            _, _, power_db = create_stft(item["signal"])
            training_values.append(power_db.ravel())
            training_counts[emotion] += 1

    missing_classes = [emotion for emotion, count in training_counts.items() if count == 0]
    if missing_classes:
        raise ValueError(
            "No selected training windows available for class(es): "
            f"{missing_classes}"
        )
    if not training_values:
        raise ValueError("No training STFT values found for global scaling.")

    values = np.concatenate(training_values)
    db_min, db_max = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(db_min) or not np.isfinite(db_max) or db_min >= db_max:
        raise ValueError(f"Invalid global STFT scale: {db_min}, {db_max}")

    os.makedirs(os.path.dirname(GLOBAL_SCALE_REPORT), exist_ok=True)
    with open(GLOBAL_SCALE_REPORT, "w", encoding="utf-8") as report_file:
        json.dump(
            {
                "seed": SEED,
                "scale_source": "selected windows from existing CRNN train trials only",
                "percentiles": [1.0, 99.0],
                "global_db_min": float(db_min),
                "global_db_max": float(db_max),
                "training_window_counts_by_emotion": training_counts,
                "training_trial_keys": {
                    emotion: sorted(trial for cls, trial in training_trial_keys if cls == emotion)
                    for emotion in EMOTIONS
                },
            },
            report_file,
            indent=2,
        )

    print("\nGLOBAL STFT SCALE (derived from training trials only)")
    print(f"  Training power dB range: {db_min:.6f} to {db_max:.6f}")
    print(f"  Training window counts: {training_counts}")
    print(f"  Scale metadata: {GLOBAL_SCALE_REPORT}")
    return float(db_min), float(db_max)


# ============================================================
# GENERATE IMAGES
# ============================================================

def generate_images(
    selected_windows,
    emotion,
    global_db_min,
    global_db_max
):

    print("\n")
    print("=" * 75)

    print(
        f"GENERATING STFT IMAGES: "
        f"{emotion.upper()}"
    )

    print("=" * 75)

    output_folder = os.path.join(
        OUTPUT_ROOT,
        emotion
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    created = 0

    total = len(
        selected_windows
    )

    for index, item in enumerate(
        selected_windows,
        start=1
    ):

        file_path = item[
            "file_path"
        ]

        window_number = item[
            "window_number"
        ]

        eeg_window = item[
            "signal"
        ]

        try:

            frequencies, times, power_db = (
                create_stft(
                    eeg_window
                )
            )

            original_name = os.path.splitext(
                os.path.basename(
                    file_path
                )
            )[0]

            output_filename = (
                f"{original_name}"
                f"_window_{window_number:04d}"
                f"_STFT_RGB.png"
            )

            output_file = os.path.join(
                output_folder,
                output_filename
            )

            save_spectrogram(
                frequencies,
                times,
                power_db,
                output_file,
                global_db_min,
                global_db_max
            )

            created += 1

            if (
                index % 25 == 0
                or index == total
            ):

                print(
                    f"{emotion.upper()}: "
                    f"{index}/{total} images"
                )

        except Exception as e:

            print("\n")
            print(
                "ERROR creating image:"
            )

            print(
                file_path
            )

            print(
                f"Window: "
                f"{window_number}"
            )

            print(
                f"Error: {e}"
            )

    return created


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print(
        "EEG EMOTION DATASET"
    )
    print(
        "STFT SPECTROGRAM GENERATOR"
    )
    print("=" * 75)

    print("\n")
    print("PROJECT SETTINGS")

    print(
        f"Sampling frequency : "
        f"{FS} Hz"
    )

    print(
        f"Frequency range    : "
        f"{FREQ_MIN}-{FREQ_MAX} Hz"
    )

    print(
        f"Window duration    : "
        f"{WINDOW_SECONDS} seconds"
    )

    print(
        f"Window samples     : "
        f"{WINDOW_SAMPLES}"
    )

    print(
        f"Overlap            : "
        f"{OVERLAP_RATIO * 100:.0f}%"
    )

    print(
        f"Step size          : "
        f"{STEP_SAMPLES} samples"
    )

    print(
        f"STFT window        : "
        f"{N_PER_SEG}"
    )

    print(
        f"STFT overlap       : "
        f"{N_OVERLAP}"
    )

    print(
        f"FFT                 : "
        f"{N_FFT}"
    )

    print(
        f"Image size         : "
        f"{IMAGE_WIDTH} x "
        f"{IMAGE_HEIGHT}"
    )

    print(
        f"Target per emotion : "
        f"{TARGET_IMAGES_PER_EMOTION}"
    )

    print(
        f"Target total       : "
        f"{TARGET_TOTAL_IMAGES}"
    )

    print("\n")
    print(
        "SOURCE:"
    )

    print(
        SOURCE_ROOT
    )

    print("\n")
    print(
        "OUTPUT:"
    )

    print(
        OUTPUT_ROOT
    )


    # ========================================================
    # FIND SOURCE FILES
    # ========================================================

    eeg_files = find_eeg_files()

    if len(
        eeg_files
    ) == 0:

        print("\n")
        print("=" * 75)
        print(
            "ERROR: NO EEG FILES FOUND"
        )
        print("=" * 75)

        return


    # ========================================================
    # CREATE THE NEW STFT OUTPUT WITHOUT TOUCHING EXISTING DATA
    # ========================================================

    prepare_new_stft_output()


    # ========================================================
    # GROUP FILES BY EMOTION
    # ========================================================

    files_by_emotion = {
        emotion: []
        for emotion in EMOTIONS
    }

    for file_path in eeg_files:

        emotion = get_emotion_from_path(
            file_path
        )

        if emotion is not None:

            files_by_emotion[
                emotion
            ].append(
                file_path
            )


    # ========================================================
    # COLLECT WINDOWS
    # ========================================================

    windows_by_emotion = {}

    for emotion in EMOTIONS:

        windows_by_emotion[
            emotion
        ] = collect_windows_for_emotion(
            files_by_emotion[
                emotion
            ],
            emotion
        )


    # ========================================================
    # SHOW AVAILABLE WINDOWS
    # ========================================================

    print("\n\n")
    print("=" * 75)
    print(
        "AVAILABLE EEG WINDOWS"
    )
    print("=" * 75)

    for emotion in EMOTIONS:

        print(
            f"{emotion.capitalize():<10}: "
            f"{len(windows_by_emotion[emotion])}"
        )

    print("=" * 75)


    # ========================================================
    # SELECT EXACTLY 256 PER EMOTION
    # ========================================================

    selected_by_emotion = {}

    print("\n")
    print("=" * 75)
    print(
        "SELECTING DATASET WINDOWS"
    )
    print("=" * 75)

    for emotion in EMOTIONS:

        selected = select_windows(
            windows_by_emotion[
                emotion
            ],
            TARGET_IMAGES_PER_EMOTION
        )

        selected_by_emotion[
            emotion
        ] = selected

        print(
            f"{emotion.capitalize():<10}: "
            f"{len(selected)} selected"
        )

    training_trial_keys = load_training_trial_keys()
    global_db_min, global_db_max = compute_global_training_scale(
        selected_by_emotion,
        training_trial_keys
    )


    # ========================================================
    # GENERATE FINAL IMAGES
    # ========================================================

    final_counts = {}

    print("\n")
    print("=" * 75)
    print(
        "CREATING FINAL STFT DATASET"
    )
    print("=" * 75)

    for emotion in EMOTIONS:

        count = generate_images(
            selected_by_emotion[
                emotion
            ],
            emotion,
            global_db_min,
            global_db_max
        )

        final_counts[
            emotion
        ] = count


    # ========================================================
    # FINAL REPORT
    # ========================================================

    total_images = sum(
        final_counts.values()
    )

    print("\n\n")
    print("=" * 75)
    print(
        "STFT DATASET GENERATION COMPLETE"
    )
    print("=" * 75)

    print("\n")
    print(
        "IMAGES CREATED BY EMOTION"
    )

    print(
        "-" * 40
    )

    for emotion in EMOTIONS:

        print(
            f"{emotion.capitalize():<10}: "
            f"{final_counts[emotion]}"
        )

    print(
        "-" * 40
    )

    print(
        f"TOTAL IMAGES: "
        f"{total_images}"
    )

    print("\n")

    if total_images == TARGET_TOTAL_IMAGES:

        print(
            "SUCCESS:"
        )

        print(
            "EXACTLY 1024 UNIQUE STFT "
            "IMAGES WERE CREATED."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            f"Expected {TARGET_TOTAL_IMAGES} "
            f"images, but created "
            f"{total_images}."
        )

        print(
            "The available EEG recordings "
            "did not provide enough unique "
            "windows for the requested "
            "dataset size."
        )

    print("\n")
    print(
        "STFT DATASET LOCATION:"
    )

    print(
        OUTPUT_ROOT
    )

    print("\n")
    print(
        "Original EEG files were NOT modified."
    )

    print(
        "CNN training was NOT started."
    )

    print("=" * 75)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
