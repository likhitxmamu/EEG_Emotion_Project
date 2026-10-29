"""Render the frozen STFT PNG format for an unseen 62-channel recording.

This is separate from ``convert_to_spectrogram.py``: that script's current
output uses a shared training-set scale, while the frozen checkpoint inputs
use per-window percentile clipping and autoscaled Turbo rendering.
"""

import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import signal


# Avoid Matplotlib trying to write its cache into a read-only home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "eeg_emotion_matplotlib")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusion_engine import FROZEN_CONFIG


RAW_CONFIG = FROZEN_CONFIG["common_raw_eeg_preprocessing"]
STFT_CONFIG = FROZEN_CONFIG["stft_raster_provenance"][
    "stft_parameters_from_current_converter_and_fixedscale_provenance"
]
IMAGE_CONFIG = FROZEN_CONFIG["input_assets"]["stft_images"]
EXPECTED_CHANNELS = int(RAW_CONFIG["channels"])
SAMPLING_RATE_HZ = int(STFT_CONFIG["sampling_rate_hz"])
CHANNEL_INDEX = int(STFT_CONFIG["channel_index"])
WINDOW_SAMPLES = int(STFT_CONFIG["window_samples"])
HOP_SAMPLES = int(STFT_CONFIG["hop_samples"])
IMAGE_WIDTH, IMAGE_HEIGHT = (int(value) for value in IMAGE_CONFIG["image_dimensions_px"])
CLIP_PERCENTILES = tuple(
    float(value)
    for value in STFT_CONFIG["base_image_scale_documented_by_audit"]["percentiles"]
)
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class STFTRenderError(ValueError):
    """Raised when an input cannot be rendered with the frozen STFT recipe."""


def _validate_eeg(eeg):
    try:
        values = np.asarray(eeg)
    except Exception as exc:
        raise STFTRenderError(f"EEG input cannot be converted to an array: {exc}") from exc
    if values.dtype.kind not in "iuf" or np.iscomplexobj(values):
        raise STFTRenderError("EEG input must contain real numeric values")
    if values.ndim != 2 or values.shape[0] != EXPECTED_CHANNELS:
        raise STFTRenderError(
            f"Expected EEG shape ({EXPECTED_CHANNELS}, samples); received {values.shape}"
        )
    if values.shape[1] < WINDOW_SAMPLES:
        raise STFTRenderError(
            f"EEG recording must contain at least {WINDOW_SAMPLES} samples; received {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise STFTRenderError("EEG input contains NaN or infinite values")
    return values.astype(np.float64, copy=False)


def _clean_stft_channel(channel):
    """Match the original converter's whole-recording channel cleaning."""
    cleaned = np.asarray(channel, dtype=np.float64)
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = cleaned - np.mean(cleaned)
    low_hz, high_hz = (float(value) for value in RAW_CONFIG["bandpass_hz"])
    filter_order = int(RAW_CONFIG["filter_order"])
    nyquist = SAMPLING_RATE_HZ / 2.0
    b, a = signal.butter(
        filter_order,
        (low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
    )
    if cleaned.size > 30:
        cleaned = signal.filtfilt(b, a, cleaned)
    cleaned = signal.detrend(cleaned, type="linear")
    return cleaned


def _window_stft(window):
    frequencies, times, coefficients = signal.stft(
        window,
        fs=SAMPLING_RATE_HZ,
        window=STFT_CONFIG["window_function"],
        nperseg=int(STFT_CONFIG["nperseg"]),
        noverlap=int(STFT_CONFIG["noverlap"]),
        nfft=int(STFT_CONFIG["nfft"]),
        detrend=STFT_CONFIG["detrend"],
        boundary=STFT_CONFIG["boundary"],
        padded=bool(STFT_CONFIG["padded"]),
        scaling="spectrum",
    )
    power_db = 10.0 * np.log10(np.abs(coefficients) ** 2 + 1e-12)
    low_hz, high_hz = (float(value) for value in RAW_CONFIG["bandpass_hz"])
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return frequencies[mask], times, power_db[mask]


def _render_png(frequencies, times, power_db, output_path):
    low_clip, high_clip = np.percentile(power_db, CLIP_PERCENTILES)
    clipped_db = np.clip(power_db, low_clip, high_clip)

    # These settings reproduce the original frozen RGBA rasters. They match
    # src/test_stft.py and were checked pixel-for-pixel against the pinned
    # validation trial (36 of 36 selected windows).
    dpi = 100
    fig = plt.figure(
        figsize=(IMAGE_WIDTH / dpi, IMAGE_HEIGHT / dpi), dpi=dpi, frameon=False
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(
        times,
        frequencies,
        clipped_db,
        shading="gouraud",
        cmap="turbo",
        antialiased=True,
    )
    ax.set_axis_off()
    ax.set_xlim(times[0], times[-1])
    low_hz, high_hz = (float(value) for value in RAW_CONFIG["bandpass_hz"])
    ax.set_ylim(low_hz, high_hz)
    ax.margins(x=0, y=0)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor="black",
        edgecolor="none",
    )
    plt.close(fig)


def render_stft_pngs(eeg, output_directory, recording_id="recording", window_ids=None):
    """Write frozen-format STFT PNGs and return paths in numeric window order.

    By default, every complete 400-sample window at a 100-sample hop is
    rendered. ``window_ids`` may select existing one-based window numbers for
    regression checks. Offline class balancing selected 256 windows across
    all labeled source trials; that dataset-level selection is not applicable
    to an unlabeled single-recording inference request.
    """
    values = _validate_eeg(eeg)
    safe_stem = _SAFE_STEM_RE.sub("_", str(recording_id)).strip("._-")
    if not safe_stem:
        raise STFTRenderError("recording_id must contain at least one filename-safe character")

    sample_count = values.shape[1]
    total_windows = (sample_count - WINDOW_SAMPLES) // HOP_SAMPLES + 1
    if window_ids is None:
        selected_ids = list(range(1, total_windows + 1))
    else:
        try:
            selected_ids = [int(value) for value in window_ids]
        except (TypeError, ValueError) as exc:
            raise STFTRenderError("window_ids must be integer one-based window numbers") from exc
        if not selected_ids:
            raise STFTRenderError("At least one window ID must be selected")
        if len(set(selected_ids)) != len(selected_ids):
            raise STFTRenderError("window_ids must not contain duplicates")
        if any(value < 1 or value > total_windows for value in selected_ids):
            raise STFTRenderError(
                f"window_ids must be between 1 and {total_windows} for this recording"
            )
        selected_ids.sort()

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = _clean_stft_channel(values[CHANNEL_INDEX])
    outputs = []
    for window_id in selected_ids:
        start = (window_id - 1) * HOP_SAMPLES
        window = cleaned[start:start + WINDOW_SAMPLES]
        frequencies, times, power_db = _window_stft(window)
        output_path = output_dir / f"{safe_stem}_window_{window_id:04d}_STFT_RGB.png"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing STFT image: {output_path}")
        _render_png(frequencies, times, power_db, output_path)
        with Image.open(output_path) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT) or image.mode != IMAGE_CONFIG["image_mode"]:
                output_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Renderer produced unexpected image format: {image.size}/{image.mode}"
                )
        outputs.append(output_path)
    return outputs

