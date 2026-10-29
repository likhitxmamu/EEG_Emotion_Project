# Frozen EEG Emotion Inference Pipeline

The single source of truth is [`frozen_ml_config.json`](frozen_ml_config.json). It freezes the existing four-model inference and adaptive-fusion behavior from `adaptive_run_20261027_132228`. No model was retrained and no architecture or experimental setting was changed.

## Included models and checkpoints

| Model | Frozen checkpoint | Validation-selected epoch |
|---|---|---:|
| EEGNet + GCN | `models/eegnet_gcn/run_20261027_122050/best_eegnet_gcn.pth` | 11 |
| Raw EEG Transformer | `models/raw_eeg_transformer/run_20261027_123703/best_raw_eeg_transformer.pth` | 21 |
| STFT CRNN | `models/best_crnn_model.pth` | 10 |
| STFT Transformer | `models/best_transformer_model.pth` | Not recorded in the bare state-dict checkpoint |

The configuration records SHA-256 hashes for each checkpoint. The GCN also uses `models/eegnet_gcn/run_20261027_122050/normalization_and_graph.json`, whose hash is pinned in the configuration.

## Frozen inputs and preprocessing

The class order is `fear, happy, neutral, sad`. Fixed trial assignments for train, validation, and test are listed in the configuration and checked against both raw-model assignment definitions.

Both raw EEG models use the 62-channel `EEG` array from MAT files. The existing pipeline assumes 200 Hz because the MAT files do not contain a sampling-rate field. It replaces nonfinite values with zero, removes each channel's full-trial mean, applies the 4th-order 1–45 Hz zero-phase Butterworth bandpass, linearly detrends, and makes 400-sample windows every 100 samples. This is an explicit project assumption, not an independently verified dataset property.

EEGNet + GCN uses the saved training-only channel normalization and functional adjacency graph. Raw EEG Transformer uses the frozen training-only channel means and standard deviations recorded in the configuration; inference verifies these against the fixed training trials.

The two image models consume the frozen STFT raster format. The saved `data/processed/crnn_dataset` corpus is pinned by per-split SHA-256 manifests. CRNN inference converts to RGB, resizes to 224×224, converts to a tensor, and applies ImageNet mean/std normalization. STFT Transformer inference converts to RGB, resizes to 224×224, converts to grayscale, converts to a tensor, and normalizes with mean/std 0.5.

For new recordings, `src/frozen_stft_renderer.py` regenerates that raster format from channel 0. It uses the frozen provenance recipe and reproduces the 36 saved validation PNGs for `1_20160518_trial17` pixel-for-pixel. The current `src/convert_to_spectrogram.py` writes the separate global-scale experiment under `data/processed/stft_spectrograms_global`; that output is excluded from this freeze and must not replace the frozen per-window rendering recipe. The immutable PNG manifest remains the input authority for the saved evaluation.

## Prediction and fusion

Each model applies softmax per window. Its trial probability is the mean of its window probability vectors, normalized to sum to one. The trial class is the maximum-probability class in the frozen order.

The adaptive fusion prior uses each model's macro-F1 on the four fixed validation trials:

| Model | Validation macro-F1 | Frozen prior weight |
|---|---:|---:|
| EEGNet + GCN | 0.3333333333 | 0.2371134021 |
| Raw EEG Transformer | 0.3750000000 | 0.2628865979 |
| STFT CRNN | 0.3333333333 | 0.2371134021 |
| STFT Transformer | 0.3750000000 | 0.2628865979 |

For each model `i` and trial `t`, normalized entropy is `H(p_i,t) / ln(4)`, where `H(p) = -sum_c p_c ln(max(p_c, 1e-15))`. Certainty is `clip(1 - normalized_entropy, 0, 1)`. The gate is `0.05 + certainty`; its product with the frozen prior is normalized across models for that trial. The final probability is the weighted sum of the four trial vectors, normalized once more. The equal-weight baseline uses 0.25 for each model.

The validation-derived reliability scores and weights are frozen values. Test labels and results were not used to derive or alter them. Test results are reference-only and must never be used to modify this configuration.

## How future inference code must use this freeze

Use `models/fusion/frozen_ml_config.json` for paths, class order, trial assignments, preprocessing values, checkpoint selection, and fusion. `src/fusion_engine.py` and `src/adaptive_fusion.py` already load it. Inference verifies the selected checkpoints and GCN metadata, pinned implementation source files, and the input manifest for each split before prediction. Keep the four branches, their class order, preprocessing, trial aggregation, and fusion rule unchanged under this configuration.

Run the requested parity check with:

```bash
./.venv/bin/python src/validate_frozen_pipeline.py
```

This check reads the held-out test inputs only to compare predictions with the already-saved test evaluation. It uses the frozen validation weights and never fits or changes configuration from test results.

## Frozen evaluation reference

| Split | Fusion | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Validation | Equal-weight | 0.7500 | 0.6667 |
| Validation | Adaptive | 0.7500 | 0.6667 |
| Test | Equal-weight | 0.5000 | 0.3750 |
| Test | Adaptive | 0.5000 | 0.3750 |

The test set contains four trials, one per class. Its scores are descriptive and are not tuning signals.

## Python inference service

`src/inference_service.py` provides the supported Python entry point around the frozen configuration and current model-loading, preprocessing, aggregation, and adaptive-fusion functions. It loads the four pinned checkpoints on first prediction and reuses them for later calls in the same service instance. It does not read labels or fit reliability values.

### Input

Call `InferenceService.predict(request)` with a mapping containing:

- Exactly one of `eeg` or `mat_path`.
- `eeg`: a real numeric array with shape `62 x samples`, at least 400 samples long, using the frozen 200 Hz assumption; or `mat_path`: a `.mat` file containing an `EEG` variable with that shape. Any MAT `label` value is ignored.
- `stft_image_paths` is optional. When omitted, the service automatically generates every complete STFT window in a safe temporary directory using the frozen renderer. It is available for saved-reference regression checks that need the exact pinned window subset.
- Optional `recording_id`: a short identifier for logs and the response.

The frozen raw branches preprocess the supplied EEG using the values above. The STFT renderer uses channel 0, removes its full-recording mean, applies the order-4 1–45 Hz zero-phase Butterworth filter when the recording exceeds 30 samples, and linearly detrends it. It creates 400-sample windows every 100 samples. Within each window it computes a Hann STFT at 200 Hz with `nperseg=256`, `noverlap=192`, `nfft=512`, constant detrending, no boundary extension, and no padding. The STFT uses SciPy's default spectrum scaling, converts power to `10*log10(abs(STFT)**2 + 1e-12)`, retains 1–45 Hz bins, clips each image to that window's 5th–98th percentile, and renders a 1024×1024 RGBA Turbo raster with automatic per-image color scaling and no axes or annotations. The service sorts generated windows by numeric ID and removes its temporary PNGs after inference. The saved evaluation uses a dataset-level selected subset, so its STFT window count may differ from the all-window inference count.

The two STFT branches use the existing frozen image transforms and average their own per-window outputs. Their image-window count may differ from the raw EEG-window count.

Example:

```python
from src.inference_service import InferenceService

service = InferenceService()
result = service.predict({
    "recording_id": "session_001",
    "mat_path": "recording.mat",
})
if not result["ok"]:
    raise RuntimeError(result["error"])
print(result["prediction"], result["probabilities"])
```

The response is JSON-serializable. A successful response has `ok`, `configuration_id`, `recording_id`, `class_order`, `prediction`, `confidence`, `adaptive_confidence_score`, `confidence_description`, `probabilities`, `models`, `adaptive_fusion`, and `input` fields. The class order is `fear, happy, neutral, sad`. `probabilities` is a class-to-probability mapping in that order. Each model entry reports its trial-level prediction, probability map, window count, and per-window logits/probabilities. `adaptive_fusion` includes the fixed reliability prior, that recording's normalized adaptive weights, entropy, and certainty gates. The confidence fields explicitly identify the frozen adaptive confidence score as not being a calibrated probability. Failures return `ok: false` with a machine-readable `error.code` and `error.message` instead of a prediction.

### Regression check

Run the focused service check against the existing validation result for the known trial `1_20160518_trial17`:

```bash
./.venv/bin/python src/check_inference_service.py
```

It asks the service to render the validation recording's saved window IDs automatically, then compares all four model predictions, per-model probability maps and window counts, adaptive prediction, confidence, fused probabilities, and adaptive weights against `adaptive_run_20261027_132228/validation_predictions.csv`, using absolute tolerance `1e-5` for numeric values. It does not pass PNG paths to the service. The sample is from validation; the check does not load test labels or use test results to configure inference.

Run the renderer parity and malformed-input checks with:

```bash
./.venv/bin/python src/test_frozen_stft_renderer.py
./.venv/bin/python src/test_inference_service.py
```

## HTTP backend API

`src/backend_api.py` provides a dependency-free WSGI wrapper around `InferenceService`. It does not duplicate preprocessing, STFT rendering, model execution, prediction aggregation, or fusion. The API uses Python's standard-library WSGI server and has no FastAPI or Flask dependency.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Accept one MATLAB EEG upload and run the existing inference service. |
| `GET` | `/health` | Return process health without loading model checkpoints or running inference. |

### Prediction request

Send `multipart/form-data` with one file part named `file`. The filename must end in `.mat`; the file must contain a real `EEG` variable shaped `62 x samples` and at least 400 samples. A `label` variable is not required. The current request-size limit is 100 MiB, including multipart overhead. Uploads use the existing SciPy MAT reader; MATLAB v7.3/HDF5 files are not verified by this pipeline and may be rejected as unreadable.

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F 'file=@recording.mat'
```

The backend writes the upload to a private temporary directory and passes only its path to `InferenceService`. The service generates the frozen STFT windows automatically; the client does not supply STFT PNGs or preprocessing details. Temporary upload and rendered image files are removed after the request.

### Successful response

The response is JSON and includes the final prediction, confidence, class probabilities, per-model results, window counts, and adaptive-fusion weights and gates. Confidence remains the frozen adaptive confidence score and is explicitly described as not being a calibrated probability. The API omits temporary paths, per-window logits, and internal configuration details. The example below abbreviates the populated per-model entries and fusion maps; the live response contains all four model entries.

```json
{
  "ok": true,
  "prediction": "fear",
  "confidence": 0.42,
  "adaptive_confidence_score": 0.42,
  "confidence_type": "frozen_adaptive_confidence_score",
  "confidence_description": "Frozen adaptive confidence score (the winning fused probability value); this is not a calibrated probability.",
  "class_order": ["fear", "happy", "neutral", "sad"],
  "class_probabilities": {
    "fear": 0.42,
    "happy": 0.18,
    "neutral": 0.21,
    "sad": 0.19
  },
  "models": {},
  "window_counts": {
    "by_model": {},
    "raw": 133,
    "stft": 133
  },
  "adaptive_fusion": {
    "reliability_prior_weights": {},
    "weights": {},
    "normalized_entropy_by_model": {},
    "certainty_gate_by_model": {}
  }
}
```

### Errors and health

All errors use the same JSON shape. Public messages are allow-listed; implementation exceptions and filesystem paths are logged server-side and are never returned to the client.

```json
{
  "ok": false,
  "error": {
    "code": "missing_eeg_variable",
    "message": "The MATLAB file must contain an EEG variable."
  }
}
```

`GET /health` returns `{"ok":true,"status":"ok"}`. The route does not load checkpoints or call inference.

### Start and test

Start the local development server from the project root:

```bash
./.venv/bin/python src/backend_api.py
```

It binds to `127.0.0.1:8000` by default. Set `EEG_API_HOST` and `EEG_API_PORT` to change the bind address. This standard-library server is for local development; use a production WSGI server for deployment. CORS is allowed only for the local static frontend origin `http://127.0.0.1:8080`; other origins receive no CORS permission. If the frontend is served from a different origin, update the explicit allowed origin in `src/backend_api.py` to that development origin.

Run API contract tests with:

```bash
./.venv/bin/python src/test_backend_api.py
```

The API validation test uploads the saved validation recording, re-renders its frozen validation window IDs, and compares public probabilities, predictions, confidence, window counts, and fusion weights with the saved validation predictions. The wrapper only selects and renames response fields; it does not recalculate them.

The HTTP layer is additive. Frozen checkpoints, architectures, preprocessing values, class order, trial assignments, prediction aggregation, and adaptive-fusion behavior remain unchanged.
