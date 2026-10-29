# EEG Emotion Recognition

EEG emotion-recognition training and inference code, with a browser-based presentation frontend. The frozen inference pipeline and its model checkpoints are in `src/` and `models/`; the UI is in `frontend/`.

## Run the local presentation app

The Python environment must include the project's inference dependencies. From the project root, start the LAN share server:

```bash
./.venv/bin/python src/run_lan_share.py
```

The terminal prints a local-network URL. The inference API stays on the host laptop, and the server forwards the frontend's `/health` and `/predict` requests to it. Keep the process running while the app is in use. See [`frontend/README.md`](frontend/README.md) for the separate local frontend and API commands.

## Data and local files

EEG recordings, prepared datasets, and generated run outputs are excluded from Git by `.gitignore`. Place any required data locally before running dataset preparation or training scripts.
