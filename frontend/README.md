# EEG Emotion Recognition Frontend

This is a lightweight static frontend for the existing frozen inference service. It uses plain HTML, CSS, and JavaScript and has no frontend package dependencies or build step.

## Start the application

From the project root, start the presentation backend in one terminal:

```bash
EEG_API_PORT=8001 ./.venv/bin/python src/run_presentation_api.py
```

This launcher permits only the known checksum mismatch in the standalone
`src/train_crnn.py` training script. It leaves the frozen configuration and
all checkpoints unchanged, and continues to verify the other implementation
sources and model files. Use `src/backend_api.py` directly when you want the
strict source-integrity check with no presentation exception.

In a second terminal, serve the frontend:

```bash
./.venv/bin/python -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Open <http://127.0.0.1:8080>. The frontend calls the existing `GET /health` and `POST /predict` endpoints at `http://127.0.0.1:8000`. The upload is sent as `multipart/form-data` using the existing `file` field. The backend continues to validate the MAT file, render STFT windows, and run inference.

The backend permits browser CORS requests only from `http://127.0.0.1:8080`; it does not allow arbitrary origins. If you serve the frontend from another origin, update the one explicit local origin in `src/backend_api.py` to match.

## Share on the same Wi-Fi

To let other devices on your local Wi-Fi use the app while it runs on this laptop, start the combined frontend/API share server from the project root:

```bash
./.venv/bin/python src/run_lan_share.py
```

Open the `http://<laptop-ip>:8080` link printed in the terminal on the other device. Keep the terminal open while people use it. The inference API itself stays bound to the laptop; only the frontend share server accepts connections from the local network.

## Verify

Run the frontend contract checks and backend API tests from the project root:

```bash
./.venv/bin/python src/test_frontend.py
./.venv/bin/python -m unittest src/test_backend_api.py
```

The complete frozen pipeline regression commands are documented in `models/fusion/FROZEN_ML_PIPELINE.md`. They verify the unchanged inference implementation and saved prediction parity.
