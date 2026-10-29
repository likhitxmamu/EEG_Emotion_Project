"""Serve the EEG presentation app to devices on the same local network.

The inference API stays bound to loopback. This server serves the frontend and
proxies only /health and /predict to the local API, so browsers on other
devices use the laptop's address for both the page and API requests.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
API_URL = "http://127.0.0.1:8001"
API_ROUTES = {"/health", "/predict"}
MAX_PROXY_BODY_BYTES = 105 * 1024 * 1024


def _local_address() -> str | None:
    """Return the preferred outbound IPv4 address without sending traffic."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))
        address = probe.getsockname()[0]
        return address if not address.startswith("127.") else None
    except OSError:
        return None
    finally:
        probe.close()


class ShareHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in API_ROUTES:
            self._proxy_api()
        elif path == "/brain-signal-loop.mp4":
            self._serve_brain_video()
        elif path in ("/", "/index.html"):
            self._serve_shared_index()
        else:
            super().do_GET()

    def do_HEAD(self):
        if urlsplit(self.path).path == "/brain-signal-loop.mp4":
            video_path = FRONTEND / "brain-signal-loop.mp4"
            try:
                size = video_path.stat().st_size
            except OSError:
                self.send_error(404, "Video not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return
        super().do_HEAD()

    def do_POST(self):
        if urlsplit(self.path).path in API_ROUTES:
            self._proxy_api()
        else:
            self.send_error(404, "Not found")

    def _serve_shared_index(self):
        page = (FRONTEND / "index.html").read_text(encoding="utf-8")
        page = page.replace(
            'window.EEG_API_BASE_URL = "http://127.0.0.1:8001";',
            "window.EEG_API_BASE_URL = window.location.origin;",
            1,
        )
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_brain_video(self):
        video_path = FRONTEND / "brain-signal-loop.mp4"
        try:
            size = video_path.stat().st_size
        except OSError:
            self.send_error(404, "Video not found")
            return

        requested_range = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if requested_range:
            try:
                unit, values = requested_range.strip().split("=", 1)
                if unit != "bytes" or "," in values:
                    raise ValueError
                first, last = values.split("-", 1)
                if not first:
                    suffix_length = int(last)
                    if suffix_length <= 0:
                        raise ValueError
                    start = max(0, size - suffix_length)
                else:
                    start = int(first)
                    end = int(last) if last else size - 1
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
                status = 206
            except (ValueError, TypeError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with video_path.open("rb") as video:
            video.seek(start)
            remaining = length
            while remaining:
                chunk = video.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _proxy_api(self):
        path = urlsplit(self.path).path
        body = None
        headers = {}
        if self.command == "POST":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid request length")
                return
            if length <= 0 or length > MAX_PROXY_BODY_BYTES:
                self.send_error(413, "Request body is too large")
                return
            body = self.rfile.read(length)
            if len(body) != length:
                self.send_error(400, "Incomplete request body")
                return
            content_type = self.headers.get("Content-Type")
            if content_type:
                headers["Content-Type"] = content_type

        request = urllib.request.Request(
            API_URL + path,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            response = error
        except (urllib.error.URLError, TimeoutError) as error:
            self.send_error(502, f"Local inference API unavailable: {error}")
            return

        with response:
            response_body = response.read()
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in {"connection", "transfer-encoding", "date", "server"}:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)


def _wait_for_api(process: subprocess.Popen, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = API_URL + "/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Inference API exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    json.loads(response.read())
                    return
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            time.sleep(0.5)
    raise TimeoutError(f"Inference API did not become ready at {health_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Share server bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8080, help="Share server port (default: 8080)")
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")

    env = os.environ.copy()
    env["EEG_API_HOST"] = "127.0.0.1"
    env["EEG_API_PORT"] = "8001"
    api_process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_presentation_api.py")],
        cwd=PROJECT_ROOT,
        env=env,
    )
    server = None
    try:
        _wait_for_api(api_process)
        server = http.server.ThreadingHTTPServer((args.host, args.port), ShareHandler)
        server.daemon_threads = True
        ip_address = _local_address()
        print("EEG app is being served from this laptop.", flush=True)
        if ip_address:
            print(f"Share this link with people on the same Wi-Fi: http://{ip_address}:{args.port}", flush=True)
        else:
            print(f"Server listening on {args.host}:{args.port}; find this laptop's LAN IP to share the link.", flush=True)
        print("Keep this terminal open while others use the app. Press Ctrl+C to stop.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()
            api_process.wait()


if __name__ == "__main__":
    main()
