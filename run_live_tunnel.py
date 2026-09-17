"""
SupportHR Live Classifier Service Runner
---------------------------------------
Starts the Classifier FastAPI server on port 8123 and establishes
an instant, public Cloudflare HTTPS Tunnel.

Keeps running to serve requests and auto-updates cv-match-api/api_server/.env.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from pycloudflared import try_cloudflare

PORT = 8123
SERVICE_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = SERVICE_DIR / "server.py"
API_SERVER_DIR = SERVICE_DIR.parent / "cv-match-api" / "api_server"
ENV_FILE = API_SERVER_DIR / ".env"
TUNNEL_FILE = SERVICE_DIR / "tunnel_url.txt"
LOG_FILE = SERVICE_DIR / "server.log"


def update_env_file(tunnel_url: str) -> None:
    classify_url = f"{tunnel_url}/api/cv/classify-industry"
    status_url = f"{tunnel_url}/api/cv/classifier-status"

    if not ENV_FILE.exists():
        example = API_SERVER_DIR / ".env.example"
        if example.exists():
            ENV_FILE.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            ENV_FILE.write_text("", encoding="utf-8")

    content = ENV_FILE.read_text(encoding="utf-8")
    updates = {
        "LOCAL_CLASSIFIER_MODE": "auto",
        "LOCAL_CLASSIFIER_REMOTE_CLASSIFY_URL": classify_url,
        "LOCAL_CLASSIFIER_REMOTE_STATUS_URL": status_url,
        "LOCAL_CLASSIFIER_REMOTE_TIMEOUT_SECONDS": "10.0",
    }

    lines = content.splitlines()
    existing_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                existing_keys.add(k)
                continue
        new_lines.append(line)

    for k, v in updates.items():
        if k not in existing_keys:
            new_lines.append(f"{k}={v}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[OK] Updated cv-match-api/api_server/.env with live remote endpoints!")


def main() -> None:
    print(f"=======================================================")
    print(f" Starting SupportHR Classifier Service on Port {PORT}")
    print(f"=======================================================")

    log_handle = LOG_FILE.open("w", encoding="utf-8")

    # 1. Start FastAPI server with output redirected to log file (non-blocking)
    server_proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "--port", str(PORT), "--model-source", "cloud://support-hr-live-model"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("Waiting for server to initialize model...")
    time.sleep(3)

    # 2. Establish Cloudflare Tunnel
    print("Starting Cloudflare Tunnel...")
    tunnel = try_cloudflare(port=PORT)
    public_url = tunnel.tunnel.rstrip("/")

    TUNNEL_FILE.write_text(public_url, encoding="utf-8")

    print("\n" + "=" * 65)
    print(" LIVE PUBLIC API READY!")
    print(f" Public HTTPS Base URL: {public_url}")
    print(f" Classify Endpoint:     {public_url}/api/cv/classify-industry")
    print(f" Status Endpoint:       {public_url}/api/cv/classifier-status")
    print("=" * 65 + "\n")

    # 3. Auto-update .env
    update_env_file(public_url)

    # 4. Keep running
    try:
        while True:
            time.sleep(1)
            if server_proc.poll() is not None:
                print("Server exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nShutting down live tunnel and server...")
    finally:
        server_proc.terminate()
        log_handle.close()


if __name__ == "__main__":
    main()
