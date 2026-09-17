"""
SupportHR Remote Classifier Connection Verifier
----------------------------------------------
Tests whether a remote classifier server (on Colab, Kaggle, or local)
matches SupportHR's expected contract.

Usage:
    python test_connection.py --url https://<your-tunnel-url>.trycloudflare.com
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check_endpoint(name: str, url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    start = time.time()
    try:
        with urlopen(req, timeout=15) as resp:
            elapsed = time.time() - start
            body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            print(f" [PASS] {name} ({elapsed:.2f}s) -> HTTP {resp.status}")
            return parsed
    except HTTPError as e:
        elapsed = time.time() - start
        body = e.read().decode("utf-8", errors="replace")
        print(f" [FAIL] {name} ({elapsed:.2f}s) -> HTTP {e.code}: {body}")
        raise
    except URLError as e:
        print(f" [FAIL] {name} -> Network error: {e.reason}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SupportHR Classifier Service")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of remote server")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\n=======================================================")
    print(f"Verifying SupportHR Classifier Server: {base_url}")
    print(f"=======================================================\n")

    # 1. Health
    try:
        health_res = check_endpoint("Health Check", f"{base_url}/health")
        if not health_res.get("ready"):
            print("  Warning: Server returned ready=false in /health")
    except Exception:
        print("\n Aborting: Remote server is unreachable.")
        return 1

    # 2. Status
    try:
        status_res = check_endpoint("Classifier Status", f"{base_url}/api/cv/classifier-status")
        ready = status_res.get("ready")
        labels = status_res.get("labels", [])
        source = status_res.get("model_source", "unknown")
        print(f"    - Ready: {ready}")
        print(f"    - Model Source: {source}")
        print(f"    - Label Count: {len(labels)}")
        if not ready:
            return 1
    except Exception as e:
        print(f"\n Status check failed: {e}")
        return 1

    # 3. Classify
    sample_cv = (
        "Experienced Senior Backend Software Engineer with 6 years designing and building "
        "scalable RESTful APIs using Python, FastAPI, Docker, Kubernetes, and PostgreSQL."
    )
    try:
        classify_res = check_endpoint(
            "Classify Sample CV",
            f"{base_url}/api/cv/classify-industry",
            method="POST",
            payload={"cv_text": sample_cv, "top_k": 3},
        )
        predicted = classify_res.get("predicted_label")
        confidence = classify_res.get("confidence")
        top_preds = classify_res.get("top_predictions", [])
        print(f"\n Prediction Results:")
        print(f"    - Top Label: {predicted}")
        print(f"    - Confidence: {confidence}")
        print(f"    - Top Predictions: {top_preds}")

        if not predicted or not top_preds:
            return 1
    except Exception as e:
        print(f"\n Classification check failed: {e}")
        return 1

    print("\n=======================================================")
    print(" ALL CHECKS PASSED: Classifier Service is 100% compatible!")
    print("=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
