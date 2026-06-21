"""
core/loadtest.py
──────────────────
Real concurrency/throughput measurement against your OWN running server.
This is the "computational efficiency and scalability" evidence the problem
statement asks for — every number it prints is measured against the actual
running API, not estimated.

Usage
-----
    # Terminal 1: your server must already be running
    uvicorn main:app --port 8000

    # Terminal 2:
    python -m core.loadtest --image sample.jpg --concurrency 5 --requests 30
"""

from __future__ import annotations
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _single_request(url: str, image_bytes: bytes, filename: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, files={"file": (filename, image_bytes, "image/jpeg")}, timeout=60)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"ok": resp.status_code == 200, "status": resp.status_code, "elapsed_ms": elapsed_ms}
    except requests.exceptions.RequestException as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"ok": False, "status": None, "elapsed_ms": elapsed_ms, "error": str(e)}


def run_load_test(url: str, image_path: str, concurrency: int, total_requests: int) -> dict:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    filename = image_path.split("/")[-1]

    results = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_single_request, url, image_bytes, filename) for _ in range(total_requests)]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall_elapsed = time.perf_counter() - wall_start

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    latencies = sorted(r["elapsed_ms"] for r in successes)

    report = {
        "url": url,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_clock_seconds": round(wall_elapsed, 2),
        "successes": len(successes),
        "failures": len(failures),
        "throughput_req_per_sec": round(total_requests / wall_elapsed, 2) if wall_elapsed > 0 else 0,
    }

    if latencies:
        n = len(latencies)
        report["latency_ms"] = {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "p95": round(latencies[min(n - 1, int(0.95 * n))], 1),
            "p99": round(latencies[min(n - 1, int(0.99 * n))], 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        }
    if failures:
        report["sample_errors"] = [f.get("error", f"HTTP {f['status']}") for f in failures[:5]]

    return report


def _main():
    parser = argparse.ArgumentParser(description="Real concurrent load test against a running VIGIL server.")
    parser.add_argument("--image", required=True, help="Path to a sample JPG/PNG to repeatedly upload")
    parser.add_argument("--url", default="http://127.0.0.1:8000/analyze/image")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of simultaneous requests in flight")
    parser.add_argument("--requests", type=int, default=30, help="Total number of requests to send")
    parser.add_argument("--report-out", default="loadtest_report.json")
    args = parser.parse_args()

    print(f"Running {args.requests} requests at concurrency {args.concurrency} against {args.url} ...")
    report = run_load_test(args.url, args.image, args.concurrency, args.requests)
    print(json.dumps(report, indent=2))

    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.report_out}")

    if report["failures"] > 0:
        print(f"\n⚠ {report['failures']} of {args.requests} requests failed — server may be saturated at this concurrency level.")


if __name__ == "__main__":
    _main()
