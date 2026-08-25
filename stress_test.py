"""
Stress test for Flask NIDS deployment.
Simulates thousands of requests to validate real-time performance.

Usage:
  python stress_test.py --url http://127.0.0.1:5000 --requests 5000 --workers 20
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def send_request(session, url, path, method="GET"):
    t0 = time.perf_counter()
    try:
        if method == "GET":
            r = session.get(f"{url}{path}", timeout=10)
        else:
            r = session.post(f"{url}{path}", json={"test": True}, timeout=10)
        ms = (time.perf_counter() - t0) * 1000
        return r.status_code, ms, None
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        return 0, ms, str(exc)


def run_stress_test(url, total_requests, workers, attack_ratio=0.3):
    normal_paths = ["/", "/api/data", "/api/status", "/home", "/about"]
    attack_paths = ["/login", "/admin", "/scan/port1", "/scan/port2",
                    "/scan/port3", "/../../etc/passwd", "/wp-admin"]

    print(f"\nStress Test — {total_requests:,} requests, {workers} workers")
    print(f"Target: {url}")
    print(f"Attack ratio: {attack_ratio*100:.0f}%\n")

    latencies = []
    status_counts = {}
    errors = 0
    blocked = 0

    session = requests.Session()
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for i in range(total_requests):
            is_attack = (i % 100) < (attack_ratio * 100)
            if is_attack:
                path = attack_paths[i % len(attack_paths)]
                method = "POST" if i % 3 == 0 else "GET"
            else:
                path = normal_paths[i % len(normal_paths)]
                method = "GET"
            futures.append(pool.submit(send_request, session, url, path, method))

        for fut in as_completed(futures):
            status, ms, err = fut.result()
            latencies.append(ms)
            status_counts[status] = status_counts.get(status, 0) + 1
            if err:
                errors += 1
            if status == 403:
                blocked += 1

    elapsed = time.perf_counter() - t_start
    rps = total_requests / elapsed

    print("=" * 55)
    print(f"  Total requests:  {total_requests:,}")
    print(f"  Duration:        {elapsed:.2f}s")
    print(f"  Throughput:      {rps:.0f} req/s")
    print(f"  Errors:          {errors}")
    print(f"  Blocked (403):     {blocked}")
    print(f"  Status codes:    {status_counts}")
    print()
    print(f"  Latency (ms):")
    print(f"    min:    {min(latencies):.1f}")
    print(f"    max:    {max(latencies):.1f}")
    print(f"    mean:   {statistics.mean(latencies):.1f}")
    print(f"    median: {statistics.median(latencies):.1f}")
    if len(latencies) > 1:
        print(f"    stdev:  {statistics.stdev(latencies):.1f}")
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"    p95:    {p95:.1f}")
    print(f"    p99:    {p99:.1f}")
    print("=" * 55)

    results = {
        "total_requests": total_requests,
        "duration_s": round(elapsed, 2),
        "throughput_rps": round(rps, 1),
        "errors": errors,
        "blocked": blocked,
        "latency_ms": {
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1),
        },
    }

    import json
    from pathlib import Path
    out = Path("results/stress_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="NIDS Stress Test")
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--attack-ratio", type=float, default=0.3)
    args = parser.parse_args()

    run_stress_test(args.url, args.requests, args.workers, args.attack_ratio)


if __name__ == "__main__":
    main()
