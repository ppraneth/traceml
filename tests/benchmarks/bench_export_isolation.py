# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Before/after benchmark for issue #222.

Measures the wall time a telemetry send costs on the sampler thread:
  direct   (before) -> TCPClient.send_batch runs inline on this thread
  exporter (after)  -> TelemetryExporter.send_batch only enqueues

Both paths use a real TCPClient (real msgpack encode + socket write); the only
difference is which thread does the network work. A slow aggregator is
simulated deterministically by injecting a fixed delay inside the client.

Run (deterministic: healthy + slow):
    python tests/benchmarks/bench_export_isolation.py

Realism check on Linux/cloud (adds an unreachable/blackhole aggregator):
    python tests/benchmarks/bench_export_isolation.py --real
"""

import argparse
import socket
import threading
import time

from traceml_ai.runtime.exporter import TelemetryExporter
from traceml_ai.transport.tcp_transport import TCPClient, TCPConfig


class _SinkServer:
    """Minimal local TCP sink that accepts and discards bytes."""

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        self._sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._drain, args=(conn,), daemon=True
            ).start()

    def _drain(self, conn):
        conn.settimeout(0.5)
        try:
            while not self._stop.is_set():
                try:
                    if not conn.recv(65536):
                        break
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass


class _DelayedTCPClient(TCPClient):
    """Real TCPClient plus a fixed post-send delay to mimic a slow aggregator."""

    def __init__(self, cfg, delay_sec=0.0):
        super().__init__(cfg)
        self._delay_sec = float(delay_sec)

    def send_batch(self, payloads):
        super().send_batch(payloads)
        if self._delay_sec:
            time.sleep(self._delay_sec)

    def send(self, payload):
        super().send(payload)
        if self._delay_sec:
            time.sleep(self._delay_sec)


def _build_batch(num_payloads, rows, cols):
    batch = []
    for p in range(num_payloads):
        table_rows = [
            {f"c{c}": p * 1000 + r for c in range(cols)} for r in range(rows)
        ]
        batch.append(
            {
                "meta": {"rank": 0, "sampler": f"s{p}", "timestamp": 1.0},
                "body": {"tables": {"t": table_rows}},
            }
        )
    return batch


def _measure(send_fn, batch, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        send_fn(batch)
        times.append(time.perf_counter() - t0)
    return times


def _stats_ms(times):
    s = sorted(times)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2]
    p95 = s[min(n - 1, int(0.95 * n))]
    p99 = s[min(n - 1, int(0.99 * n))]
    return {
        "mean": mean * 1e3,
        "median": median * 1e3,
        "p95": p95 * 1e3,
        "p99": p99 * 1e3,
        "max": s[-1] * 1e3,
        "total": sum(s) * 1e3,
    }


def _print_row(label, st):
    print(
        "  %-9s mean=%8.3f median=%8.3f p95=%8.3f p99=%8.3f "
        "max=%9.3f total=%10.3f"
        % (
            label,
            st["mean"],
            st["median"],
            st["p95"],
            st["p99"],
            st["max"],
            st["total"],
        )
    )


def _run_condition(name, cfg, delay_sec, n, args):
    batch = _build_batch(args.num_payloads, args.rows, args.cols)

    direct = _DelayedTCPClient(cfg, delay_sec)
    d_stats = _stats_ms(_measure(direct.send_batch, batch, n))
    direct.close()

    client = _DelayedTCPClient(cfg, delay_sec)
    exporter = TelemetryExporter(
        tcp_client=client,
        max_queue_size=args.queue_size,
        poll_interval_sec=0.01,
    )
    exporter.start()
    e_stats = _stats_ms(_measure(exporter.send_batch, batch, n))
    dropped = exporter.dropped_exports
    exporter.stop(2.0)

    speedup = d_stats["mean"] / e_stats["mean"] if e_stats["mean"] else 0.0
    print(f"\n[{name}]  publishes={n}  (times in ms)")
    _print_row("direct", d_stats)
    _print_row("exporter", e_stats)
    print(
        "  sampler-thread mean speedup: %.1fx   exporter dropped: %d"
        % (speedup, dropped)
    )


def benchmark_export_isolation(args):
    sink = _SinkServer()
    sink.start()

    healthy = TCPConfig(host="127.0.0.1", port=sink.port)
    _run_condition("healthy", healthy, 0.0, args.publishes, args)
    _run_condition(
        "slow %dms" % args.slow_ms,
        healthy,
        args.slow_ms / 1000.0,
        args.publishes,
        args,
    )

    if args.real:
        black = TCPConfig(host=args.blackhole_host, port=args.blackhole_port)
        n = min(args.publishes, args.unreachable_publishes)
        _run_condition("unreachable", black, 0.0, n, args)

    sink.stop()


def _parse_args():
    ap = argparse.ArgumentParser(description="TraceML export isolation bench")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--publishes", type=int, default=300)
    ap.add_argument("--slow-ms", type=int, default=25)
    ap.add_argument("--queue-size", type=int, default=2048)
    ap.add_argument("--num-payloads", type=int, default=4)
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--unreachable-publishes", type=int, default=5)
    ap.add_argument("--blackhole-host", default="10.255.255.1")
    ap.add_argument("--blackhole-port", type=int, default=29999)
    return ap.parse_args()


if __name__ == "__main__":
    benchmark_export_isolation(_parse_args())
