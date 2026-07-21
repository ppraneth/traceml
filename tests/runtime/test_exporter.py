# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from traceml_ai.runtime.exporter import TelemetryExporter
from traceml_ai.runtime.sender import SenderIdentity, TelemetryPublisher


class _FakeTCP:
    """Records send calls; can fail or block to simulate a bad aggregator."""

    def __init__(self, *, fail=False, block_event=None):
        self._fail = fail
        self._block_event = block_event
        self._lock = threading.Lock()
        self.sent_batches = []
        self.sent_controls = []
        self.closed = False

    def send_batch(self, batch):
        if self._block_event is not None:
            self._block_event.wait()
        if self._fail:
            raise RuntimeError("send_batch failed")
        with self._lock:
            self.sent_batches.append(batch)

    def send(self, payload):
        if self._fail:
            raise RuntimeError("send failed")
        with self._lock:
            self.sent_controls.append(payload)

    def close(self):
        self.closed = True


class _FakeSender:
    def __init__(self, payload):
        self._payload = payload
        self.sender = None
        self.identity = None

    def collect_payload(self):
        return self._payload


class _FakeSampler:
    def __init__(self, name, payload):
        self.sampler_name = name
        self.sender = _FakeSender(payload)


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_start_enqueue_and_export_delivers_batch():
    tcp = _FakeTCP()
    exporter = TelemetryExporter(tcp_client=tcp, poll_interval_sec=0.01)
    exporter.start()
    try:
        exporter.send_batch([{"a": 1}])
        assert _wait_until(lambda: tcp.sent_batches)
        assert tcp.sent_batches == [[{"a": 1}]]
    finally:
        exporter.stop(1.0)
    assert tcp.closed is True


def test_control_payload_uses_send_not_send_batch():
    tcp = _FakeTCP()
    exporter = TelemetryExporter(tcp_client=tcp, poll_interval_sec=0.01)
    exporter.start()
    try:
        exporter.send({"ctrl": 1})
        assert _wait_until(lambda: tcp.sent_controls)
        assert tcp.sent_controls == [{"ctrl": 1}]
        assert tcp.sent_batches == []
    finally:
        exporter.stop(1.0)


def test_empty_batch_is_not_enqueued():
    tcp = _FakeTCP()
    exporter = TelemetryExporter(tcp_client=tcp, max_queue_size=2)
    exporter.send_batch([])
    assert exporter._queue.qsize() == 0
    assert exporter.dropped_exports == 0


def test_full_queue_drops_oldest_and_counts():
    tcp = _FakeTCP()
    # Do not start the thread so nothing drains and the queue stays full.
    exporter = TelemetryExporter(tcp_client=tcp, max_queue_size=2)

    exporter.send_batch([1])
    exporter.send_batch([2])
    exporter.send_batch([3])

    assert exporter.dropped_exports == 1
    remaining = [
        exporter._queue.get_nowait().payload,
        exporter._queue.get_nowait().payload,
    ]
    assert remaining == [[2], [3]]


def test_aggregator_unavailable_keeps_thread_alive():
    tcp = _FakeTCP(fail=True)
    exporter = TelemetryExporter(tcp_client=tcp, poll_interval_sec=0.01)
    exporter.start()
    try:
        for i in range(5):
            exporter.send_batch([i])
        # Thread must survive send failures and keep accepting work.
        assert _wait_until(lambda: exporter._queue.empty())
        assert exporter._thread.is_alive()
        exporter.send_batch([99])
        time.sleep(0.05)
        assert exporter._thread.is_alive()
    finally:
        exporter.stop(1.0)
    assert tcp.closed is True


def test_shutdown_drain_timeout_does_not_hang():
    block = threading.Event()
    tcp = _FakeTCP(block_event=block)
    exporter = TelemetryExporter(tcp_client=tcp, poll_interval_sec=0.01)
    exporter.start()
    try:
        exporter.send_batch([1])
        # Wait for the thread to pick up the item and block inside send_batch.
        assert _wait_until(lambda: exporter._queue.empty())

        start = time.perf_counter()
        exporter.stop(timeout_sec=0.2)
        elapsed = time.perf_counter() - start

        # Bounded by drain budget + join margin, not an unbounded hang.
        assert elapsed < 3.0
    finally:
        block.set()
    assert tcp.closed is True


def test_publish_tick_does_not_send_directly_until_drained():
    tcp = _FakeTCP()
    exporter = TelemetryExporter(tcp_client=tcp, poll_interval_sec=0.01)
    publisher = TelemetryPublisher(
        tcp_client=exporter,
        identity=SenderIdentity(global_rank=0, local_rank=0),
    )
    sampler = _FakeSampler("SamplerA", payload={"rows": [1]})

    # Exporter not started: publish enqueues but performs no TCP send.
    publisher.publish([sampler])
    assert tcp.sent_batches == []
    assert exporter._queue.qsize() == 1

    exporter.start()
    try:
        assert _wait_until(lambda: tcp.sent_batches)
        assert tcp.sent_batches == [[{"rows": [1]}]]
    finally:
        exporter.stop(1.0)
