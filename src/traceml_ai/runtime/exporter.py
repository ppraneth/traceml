# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated telemetry exporter thread for the per-rank runtime agent.

Inserts a bounded handoff queue and a dedicated thread between payload
collection and network send, so TCP work never runs on the sampler thread:

    sampler thread -> export queue -> exporter thread -> TCP send

Presents the same send surface as TCPClient (send / send_batch / close) but
those calls only enqueue. Best-effort throughout: enqueue never blocks, and
shutdown does a bounded drain that never hangs.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from traceml_ai.loggers.error_log import get_error_logger

DEFAULT_EXPORT_QUEUE_SIZE = 2048
DEFAULT_EXPORT_DRAIN_TIMEOUT_SEC = 5.0
DEFAULT_EXPORT_POLL_INTERVAL_SEC = 0.2

_JOIN_MARGIN_SEC = 1.0
_DROP_LOG_EVERY = 100


@dataclass(frozen=True)
class _ExportItem:
    """One unit of exporter work; control frames use send, batches send_batch."""

    is_control: bool
    payload: Any


class TelemetryExporter:
    """Bounded, queue-backed async facade over a TCPClient."""

    def __init__(
        self,
        *,
        tcp_client: Any,
        logger: Optional[Any] = None,
        max_queue_size: int = DEFAULT_EXPORT_QUEUE_SIZE,
        drain_timeout_sec: float = DEFAULT_EXPORT_DRAIN_TIMEOUT_SEC,
        poll_interval_sec: float = DEFAULT_EXPORT_POLL_INTERVAL_SEC,
    ) -> None:
        self._tcp_client = tcp_client
        self._logger = logger or get_error_logger("TraceMLExporter")
        self._drain_timeout_sec = float(drain_timeout_sec)
        self._poll_interval_sec = float(poll_interval_sec)

        self._queue: "queue.Queue[_ExportItem]" = queue.Queue(
            maxsize=int(max_queue_size)
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._drop_lock = threading.Lock()
        self._dropped_exports = 0

        self._started = False
        self._stopped = False
        self._final_deadline = 0.0

    # Lifecycle

    def start(self) -> None:
        """Start the exporter thread. Idempotent."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            name="TraceMLExporter",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_sec: Optional[float] = None) -> None:
        """Signal shutdown, run a bounded final drain, close the client.

        Never blocks longer than the drain budget plus a small margin, never
        raises. Idempotent.
        """
        if self._stopped:
            return
        self._stopped = True

        budget = (
            self._drain_timeout_sec
            if timeout_sec is None
            else float(timeout_sec)
        )
        budget = max(0.0, budget)
        self._final_deadline = time.monotonic() + budget
        self._stop_event.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=budget + _JOIN_MARGIN_SEC)
            if thread.is_alive():
                self._logger.error(
                    "[TraceML] exporter thread did not terminate within "
                    "the shutdown drain timeout"
                )
                # Closing the socket unblocks a stuck send.
                self._close_tcp()
        else:
            self._close_tcp()

    def close(self) -> None:
        """Graceful bounded close, drop-in for TCPClient.close."""
        self.stop(self._drain_timeout_sec)

    # Producer API (sampler / stop thread)

    def send_batch(self, batch: list) -> None:
        """Enqueue a payload batch. Never raises."""
        if not batch:
            return
        self._enqueue(_ExportItem(is_control=False, payload=batch))

    def send(self, payload: dict) -> None:
        """Enqueue a single control payload. Never raises."""
        if payload is None:
            return
        self._enqueue(_ExportItem(is_control=True, payload=payload))

    @property
    def dropped_exports(self) -> int:
        """Items dropped so far because the queue was full."""
        return self._dropped_exports

    # Internals

    def _enqueue(self, item: _ExportItem) -> None:
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass

        # Full: drop the oldest queued item, enqueue the newest.
        with self._drop_lock:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._record_drop()
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._record_drop()

    def _record_drop(self) -> None:
        self._dropped_exports += 1
        if self._dropped_exports % _DROP_LOG_EVERY == 1:
            self._logger.error(
                "[TraceML] export queue full; dropped %d payload batches "
                "(queue size %d)",
                self._dropped_exports,
                self._queue.maxsize,
            )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=self._poll_interval_sec)
                except queue.Empty:
                    continue
                self._export_item(item)

            self._final_drain()
        except Exception as exc:
            self._logger.error("[TraceML] exporter thread crashed: %s", exc)
        finally:
            self._close_tcp()

    def _final_drain(self) -> None:
        while time.monotonic() < self._final_deadline:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            self._export_item(item)

        remaining = self._queue.qsize()
        if remaining:
            self._logger.error(
                "[TraceML] exporter shutdown drain timed out; %d items "
                "not sent",
                remaining,
            )

    def _export_item(self, item: _ExportItem) -> None:
        try:
            if item.is_control:
                self._tcp_client.send(item.payload)
            else:
                self._tcp_client.send_batch(item.payload)
        except Exception as exc:
            self._logger.error("[TraceML] exporter send failed: %s", exc)

    def _close_tcp(self) -> None:
        try:
            self._tcp_client.close()
        except Exception as exc:
            self._logger.error("[TraceML] exporter TCP close failed: %s", exc)


__all__ = [
    "TelemetryExporter",
    "DEFAULT_EXPORT_QUEUE_SIZE",
    "DEFAULT_EXPORT_DRAIN_TIMEOUT_SEC",
    "DEFAULT_EXPORT_POLL_INTERVAL_SEC",
]
