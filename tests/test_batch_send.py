"""
test_batch_send.py
==================
Tests for the TCP batch-send optimisation (Perf #2).

What this tests
---------------
1. ``DBIncrementalSender.collect_payload()``   \u2014 collect without sending
2. ``TCPClient.send_batch()``                  \u2014 one sendall() per batch
3. ``RemoteDBStore.ingest()``                  \u2014 handles both list and dict
4. Before/after syscall-count comparison       \u2014 the measurable gain

Run with:
    pytest tests/test_batch_send.py -v
"""

import struct
import threading
from collections import deque
from unittest.mock import MagicMock, call, patch

import msgspec
import pytest


def _make_db_with_rows(n_rows: int):
    """Return a Database with n_rows pre-inserted into 'TableA'."""
    from traceml.database.database import Database

    db = Database(sampler_name="TestSampler")
    for i in range(n_rows):
        db.add_record("TableA", {"step": i, "val": float(i)})
    return db


def _make_sender(db, n_rows: int = 3):
    """Return a DBIncrementalSender wired to *db*, with a mock transport."""
    from traceml.database.database_sender import DBIncrementalSender

    mock_transport = MagicMock()
    return DBIncrementalSender(
        db=db,
        sampler_name="TestSampler",
        sender=mock_transport,
        rank=0,
    )


# 1. DBIncrementalSender.collect_payload()


class TestCollectPayload:

    def test_returns_none_when_db_is_empty(self):
        """collect_payload() returns None when there is nothing to send."""
        from traceml.database.database import Database
        from traceml.database.database_sender import DBIncrementalSender

        db = Database(sampler_name="EmptySampler")
        sender = DBIncrementalSender(
            db=db, sampler_name="EmptySampler", rank=0
        )
        assert sender.collect_payload() is None

    def test_returns_dict_with_new_rows(self):
        """collect_payload() returns a dict containing the new rows."""
        db = _make_db_with_rows(3)
        sender = _make_sender(db)

        payload = sender.collect_payload()

        assert payload is not None
        assert payload["rank"] == 0
        assert payload["sampler"] == "TestSampler"
        assert "tables" in payload
        assert "TableA" in payload["tables"]
        assert len(payload["tables"]["TableA"]) == 3

    def test_second_call_returns_none_no_new_rows(self):
        """
        After collect_payload() advances the cursor, a second call with no new
        data returns None — ensuring rows are not re-sent.
        """
        db = _make_db_with_rows(3)
        sender = _make_sender(db)

        first = sender.collect_payload()
        second = sender.collect_payload()

        assert first is not None
        assert second is None

    def test_new_rows_after_first_collect(self):
        """Only the rows added after the first call appear in the second call."""
        db = _make_db_with_rows(2)
        sender = _make_sender(db)

        sender.collect_payload()  # drain existing rows

        db.add_record("TableA", {"step": 99, "val": 99.0})
        second = sender.collect_payload()

        assert second is not None
        rows = second["tables"]["TableA"]
        assert len(rows) == 1
        assert rows[0]["step"] == 99

    def test_does_not_call_transport_send(self):
        """collect_payload() must NOT call the transport — that's the caller's job."""
        db = _make_db_with_rows(3)
        sender = _make_sender(db)

        sender.collect_payload()

        sender.sender.send.assert_not_called()

    def test_flush_still_works_as_wrapper(self):
        """flush() is a backward-compatible wrapper: it calls transport.send()."""
        db = _make_db_with_rows(2)
        sender = _make_sender(db)

        sender.flush()

        sender.sender.send.assert_called_once()
        call_kwargs = sender.sender.send.call_args[0][0]
        assert "tables" in call_kwargs


# 2. TCPClient.send_batch()


class TestSendBatch:

    def _make_connected_client(self, mock_sock):
        """Return a TCPClient with a pre-connected mock socket."""
        from traceml.transport.tcp_transport import TCPClient, TCPConfig

        client = TCPClient(TCPConfig())
        client._sock = mock_sock
        client._connected = True
        return client

    def test_send_batch_calls_sendall_exactly_once(self):
        """
        BEFORE: N payloads → N sendall() calls.
        AFTER:  N payloads → 1 sendall() call.
        This is the core assertion of the optimisation.
        """
        mock_sock = MagicMock()
        client = self._make_connected_client(mock_sock)

        payloads = [
            {"rank": 0, "sampler": f"Sampler{i}", "tables": {}}
            for i in range(10)
        ]

        # --- BEFORE behaviour: 10 individual send() calls ---
        before_client = self._make_connected_client(MagicMock())
        for p in payloads:
            before_client.send(p)
        before_sendall_count = before_client._sock.sendall.call_count
        assert (
            before_sendall_count == 10
        ), "Sanity check: old path should call sendall 10 times"

        # --- AFTER behaviour: 1 send_batch() call ---
        after_client = self._make_connected_client(MagicMock())
        after_client.send_batch(payloads)
        after_sendall_count = after_client._sock.sendall.call_count
        assert (
            after_sendall_count == 1
        ), "New path must call sendall exactly once"

    def test_send_batch_empty_list_is_noop(self):
        """send_batch([]) must not touch the socket at all."""
        mock_sock = MagicMock()
        client = self._make_connected_client(mock_sock)

        client.send_batch([])

        mock_sock.sendall.assert_not_called()

    def test_send_batch_encodes_as_list(self):
        """The wire payload must be decodable as a list of dicts."""
        mock_sock = MagicMock()
        client = self._make_connected_client(mock_sock)

        payloads = [
            {"rank": 0, "sampler": "A", "tables": {"t": [{"step": 1}]}},
            {"rank": 1, "sampler": "B", "tables": {"t": [{"step": 2}]}},
        ]
        client.send_batch(payloads)

        # Recover the raw bytes that were written
        raw = mock_sock.sendall.call_args[0][0]
        # Strip 4-byte length prefix
        length = struct.unpack("!I", raw[:4])[0]
        body = raw[4 : 4 + length]

        decoded = msgspec.msgpack.decode(body)
        assert isinstance(decoded, list)
        assert len(decoded) == 2
        assert decoded[0]["sampler"] == "A"
        assert decoded[1]["sampler"] == "B"

    def test_send_batch_single_payload_still_works(self):
        """send_batch() with one item still results in exactly one sendall()."""
        mock_sock = MagicMock()
        client = self._make_connected_client(mock_sock)

        client.send_batch([{"rank": 0, "sampler": "X", "tables": {}}])

        assert mock_sock.sendall.call_count == 1


# 3. RemoteDBStore.ingest() — list and dict formats


class TestRemoteDBStoreIngest:

    def _make_store(self):
        from traceml.database.remote_database_store import RemoteDBStore

        return RemoteDBStore(max_rows=100)

    def _make_message(self, rank: int, sampler: str, step: int) -> dict:
        return {
            "rank": rank,
            "sampler": sampler,
            "tables": {
                "TestTable": [{"step": step, "val": float(step)}],
            },
        }

    def test_ingest_single_dict_legacy_format(self):
        """ingest(dict) still works — backward compatible with old send()."""
        store = self._make_store()
        msg = self._make_message(rank=0, sampler="MySampler", step=1)

        store.ingest(msg)

        db = store.get_db(rank=0, sampler_name="MySampler")
        assert db is not None
        rows = list(db.get_table("TestTable"))
        assert len(rows) == 1
        assert rows[0]["step"] == 1

    def test_ingest_list_batch_format(self):
        """ingest(list) correctly dispatches each item — new send_batch() format."""
        store = self._make_store()
        msgs = [
            self._make_message(rank=0, sampler="SamplerA", step=10),
            self._make_message(rank=1, sampler="SamplerB", step=20),
        ]

        store.ingest(msgs)

        db_a = store.get_db(rank=0, sampler_name="SamplerA")
        db_b = store.get_db(rank=1, sampler_name="SamplerB")

        assert db_a is not None
        assert db_b is not None
        assert list(db_a.get_table("TestTable"))[0]["step"] == 10
        assert list(db_b.get_table("TestTable"))[0]["step"] == 20

    def test_ingest_list_multiple_samplers_same_rank(self):
        """A list with 10 payloads (one per sampler) from rank 0 all land correctly."""
        store = self._make_store()
        msgs = [
            self._make_message(rank=0, sampler=f"Sampler{i}", step=i)
            for i in range(10)
        ]

        store.ingest(msgs)

        for i in range(10):
            db = store.get_db(rank=0, sampler_name=f"Sampler{i}")
            assert db is not None, f"Sampler{i} DB missing"

    def test_ingest_none_is_noop(self):
        """ingest(None) should not raise."""
        store = self._make_store()
        store.ingest(None)  # must not raise

    def test_ingest_empty_list_is_noop(self):
        """ingest([]) should not raise or create any databases."""
        store = self._make_store()
        store.ingest([])
        assert store.ranks() == []


# 4. Before / after: sendall call-count comparison summary


class TestSyscallReduction:
    """
    Explicit before-vs-after test that documents the win.

    N_SAMPLERS independent send() calls  →  1 send_batch() call.
    """

    N_SAMPLERS = 10

    def _build_payloads(self):
        return [
            {
                "rank": 0,
                "sampler": f"Sampler{i}",
                "tables": {"T": [{"step": i}]},
            }
            for i in range(self.N_SAMPLERS)
        ]

    def test_before_n_sendall_calls(self):
        """OLD behaviour: N samplers = N sendall() calls."""
        from traceml.transport.tcp_transport import TCPClient, TCPConfig

        mock_sock = MagicMock()
        client = TCPClient(TCPConfig())
        client._sock = mock_sock
        client._connected = True

        for p in self._build_payloads():
            client.send(p)

        assert mock_sock.sendall.call_count == self.N_SAMPLERS, (
            f"Expected {self.N_SAMPLERS} sendall() calls (one per sampler), "
            f"got {mock_sock.sendall.call_count}"
        )

    def test_after_one_sendall_call(self):
        """NEW behaviour: N samplers = 1 sendall() call."""
        from traceml.transport.tcp_transport import TCPClient, TCPConfig

        mock_sock = MagicMock()
        client = TCPClient(TCPConfig())
        client._sock = mock_sock
        client._connected = True

        client.send_batch(self._build_payloads())

        assert mock_sock.sendall.call_count == 1, (
            f"Expected 1 sendall() call for the whole batch, "
            f"got {mock_sock.sendall.call_count}"
        )

    def test_reduction_factor(self):
        """The reduction is exactly N_SAMPLERS : 1."""
        from traceml.transport.tcp_transport import TCPClient, TCPConfig

        payloads = self._build_payloads()

        # Before
        before_sock = MagicMock()
        before_client = TCPClient(TCPConfig())
        before_client._sock = before_sock
        before_client._connected = True
        for p in payloads:
            before_client.send(p)

        # After
        after_sock = MagicMock()
        after_client = TCPClient(TCPConfig())
        after_client._sock = after_sock
        after_client._connected = True
        after_client.send_batch(payloads)

        before_calls = before_sock.sendall.call_count
        after_calls = after_sock.sendall.call_count

        assert before_calls == self.N_SAMPLERS
        assert after_calls == 1
        reduction = before_calls / after_calls
        assert (
            reduction == self.N_SAMPLERS
        ), f"Expected {self.N_SAMPLERS}× reduction, got {reduction:.1f}×"
