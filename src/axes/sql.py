"""
sql() — primary query call, and SqlResult.

Usage::

    from axes import sql

    result = sql("SELECT state, AVG(income) FROM acs.demographics GROUP BY state")

    result.bytes       # size of the Arrow IPC response in bytes — use this
                       # to decide whether to load into memory
    df = result.collect()  # load fully into memory as a polars DataFrame
    lf = result.scan()     # lazy polars LazyFrame — safe for large results
    result.save("/work/result.parquet")  # write to disk as parquet
"""

from __future__ import annotations

import io
import logging
import queue
import tempfile
import threading
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from axes.client import Client, get_default_client
from axes.exceptions import AuthError, QueryError, ResultTooLarge

log = logging.getLogger(__name__)

_SPILL_THRESHOLD = 10 * 1024 * 1024  # 10 MB
_ARROW_MIME = "application/vnd.apache.arrow.stream"
_CHUNK_SIZE = 64 * 1024


class SqlResult:
    """Result of a completed SQL query.

    Attributes:
        bytes:      Size of the Arrow IPC response in bytes.  Use this to
                    decide whether calling ``.collect()`` is safe for your
                    available memory before loading the full result.
        columns:    Ordered list of column names.
        elapsed_ms: Wall-clock time for the full request in milliseconds.
        rows:       Number of rows in the result.
    """

    def __init__(
        self,
        *,
        df: pl.DataFrame | None,
        spill_path: Path | None,
        bytes: int,
        columns: list[str],
        elapsed_ms: int,
        rows: int,
        _owns_spill: bool = True,
    ) -> None:
        self._df = df
        self._spill_path = spill_path
        self._owns_spill = _owns_spill
        self.bytes = bytes
        self.columns = columns
        self.elapsed_ms = elapsed_ms
        self.rows = rows

    def collect(self) -> pl.DataFrame:
        """Return the full result as an eager polars DataFrame.

        For large results this materialises everything in memory.  Check
        ``result.bytes`` first if you are concerned about available RAM.
        """
        if self._df is not None:
            return self._df
        assert self._spill_path is not None
        return pl.read_parquet(self._spill_path)

    def scan(self) -> pl.LazyFrame:
        """Return a polars LazyFrame over the result.

        Safe for arbitrarily large results — when the response was spilled
        to disk this returns ``pl.scan_parquet`` over the temp file rather
        than loading into memory.
        """
        if self._df is not None:
            return self._df.lazy()
        assert self._spill_path is not None
        return pl.scan_parquet(self._spill_path)

    def save(self, path: str | Path) -> Path:
        """Write the result to *path* as a parquet file and return the path."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self._spill_path is not None:
            import shutil
            shutil.copy2(self._spill_path, dest)
        else:
            assert self._df is not None
            self._df.write_parquet(dest)
        return dest

    def __del__(self) -> None:
        if self._owns_spill and self._spill_path is not None:
            try:
                self._spill_path.unlink(missing_ok=True)
            except Exception:
                pass


def sql(
    query: str,
    *,
    out: str | Path | None = None,
    client: Client | None = None,
) -> SqlResult:
    """Execute *query* against the Axes API and return a ``SqlResult``.

    The server streams Arrow IPC record batches. Results under 10 MB are
    kept in memory as a polars DataFrame; larger results spill to a
    temporary parquet file in ``/tmp`` owned by the returned ``SqlResult``
    and deleted when it is garbage collected.

    When *out* is provided the result is written as parquet to that path
    as it streams in — no full materialisation in memory. The returned
    ``SqlResult`` is backed by that file and will not be deleted on
    garbage collection since the caller owns it.

    Check ``result.bytes`` before calling ``.collect()`` if you are
    concerned about available memory. Use ``.scan()`` for safe lazy
    access regardless of result size.

    Args:
        query:  SQL string to execute.
        out:    Optional path to write the result as parquet directly.
        client: Use this ``Client`` instead of the module-level default.

    Returns:
        A ``SqlResult`` with the DataFrame and result metadata.

    Raises:
        QueryError:     The server rejected the SQL (HTTP 400).
        ResultTooLarge: The result exceeded server caps (HTTP 413).
        AuthError:      Authentication / authorisation failure (401/403).
    """
    client = client or get_default_client()
    t0 = time.monotonic()

    if out is not None:
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows, columns, byte_count = _stream_to_parquet_file(client, query, dest)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return SqlResult(
            df=None,
            spill_path=dest,
            bytes=byte_count,
            columns=columns,
            elapsed_ms=elapsed_ms,
            rows=rows,
            _owns_spill=False,
        )

    with tempfile.SpooledTemporaryFile(
        max_size=_SPILL_THRESHOLD, dir="/tmp"
    ) as buf:
        with client.query(query, accept=_ARROW_MIME) as response:
            if response.status_code != 200:
                response.read()
            _raise_for_status(response, query)
            for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                buf.write(chunk)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        byte_count = buf.tell()

        buf.seek(0)

        if byte_count > _SPILL_THRESHOLD:
            spill_path = _arrow_buf_to_parquet_tempfile(buf)
            lf = pl.scan_parquet(spill_path)
            columns = list(lf.collect_schema().keys())
            rows = lf.select(pl.len()).collect().item()
            return SqlResult(
                df=None,
                spill_path=spill_path,
                bytes=byte_count,
                columns=columns,
                elapsed_ms=elapsed_ms,
                rows=rows,
            )
        else:
            df = pl.from_arrow(pa.ipc.open_stream(buf).read_all())
            return SqlResult(
                df=df,
                spill_path=None,
                bytes=byte_count,
                columns=list(df.schema.keys()),
                elapsed_ms=elapsed_ms,
                rows=len(df),
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stream_to_parquet_file(
    client: Client,
    query: str,
    dest: Path,
) -> tuple[int, list[str], int]:
    """Stream Arrow IPC response directly to a parquet file.

    Uses a queue to pipe HTTP chunks to a reader thread that converts
    each Arrow record batch to a parquet row group as it arrives.
    Memory use is bounded to one batch at a time — the full response
    is never materialised.

    Returns (rows, columns, byte_count).
    """
    chunk_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=16)
    byte_count = 0
    error: list[Exception] = []

    class _QueueReader(io.RawIOBase):
        """Feed Arrow IPC bytes from the HTTP chunk queue to pyarrow.

        PyArrow's IPC reader calls readinto expecting a full read (like a
        file), so we must loop until the requested buffer is filled or the
        stream ends — short reads cause it to bail out with an OSError.
        """

        def __init__(self) -> None:
            self._buf = b""
            self._eof = False

        def readable(self) -> bool:
            return True

        def readinto(self, b: bytearray) -> int:
            total = 0
            while total < len(b):
                if not self._buf:
                    if self._eof:
                        break
                    chunk = chunk_queue.get()
                    if chunk is None:
                        self._eof = True
                        break
                    self._buf = chunk
                n = min(len(b) - total, len(self._buf))
                b[total:total + n] = self._buf[:n]
                self._buf = self._buf[n:]
                total += n
            return total

    rows = 0
    columns: list[str] = []
    writer: pq.ParquetWriter | None = None

    def _parquet_writer_thread() -> None:
        nonlocal rows, columns, writer
        try:
            log.debug("writer thread: opening Arrow IPC stream")
            arrow_reader = pa.ipc.open_stream(_QueueReader())
            columns = arrow_reader.schema.names
            log.debug("writer thread: schema read, columns=%s", columns)
            for batch in arrow_reader:
                if writer is None:
                    log.debug("writer thread: opening ParquetWriter")
                    writer = pq.ParquetWriter(dest, batch.schema, compression="zstd")
                writer.write_table(pa.Table.from_batches([batch]))
                rows += len(batch)
                log.debug("writer thread: wrote batch rows=%d total=%d", len(batch), rows)
        except Exception as exc:
            log.debug("writer thread: exception: %r", exc)
            error.append(exc)
        finally:
            if writer is not None:
                log.debug("writer thread: closing ParquetWriter")
                writer.close()
            log.debug("writer thread: done")

    writer_thread = threading.Thread(target=_parquet_writer_thread, daemon=True)
    writer_thread.start()

    try:
        log.debug("http thread: opening stream request")
        with client.query(query, accept=_ARROW_MIME) as response:
            log.debug("http thread: response status=%d", response.status_code)
            if response.status_code != 200:
                response.read()
            try:
                _raise_for_status(response, query)
            except Exception:
                chunk_queue.put(None)
                writer_thread.join()
                raise
            for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                byte_count += len(chunk)
                log.debug("http thread: got chunk size=%d total=%d", len(chunk), byte_count)
                chunk_queue.put(chunk)
        log.debug("http thread: stream done, total bytes=%d", byte_count)
    finally:
        log.debug("http thread: sending EOF to writer")
        chunk_queue.put(None)
        log.debug("http thread: joining writer thread")
        writer_thread.join()
        log.debug("http thread: writer thread joined")

    if error:
        raise error[0]

    return rows, columns, byte_count


def _arrow_buf_to_parquet_tempfile(buf: object) -> Path:
    """Convert an Arrow IPC stream buffer to a parquet temp file."""
    reader = pa.ipc.open_stream(buf)  # type: ignore[arg-type]
    tmp = tempfile.NamedTemporaryFile(
        suffix=".parquet", dir="/tmp", delete=False
    )
    try:
        writer: pq.ParquetWriter | None = None
        try:
            for batch in reader:
                if writer is None:
                    writer = pq.ParquetWriter(tmp.name, batch.schema)
                writer.write_table(pa.Table.from_batches([batch]))
        finally:
            if writer is not None:
                writer.close()
    finally:
        tmp.close()
    return Path(tmp.name)


def _raise_for_status(response: object, query: str) -> None:
    """Translate HTTP error codes to typed exceptions."""
    status = getattr(response, "status_code", None)
    if status is None:
        return

    if status in (401, 403):
        text = _read_text(response)
        raise AuthError(status, text or f"HTTP {status}")

    if status == 413:
        text = _read_text(response)
        raise ResultTooLarge(text or "Result exceeded server caps")

    if status == 400:
        text = _read_text(response)
        raise QueryError(text or "Bad query", query=query)

    if status >= 400:
        text = _read_text(response)
        raise QueryError(f"HTTP {status}: {text}", query=query)


def _read_text(response: object) -> str:
    try:
        return response.text  # type: ignore[attr-defined]
    except Exception:
        return ""
