"""Small completion-order request pool used by all streaming examples.

Each handler owns one serial inference connection.  Requests may complete out of
submission order, therefore identity and caller context travel with every result.
Worker failures are delivered through the result queue immediately instead of
being discovered only after the caller's full timeout expires.
"""

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Iterable, Optional


@dataclass(frozen=True)
class _QueuedRequest:
    request_id: int
    payload: Any
    context: Any
    submitted_at: float


@dataclass
class InferenceResult:
    request_id: int
    value: Any = None
    context: Any = None
    infer_ms: float = 0.0
    error: Optional[BaseException] = None

    def __iter__(self):
        """Compatibility with the old ``(value, infer_ms) = pool.get()`` API."""
        yield self.value
        yield self.infer_ms


class RequestPool:
    """Run one worker thread per serial inference server.

    ``submit(payload, context=...)`` returns a monotonically increasing request
    id. ``get()`` returns :class:`InferenceResult` in completion order.  The
    result object remains iterable as ``(value, infer_ms)`` for compatibility
    with the pre-refactor examples.
    """

    _STOP = object()

    def __init__(self, handlers: Iterable[Callable[[Any], Any]], queue_limit=None,
                 default_timeout=None):
        self._handlers = list(handlers)
        if not self._handlers:
            raise ValueError("RequestPool needs at least one handler")
        if queue_limit is None:
            queue_limit = 2 * len(self._handlers)
        if int(queue_limit) < 1:
            raise ValueError("queue_limit must be >= 1")
        if default_timeout is not None and float(default_timeout) <= 0:
            raise ValueError("default_timeout must be > 0")

        self._default_timeout = default_timeout
        self._todo = queue.Queue(maxsize=int(queue_limit))
        self._done = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._fatal = None
        self._in_flight = 0
        self._next_request_id = 1
        self._threads = [
            threading.Thread(target=self._work, args=(handler,), daemon=True,
                             name="inference-worker-%d" % idx)
            for idx, handler in enumerate(self._handlers)
        ]
        for thread in self._threads:
            thread.start()

    def _set_fatal(self, exc):
        with self._lock:
            if self._fatal is None:
                self._fatal = exc

    def _raise_if_fatal(self):
        with self._lock:
            err = self._fatal
        if err is not None:
            raise RuntimeError("inference failed: %s" % err) from err

    def _work(self, handler):
        while True:
            try:
                req = self._todo.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if req is self._STOP:
                return

            try:
                value = handler(req.payload)
                msg = InferenceResult(
                    request_id=req.request_id,
                    value=value,
                    context=req.context,
                    infer_ms=(time.monotonic() - req.submitted_at) * 1000.0,
                )
            except BaseException as exc:  # propagate dead servers immediately
                msg = InferenceResult(
                    request_id=req.request_id,
                    context=req.context,
                    infer_ms=(time.monotonic() - req.submitted_at) * 1000.0,
                    error=exc,
                )
                self._set_fatal(exc)
                self._stop.set()
            finally:
                self._done.put(msg)
            if msg.error is not None:
                return

    def submit(self, payload, context=None):
        """Queue a request and return its request id.

        Back-pressure remains bounded, but a worker failure is checked while the
        caller waits for queue space so a full queue cannot mask a dead server.
        """
        self._raise_if_fatal()
        if self._stop.is_set():
            raise RuntimeError("request pool is closed")
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._in_flight += 1
        req = _QueuedRequest(request_id, payload, context, time.monotonic())
        try:
            while True:
                try:
                    self._todo.put(req, timeout=0.1)
                    break
                except queue.Full:
                    self._raise_if_fatal()
                    if self._stop.is_set():
                        raise RuntimeError("request pool is closed")
        except BaseException:
            with self._lock:
                self._in_flight = max(0, self._in_flight - 1)
            raise
        return request_id

    def get(self, timeout=None):
        """Return the next finished result in completion order."""
        timeout = self._default_timeout if timeout is None else timeout
        if timeout is not None and float(timeout) <= 0:
            raise ValueError("timeout must be > 0")
        try:
            msg = self._done.get(timeout=timeout)
        except queue.Empty:
            self._raise_if_fatal()
            if self._stop.is_set():
                raise RuntimeError("request pool closed before a result arrived")
            if timeout is None:
                raise RuntimeError("no inference response")
            raise TimeoutError("no inference response within %.1f seconds" % timeout)

        if msg is self._STOP:
            raise RuntimeError("request pool is closed")

        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
        if msg.error is not None:
            raise RuntimeError(
                "inference request %d failed: %s" % (msg.request_id, msg.error)
            ) from msg.error
        return msg

    @property
    def servers(self):
        return len(self._handlers)

    @property
    def in_flight(self):
        with self._lock:
            return self._in_flight

    def close(self):
        """Stop workers. Safe to call repeatedly."""
        first_close = not self._stop.is_set()
        if first_close:
            self._stop.set()
            # Wake a caller already blocked in get().
            self._done.put(self._STOP)
        for _ in self._threads:
            try:
                self._todo.put_nowait(self._STOP)
            except queue.Full:
                break
        for thread in self._threads:
            thread.join(timeout=1.5)

