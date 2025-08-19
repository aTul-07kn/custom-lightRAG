# async_runner.py
import asyncio
import threading
from typing import Any, Optional

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


def start_background_loop() -> asyncio.AbstractEventLoop:
    """Start a background event loop in a daemon thread (idempotent)."""
    global _loop, _loop_thread
    if _loop and _loop.is_running():
        return _loop

    def _run(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(target=_run, args=(_loop,), daemon=True)
    _loop_thread.start()
    return _loop


def stop_background_loop() -> None:
    """Stop the background loop (best-effort)."""
    global _loop, _loop_thread
    if _loop and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)
    _loop = None
    _loop_thread = None


def run_coro_threadsafe(coro, timeout: Optional[float] = None) -> Any:
    """
    Schedule a coroutine on the background loop and wait for the result.
    Raises exceptions coming from the coroutine.
    """
    loop = start_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout)
