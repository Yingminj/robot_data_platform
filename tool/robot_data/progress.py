"""Terminal progress reporting for long conversions.

Purely a display concern, so it is module state rather than an
:class:`~robot_data.align.config.AlignmentConfig` field -- the config is
serialised verbatim into the dataset manifest, which should describe the
conversion, not the terminal it ran in.
"""

from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

PROGRESS_MODES = ("auto", "bar", "plain", "none")

_PROGRESS_MODE = "none"
_PROGRESS_PLAIN_INTERVAL_S = 10.0


def resolve_progress_mode(choice: str) -> str:
    """Resolve a ``--progress`` choice; ``auto`` follows stderr being a TTY."""
    if choice not in PROGRESS_MODES:
        raise ValueError(f"unknown progress mode: {choice}")
    if choice != "auto":
        return choice
    return "bar" if sys.stderr.isatty() else "plain"


def set_progress_mode(choice: str) -> None:
    """Select the progress display for this process (accepts ``auto``)."""
    global _PROGRESS_MODE
    _PROGRESS_MODE = resolve_progress_mode(choice)


def progress_enabled() -> bool:
    return _PROGRESS_MODE != "none"


@contextmanager
def progress_bar(desc: str, total: int | None, unit: str = "msg") -> Iterator[Callable[..., None]]:
    """Yield an ``advance(n=1)`` callable that drives a progress display.

    ``bar`` uses ``tqdm``; ``plain`` prints a line every few seconds, which is
    what you want in a log file or under nohup.  ``tqdm`` is optional -- if it
    is missing, ``bar`` degrades to ``plain`` rather than failing, so progress
    never becomes a hard dependency.  Output goes to stderr and bars erase
    themselves, leaving the per-bag stdout log as the durable record.
    """
    if _PROGRESS_MODE == "none":
        yield lambda n=1: None
        return

    tqdm = None
    if _PROGRESS_MODE == "bar":
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None

    if tqdm is not None:
        bar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            unit_scale=True,
            leave=False,
            file=sys.stderr,
            dynamic_ncols=True,
        )
        try:
            yield bar.update
        finally:
            bar.close()
        return

    seen = 0
    last = time.monotonic()

    def advance(n: int = 1) -> None:
        nonlocal seen, last
        seen += n
        now = time.monotonic()
        if now - last >= _PROGRESS_PLAIN_INTERVAL_S:
            last = now
            share = f" ({100.0 * seen / total:.1f}%)" if total else ""
            print(f"  {desc}: {seen}/{total or '?'} {unit}{share}", file=sys.stderr, flush=True)

    try:
        yield advance
    finally:
        print(f"  {desc}: {seen} {unit} done", file=sys.stderr, flush=True)


def tracked(iterable: Any, desc: str, total: int | None, unit: str = "msg") -> Iterator[Any]:
    """Yield from ``iterable``, advancing a progress bar for each item.

    A generator wrapper rather than an inline ``with`` so that hot loop bodies
    stay at their original indentation.  Closing the generator -- which Python
    does when the consuming loop is abandoned or raises -- tears the bar down.
    """
    with progress_bar(desc, total, unit) as advance:
        for item in iterable:
            advance()
            yield item


def connection_total(connections: list[Any]) -> int | None:
    """Exact message count for the connections about to be iterated."""
    total = 0
    for connection in connections:
        count = getattr(connection, "msgcount", None)
        if not count:
            return None
        total += int(count)
    return total or None


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


class EpisodeProgress:
    """Single redrawing block on a TTY, one line per episode otherwise.

    Used by the file-per-episode converters, where the natural unit of progress
    is a whole episode rather than a message.  :func:`progress_bar` covers the
    inner loops.
    """

    LINES = 4

    def __init__(self, mode: str, total: int, source: Path, output: Path) -> None:
        if mode == "auto":
            mode = "bar" if sys.stdout.isatty() else "plain"
        self.mode = mode
        self.total = total
        self.title = f"Converting {source.name} -> {output.name}"
        self.started = time.monotonic()
        self.done = 0
        self.frames = 0
        self.degenerate = 0
        self.failed = 0
        self.current = ""
        self.drawn = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def start(self) -> None:
        if self.mode == "bar":
            self._draw()  # the block carries its own title line
        elif self.mode != "none":
            print(self.title, flush=True)

    def set_current(self, path: Path) -> None:
        self.current = path.name
        if self.mode == "bar":
            self._draw()

    def episode_done(self, frames: int, degenerate: bool) -> None:
        self.done += 1
        self.frames += frames
        self.degenerate += int(degenerate)
        self._render_after_episode()

    def episode_failed(self) -> None:
        self.done += 1
        self.failed += 1
        self._render_after_episode()

    def log(self, message: str) -> None:
        """Emit a message without leaving a torn progress block behind."""
        if self.mode == "bar" and self.drawn:
            self._clear()
        print(message, file=sys.stderr, flush=True)
        if self.mode == "bar":
            self._draw()

    def finish(self) -> None:
        """Leave the final block on screen; safe to call more than once."""
        if self.mode == "bar" and self.drawn:
            self._draw()
            self.drawn = False
            print(flush=True)

    def _render_after_episode(self) -> None:
        if self.mode == "bar":
            self._draw()
        elif self.mode == "plain":
            percent = 100 * self.done / self.total if self.total else 0.0
            print(
                f"[{self.done}/{self.total}] {percent:5.1f}%  frames {self.frames:,}  "
                f"elapsed {format_duration(self.elapsed)}  ETA {format_duration(self._eta())}  "
                f"{self.current}",
                flush=True,
            )

    def _eta(self) -> float:
        if self.done <= 0 or self.done >= self.total:
            return 0.0
        return self.elapsed / self.done * (self.total - self.done)

    def _clear(self) -> None:
        sys.stdout.write(f"\033[{self.LINES}A\033[J")
        sys.stdout.flush()
        self.drawn = False

    def _draw(self) -> None:
        if self.drawn:
            sys.stdout.write(f"\033[{self.LINES}A")
        ratio = self.done / self.total if self.total else 0.0
        width = 28
        filled = int(round(width * ratio))
        bar = "#" * filled + "." * (width - filled)
        eta = "--" if self.done == 0 else format_duration(self._eta())
        status = f"current: {self.current}" if self.current else "current: -"
        if self.degenerate:
            status += f"  |  action==state: {self.degenerate} episode(s)"
        if self.failed:
            status += f"  |  failed: {self.failed}"
        columns = shutil.get_terminal_size((100, 24)).columns
        lines = [
            self.title,
            f"[{bar}]  {self.done}/{self.total} episodes  {100 * ratio:3.0f}%",
            f"frames {self.frames:,} | elapsed {format_duration(self.elapsed)} | ETA {eta}",
            status,
        ]
        for line in lines:
            sys.stdout.write("\033[2K" + line[: max(columns - 1, 20)] + "\n")
        sys.stdout.flush()
        self.drawn = True
