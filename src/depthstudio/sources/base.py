"""Common interface for every frame source (live cameras and recordings)."""

import abc
import time
from typing import Iterator, Optional

from ..frames import Frame


class SourceError(RuntimeError):
    pass


class Source(abc.ABC):
    """A stream of Frames. Use as a context manager, or call start()/stop() yourself."""

    name = "source"

    @abc.abstractmethod
    def start(self) -> None:
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        ...

    @abc.abstractmethod
    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        """Return the next frame, or None if none arrived within `timeout` seconds.

        Raises StopIteration when a finite source (a recording) is exhausted.
        """

    def describe(self) -> dict:
        """Static metadata written next to recordings."""
        return {"source": self.name}

    def frames(self, duration: Optional[float] = None, count: Optional[int] = None,
               timeout: float = 5.0) -> Iterator[Frame]:
        """Yield frames until `duration` seconds or `count` frames; SourceError if the stream stalls."""
        t_end = None if duration is None else time.monotonic() + duration
        n = 0
        while (count is None or n < count) and (t_end is None or time.monotonic() < t_end):
            try:
                frame = self.read(timeout)
            except StopIteration:
                return
            if frame is None:
                raise SourceError("%s: no frame within %.1f s" % (self.name, timeout))
            n += 1
            yield frame

    def __enter__(self) -> "Source":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
