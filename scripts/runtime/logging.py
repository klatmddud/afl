from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType
from typing import TextIO


class TeeStream:
    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()


class TeeOutput:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file: TextIO | None = None
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None

    def __enter__(self) -> "TeeOutput":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = TeeStream(self.stdout, self.file)  # type: ignore[assignment]
        sys.stderr = TeeStream(self.stderr, self.file)  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self.stdout is not None:
            sys.stdout = self.stdout
        if self.stderr is not None:
            sys.stderr = self.stderr
        if self.file is not None:
            self.file.close()
        return False

