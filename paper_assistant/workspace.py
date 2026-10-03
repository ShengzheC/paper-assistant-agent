"""The folder the agent works in: write scripts, run them with figure checks, read outputs.

Scripts run in a subprocess with a timeout, and paths cannot leave the folder. The code is
written by a model and runs on your machine with your permissions, so use a container or VM
for anything you would not run yourself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # so the subprocess can import paper_assistant
WRITABLE_SUFFIXES = {".py", ".csv", ".txt", ".json", ".tex", ".md"}
VECTOR_SUFFIXES = (".pdf", ".svg", ".eps")
MAX_OUTPUT_LINES = 30


class WorkspaceError(ValueError):
    """A request the workspace refuses. The message goes back to the model."""


@dataclass
class RunResult:
    exit_code: int | None  # None if the script timed out
    seconds: float
    stdout: str
    stderr: str
    figures: list[dict] = field(default_factory=list)  # one record per saved file (see _boot.py)

    @property
    def saved_vector(self) -> bool:
        return any(f["file"].lower().endswith(VECTOR_SUFFIXES) for f in self.figures)

    @property
    def issues(self) -> list[str]:
        """Distinct figure-check problems, plus a missing vector file."""
        found = list(dict.fromkeys(issue for f in self.figures for issue in f["issues"]))
        if self.figures and not self.saved_vector:
            found.append("no vector file saved; save a .pdf for LaTeX")
        return found

    def report(self) -> str:
        status = "timed out" if self.exit_code is None else f"exit code {self.exit_code}"
        lines = [f"{status} after {self.seconds:.1f} s"]
        for name, text in (("stdout", self.stdout), ("stderr", self.stderr)):
            tail = text.strip().splitlines()[-MAX_OUTPUT_LINES:]
            if tail:
                lines += [f"--- {name} (last {len(tail)} lines) ---", *tail]
        if not self.figures:
            lines.append("No figure was saved.")
        for f in self.figures:
            width, height = f["size_in"]
            verdict = "checks passed" if not f["issues"] else "issues: " + "; ".join(f["issues"])
            lines.append(f"saved {f['file']} ({width} x {height} in), preview {f['preview']}: {verdict}")
        if self.figures and not self.saved_vector:
            lines.append("No vector file (.pdf/.svg) was saved; LaTeX papers need one.")
        return "\n".join(lines)


class Workspace:
    def __init__(self, root: Path | str, data_files: list[Path] = (), timeout: float = 120.0):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        for source in data_files:
            shutil.copyfile(source, self.root / Path(source).name)

    def list_files(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def path(self, rel: str) -> Path:
        path = (self.root / str(rel)).resolve()
        if not path.is_relative_to(self.root):
            raise WorkspaceError(f"{rel} is outside the workspace")
        return path

    def read(self, rel: str) -> str:
        path = self.path(rel)
        if not path.is_file():
            raise WorkspaceError(f"no such file: {rel}")
        return path.read_text(encoding="utf-8", errors="replace")

    def write(self, rel: str, content: str) -> None:
        path = self.path(rel)
        if path.suffix.lower() not in WRITABLE_SUFFIXES:
            raise WorkspaceError(f"can only write {', '.join(sorted(WRITABLE_SUFFIXES))} files")
        if path.name.startswith("_"):
            raise WorkspaceError("files starting with '_' are reserved for the runner")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def run(self, rel: str) -> RunResult:
        script = self.path(rel)
        if script.suffix != ".py" or not script.is_file():
            raise WorkspaceError(f"no Python script at {rel}")
        for stale in [self.root / "_figures.json", *self.root.glob("_preview_*.png")]:
            stale.unlink(missing_ok=True)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(PACKAGE_ROOT), os.environ.get("PYTHONPATH")])),
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "Agg",
        }
        command = [sys.executable, "-m", "paper_assistant._boot", script.relative_to(self.root).as_posix()]
        start = time.time()
        try:
            proc = subprocess.run(
                command, cwd=self.root, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self.timeout, env=env,
            )
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return RunResult(None, time.time() - start, stdout, f"killed after {self.timeout:g} s", self._figures())
        return RunResult(proc.returncode, time.time() - start, proc.stdout, proc.stderr, self._figures())

    def _figures(self) -> list[dict]:
        report = self.root / "_figures.json"
        return json.loads(report.read_text(encoding="utf-8")) if report.is_file() else []
