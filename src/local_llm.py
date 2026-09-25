"""Runs a local GGUF model with llama.cpp's ``llama-server`` (FR-13, NFR-01).

The API starts the server when the admin selects the local model and stops it when
another model is selected, so the model only uses memory while it is in use. The
server speaks the OpenAI chat API on ``127.0.0.1``, so the model never leaves this
machine and nothing is reachable from the network.

Output from the server is written to ``LOCAL_LLM_LOG_FILE`` for troubleshooting.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO

import httpx

from src.config import LOCAL_LLM_HOST as HOST
from src.config import Settings, get_settings
from src.model_factory import ModelUnavailableError

logger = logging.getLogger(__name__)
_POLL_SECONDS = 0.5
_STOP_TIMEOUT_SECONDS = 10
# Callers wait a little longer than the loader, so the loader can report why it failed.
_WAIT_MARGIN_SECONDS = 10


class ServerStatus(str, Enum):
    STOPPED = "stopped"  # not running; starts when the model is selected
    STARTING = "starting"  # loading the model
    READY = "ready"
    ERROR = "error"  # could not start, or stopped unexpectedly


@dataclass(frozen=True)
class ServerState:
    status: ServerStatus
    detail: str = ""  # why it failed, for the admin


class LocalLLMServer:
    """Starts, watches and stops one ``llama-server`` process."""

    def __init__(
        self,
        executable: Path,
        model_path: Path,
        port: int,
        context_size: int,
        startup_timeout: float,
        log_path: Path,
        extra_args: list[str] | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.model_path = Path(model_path)
        self.port = port
        self._context_size = context_size
        self._startup_timeout = startup_timeout
        self._log_path = Path(log_path)
        self._extra_args = list(extra_args or [])

        self._process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None
        self._state = ServerState(ServerStatus.STOPPED)
        self._generation = 0  # bumped by every start/stop, so stale watchers give up
        self._changed = threading.Condition()
        atexit.register(self.stop)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LocalLLMServer | None:
        """The configured server, or None if no local model is set up."""
        settings = settings or get_settings()
        if settings.local_llm_server is None or settings.local_llm_model is None:
            return None
        return cls(
            executable=settings.local_llm_server,
            model_path=settings.local_llm_model,
            port=settings.local_llm_port,
            context_size=settings.local_llm_context_size,
            startup_timeout=settings.local_llm_startup_timeout,
            log_path=settings.local_llm_log_file,
            extra_args=settings.local_llm_args.split(),
        )

    @property
    def base_url(self) -> str:
        """OpenAI-compatible endpoint for the chat model."""
        return f"http://{HOST}:{self.port}/v1"

    def command(self) -> list[str]:
        return [
            str(self.executable),
            "--model", str(self.model_path),
            "--host", HOST,
            "--port", str(self.port),
            "--ctx-size", str(self._context_size),
            # Answer at once instead of "thinking" silently first; on a CPU the hidden
            # reasoning took longer than the answer itself.
            "--reasoning", "off",
            # One answer at a time: answers sharing a CPU are all slow.
            "--parallel", "1",
            *self._extra_args,  # last, so they can override the options above
        ]  # fmt: skip

    # --- State ---------------------------------------------------------------------

    def state(self) -> ServerState:
        with self._changed:
            return self._state

    def _set(self, generation: int, status: ServerStatus, detail: str = "") -> bool:
        """Update the state unless a newer start/stop has happened since ``generation``."""
        with self._changed:
            if generation != self._generation:
                return False
            self._state = ServerState(status, detail)
            self._changed.notify_all()
            return True

    # --- Start / stop ------------------------------------------------------------

    def start(self) -> None:
        """Start the server in the background. Does nothing if it's running or loading."""
        with self._changed:
            if self._state.status in (ServerStatus.STARTING, ServerStatus.READY):
                return
            self._generation += 1
            generation = self._generation

            missing = self._missing_file()
            if missing:
                self._state = ServerState(ServerStatus.ERROR, missing)
                self._changed.notify_all()
                return

            self._state = ServerState(ServerStatus.STARTING)
            self._changed.notify_all()

        threading.Thread(
            target=self._launch, args=(generation,), name="llama-server-start", daemon=True
        ).start()

    def stop(self) -> None:
        """Stop the server (if this API started it) and free its memory."""
        with self._changed:
            self._generation += 1
            process, log = self._process, self._log
            self._process = self._log = None
            self._state = ServerState(ServerStatus.STOPPED)
            self._changed.notify_all()

        if process is not None and process.poll() is None:
            logger.info("Stopping llama-server (pid %d)", process.pid)
            process.terminate()
            try:
                process.wait(timeout=_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log is not None:
            log.close()

    def wait_until_ready(self) -> None:
        """Start the server if needed and wait until it can answer.

        Raises ``ModelUnavailableError`` with a message for the user if it can't. A server
        that crashed since the last question is started again.
        """
        self.refresh()
        self.start()
        deadline = time.monotonic() + self._startup_timeout + _WAIT_MARGIN_SECONDS
        with self._changed:
            while self._state.status is ServerStatus.STARTING:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(remaining)
            state = self._state

        if state.status is ServerStatus.READY and not self._crashed():
            return
        if state.status is ServerStatus.STARTING:
            raise ModelUnavailableError(
                "The local model is still loading. Please try again in a minute."
            )
        raise ModelUnavailableError(
            "The local model could not be started. An administrator can see why on the "
            "Admin page (Model tab)."
        )

    # --- Internals ---------------------------------------------------------------

    def _missing_file(self) -> str:
        if not self.executable.is_file():
            return f"llama-server was not found at {self.executable}."
        if not self.model_path.is_file():
            return f"The model file was not found at {self.model_path}."
        return ""

    def _launch(self, generation: int) -> None:
        # A server may already be running on the port, e.g. left over after the API was
        # killed without shutting down. Use it rather than failing to bind the port.
        if self._healthy():
            logger.info("Using the llama-server already running on port %d", self.port)
            self._set(generation, ServerStatus.READY)
            return

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self._log_path.open("ab")
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting llama-server\n".encode())
        log.flush()
        try:
            process = subprocess.Popen(
                self.command(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Don't open a console window for the server on Windows.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except OSError as exc:
            log.close()
            logger.error("Could not start llama-server: %s", exc)
            self._set(generation, ServerStatus.ERROR, f"llama-server could not be started: {exc}")
            return

        with self._changed:
            if generation != self._generation:  # stopped while we were launching
                process.terminate()
                log.close()
                return
            self._process, self._log = process, log
        logger.info("Started llama-server (pid %d): %s", process.pid, self.model_path.name)

        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self.state().status is not ServerStatus.STARTING:
                return  # stopped meanwhile
            if process.poll() is not None:
                self._set(
                    generation,
                    ServerStatus.ERROR,
                    f"llama-server stopped while loading the model (exit code "
                    f"{process.returncode}). {self._log_hint()}",
                )
                return
            if self._healthy():
                logger.info("Local model is ready: %s", self.model_path.name)
                self._set(generation, ServerStatus.READY)
                return
            time.sleep(_POLL_SECONDS)

        if self._set(
            generation,
            ServerStatus.ERROR,
            f"The model did not finish loading within {self._startup_timeout:g} seconds. "
            f"{self._log_hint()}",
        ):
            process.terminate()

    def _crashed(self) -> bool:
        """Whether a server this API started has exited; records the error if so."""
        with self._changed:
            process, generation = self._process, self._generation
        if process is None or process.poll() is None:
            return False
        self._set(
            generation,
            ServerStatus.ERROR,
            f"llama-server stopped unexpectedly (exit code {process.returncode}). "
            f"{self._log_hint()}",
        )
        return True

    def refresh(self) -> ServerState:
        """The current state, after checking the server is still alive."""
        if self._crashed():
            return self.state()
        with self._changed:
            reused = self._process is None and self._state.status is ServerStatus.READY
            generation = self._generation
        # A server we reused but didn't start can only be checked over HTTP.
        if reused and not self._healthy():
            self._set(generation, ServerStatus.STOPPED)
        return self.state()

    def _healthy(self) -> bool:
        """llama-server answers /health with 200 once the model is loaded (503 before)."""
        if not self._port_open():
            return False
        try:
            return httpx.get(f"http://{HOST}:{self.port}/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def _port_open(self) -> bool:
        # On Windows a refused local connection takes ~2 s to fail; a listening server
        # accepts in well under this timeout.
        try:
            with socket.create_connection((HOST, self.port), timeout=0.5):
                return True
        except OSError:
            return False

    def _log_hint(self) -> str:
        last = self._last_log_line()
        where = f"See {self._log_path} for details."
        return f"Last message: {last} {where}" if last else where

    def _last_log_line(self) -> str:
        try:
            with self._log_path.open("rb") as file:
                file.seek(0, os.SEEK_END)
                file.seek(max(0, file.tell() - 4096))
                lines = file.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        lines = [line.strip() for line in lines if line.strip() and not line.startswith("---")]
        return lines[-1][:300] if lines else ""
