"""Tests for running the local model's server (llama-server).

A small fake server (``fake_llama_server.py``) is started as a real process, so
starting, waiting, crashing and stopping are exercised without loading a model.
"""

import socket
import sys
from pathlib import Path

import httpx
import pytest

from src.local_llm import LocalLLMServer, ServerStatus
from src.model_factory import LLMFactory, ModelUnavailableError
from tests.conftest import make_settings

FAKE_SERVER = str(Path(__file__).with_name("fake_llama_server.py"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeLlamaServer(LocalLLMServer):
    """Runs fake_llama_server.py with Python instead of the real executable."""

    def command(self) -> list[str]:
        return [sys.executable, FAKE_SERVER, *super().command()[1:]]


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "gemma-test.gguf"
    path.write_bytes(b"GGUF")
    return path


@pytest.fixture
def make_server(tmp_path, model_file):
    servers = []

    def make(*extra_args, port=None, timeout=10.0, model=model_file):
        server = FakeLlamaServer(
            executable=Path(sys.executable),
            model_path=model,
            port=port or free_port(),
            context_size=2048,
            startup_timeout=timeout,
            log_path=tmp_path / "logs" / "llama-server.log",
            extra_args=list(extra_args),
        )
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.stop()


def health(server) -> int | None:
    try:
        return httpx.get(f"http://127.0.0.1:{server.port}/health", timeout=2).status_code
    except httpx.HTTPError:
        return None


# --- Settings --------------------------------------------------------------------------


def test_not_configured_means_no_server():
    assert LocalLLMServer.from_settings(make_settings()) is None


def test_from_settings_builds_the_command(tmp_path):
    settings = make_settings(
        local_llm_server=tmp_path / "llama-server.exe",
        local_llm_model=tmp_path / "gemma.gguf",
        local_llm_port=9123,
        local_llm_context_size=4096,
        local_llm_args="--threads 4",
    )
    server = LocalLLMServer.from_settings(settings)

    assert server.command() == [
        str(tmp_path / "llama-server.exe"),
        "--model", str(tmp_path / "gemma.gguf"),
        "--host", "127.0.0.1",
        "--port", "9123",
        "--ctx-size", "4096",
        "--reasoning", "off",  # no silent "thinking" before the answer
        "--parallel", "1",  # one answer at a time
        "--threads", "4",  # extra args come last, so they can override the defaults
    ]  # fmt: skip
    assert server.base_url == "http://127.0.0.1:9123/v1" == settings.local_llm_base_url


# --- Starting and stopping ---------------------------------------------------------------


def test_starts_in_the_background_and_becomes_ready(make_server):
    server = make_server("--fake-load-seconds", "0.5")

    server.start()
    assert server.state().status is ServerStatus.STARTING

    server.wait_until_ready()
    assert server.state().status is ServerStatus.READY
    assert health(server) == 200


def test_start_twice_launches_one_process(make_server):
    server = make_server()
    server.start()
    server.start()
    server.wait_until_ready()
    first = server._process
    server.start()
    assert server._process is first


def test_stop_ends_the_process_and_frees_the_port(make_server):
    server = make_server()
    server.wait_until_ready()
    process = server._process

    server.stop()

    assert server.state().status is ServerStatus.STOPPED
    assert process.poll() is not None
    assert health(server) is None


def test_answers_through_the_openai_client(make_server):
    server = make_server()
    server.wait_until_ready()
    settings = make_settings(local_llm_port=server.port)
    llm = LLMFactory.create(settings, "local:gemma-test")

    assert llm.invoke("hi").text == "The local model says hi [1]."
    assert [c.text for c in llm.stream("hi") if c.text] == [
        "The local model ",
        "says hi ",
        "[1].",
    ]


# --- Failures --------------------------------------------------------------------------


def test_missing_model_file_is_reported(make_server, tmp_path):
    server = make_server(model=tmp_path / "missing.gguf")

    server.start()

    state = server.state()
    assert state.status is ServerStatus.ERROR
    assert "model file was not found" in state.detail
    with pytest.raises(ModelUnavailableError, match="could not be started"):
        server.wait_until_ready()


def test_missing_executable_is_reported(model_file, tmp_path):
    server = LocalLLMServer(
        executable=tmp_path / "no-llama-server.exe",
        model_path=model_file,
        port=free_port(),
        context_size=2048,
        startup_timeout=5,
        log_path=tmp_path / "log.txt",
    )
    server.start()
    assert "llama-server was not found" in server.state().detail


def test_server_that_exits_while_loading_is_reported_with_its_log(make_server, tmp_path):
    server = make_server("--fake-exit-code", "3")

    with pytest.raises(ModelUnavailableError):
        server.wait_until_ready()

    state = server.state()
    assert state.status is ServerStatus.ERROR
    assert "exit code 3" in state.detail
    assert "failed to load model" in state.detail  # last line of the log
    assert "failed to load model" in (tmp_path / "logs" / "llama-server.log").read_text()


def test_slow_load_times_out_and_stops_the_process(make_server):
    server = make_server("--fake-load-seconds", "30", timeout=1.0)

    with pytest.raises(ModelUnavailableError):
        server.wait_until_ready()

    state = server.state()
    assert state.status is ServerStatus.ERROR
    assert "did not finish loading within 1 seconds" in state.detail
    server._process.wait(timeout=10)  # terminated


def test_crashed_server_is_restarted_on_next_use(make_server):
    server = make_server()
    server.wait_until_ready()
    first = server._process
    first.kill()
    first.wait()

    assert server.refresh().status is ServerStatus.ERROR
    server.wait_until_ready()

    assert server.state().status is ServerStatus.READY
    assert server._process is not first
    assert health(server) == 200


def test_error_can_be_retried(make_server, tmp_path):
    missing = tmp_path / "later.gguf"
    server = make_server(model=missing)
    server.start()
    assert server.state().status is ServerStatus.ERROR

    missing.write_bytes(b"GGUF")  # e.g. the drive was plugged in
    server.wait_until_ready()
    assert server.state().status is ServerStatus.READY


def test_reuses_a_server_already_running_on_the_port(make_server):
    port = free_port()
    running = make_server(port=port)
    running.wait_until_ready()

    reuser = make_server(port=port)
    reuser.wait_until_ready()

    assert reuser.state().status is ServerStatus.READY
    assert reuser._process is None  # nothing new was launched

    running.stop()
    assert reuser.refresh().status is ServerStatus.STOPPED
