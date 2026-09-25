"""The chat model the admin selected: one of ``GROQ_MODELS``, or the local model.

The choice is saved to a small JSON file (``MODEL_SELECTION_FILE``) so it survives
restarts, and it takes effect on the next question without restarting the API.
Selecting the local model starts its server; selecting another model stops it, so
the local model only uses memory while it is selected.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.local_llm import LocalLLMServer
from src.model_factory import ChatModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    description: str
    kind: Literal["cloud", "local"] = "cloud"
    # Cloud models are always "ready"; the local model reports its server's state.
    status: Literal["stopped", "starting", "ready", "error"] = "ready"
    detail: str = ""


@dataclass(frozen=True)
class LocalModel:
    """The local model and the server that runs it."""

    id: str
    label: str
    server: LocalLLMServer


_LOCAL_DESCRIPTION = (
    "Runs on this server: questions and documents never leave the machine. "
    "Slower than the cloud models, especially the first answer."
)


# Friendly names for models we know; any other model ID is shown as-is.
_KNOWN_MODELS: dict[str, tuple[str, str]] = {
    "openai/gpt-oss-120b": (
        "GPT-OSS 120B",
        "Best answer quality. OpenAI's large open-weight model.",
    ),
    "openai/gpt-oss-20b": (
        "GPT-OSS 20B",
        "Fastest. Good for simple, direct questions.",
    ),
    "qwen/qwen3.8-27b": (
        "Qwen 3.8 27B",
        "Alibaba's Qwen model. Strong at reasoning over longer documents.",
    ),
}


def describe(model_id: str) -> ModelOption:
    label, description = _KNOWN_MODELS.get(model_id, (model_id, ""))
    return ModelOption(id=model_id, label=label, description=description)


class ModelSelector:
    """Knows which models are allowed, which one is selected, and builds it."""

    def __init__(
        self,
        available: list[str],
        factory: Callable[[str], ChatModel],
        store_path: Path | None = None,
        local: LocalModel | None = None,
    ) -> None:
        if not available:
            raise ValueError("At least one model must be available.")
        self._available = list(available)
        if local is not None and local.id not in self._available:
            self._available.append(local.id)
        self._local = local
        self._factory = factory
        self._path = Path(store_path) if store_path else None
        self._models: dict[str, ChatModel] = {}
        self._selected: str | None = None  # used when there is no file to save to
        self._lock = threading.Lock()

    @property
    def default(self) -> str:
        return self._available[0]

    def options(self) -> list[ModelOption]:
        return [self._describe(model_id) for model_id in self._available]

    def _describe(self, model_id: str) -> ModelOption:
        if not self._is_local(model_id):
            return describe(model_id)
        state = self._local.server.refresh()
        return ModelOption(
            id=model_id,
            label=self._local.label,
            description=_LOCAL_DESCRIPTION,
            kind="local",
            status=state.status.value,
            detail=state.detail,
        )

    def _is_local(self, model_id: str) -> bool:
        return self._local is not None and model_id == self._local.id

    def local_selected(self) -> bool:
        """Whether the next question will be answered by the local model."""
        return self._is_local(self.current())

    def current(self) -> str:
        """The selected model; the default if none is saved or it's no longer offered."""
        saved = self._read()
        return saved if saved in self._available else self.default

    def select(self, model_id: str) -> str:
        if model_id not in self._available:
            raise ValueError(f"'{model_id}' is not one of the available models.")
        with self._lock:
            self._selected = model_id
            if self._path is not None:
                payload = {"model": model_id, "updated_at": datetime.now(timezone.utc).isoformat()}
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_name(self._path.name + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, self._path)  # readers never see a half-written file
        logger.info("Chat model changed to %s", model_id)
        if self._local is not None:
            if self._is_local(model_id):
                self._local.server.start()  # loads in the background; retries after an error
            else:
                self._local.server.stop()
        return model_id

    def activate(self) -> None:
        """At startup: start loading the local model if it is the saved choice."""
        if self._local is not None and self._is_local(self.current()):
            self._local.server.start()

    def shutdown(self) -> None:
        """At shutdown: stop the local model's server."""
        if self._local is not None:
            self._local.server.stop()

    def llm(self) -> ChatModel:
        """The chat model to answer the next question with (created once per model).

        For the local model this waits until its server is ready, and raises
        ``ModelUnavailableError`` if it can't be started.
        """
        model_id = self.current()
        if self._is_local(model_id):
            self._local.server.wait_until_ready()
        with self._lock:
            if model_id not in self._models:
                self._models[model_id] = self._factory(model_id)
            return self._models[model_id]

    def _read(self) -> str | None:
        if self._path is None:
            return self._selected
        if not self._path.exists():
            return None
        try:
            return str(json.loads(self._path.read_text(encoding="utf-8"))["model"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("Could not read the model selection from %s: %s", self._path, exc)
            return None
