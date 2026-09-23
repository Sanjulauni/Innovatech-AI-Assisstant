"""The chat model the admin selected, from the models listed in ``GROQ_MODELS``.

The choice is saved to a small JSON file (``MODEL_SELECTION_FILE``) so it survives
restarts, and it takes effect on the next question without restarting the API.
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

from src.model_factory import ChatModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    description: str


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
    ) -> None:
        if not available:
            raise ValueError("At least one model must be available.")
        self._available = list(available)
        self._factory = factory
        self._path = Path(store_path) if store_path else None
        self._models: dict[str, ChatModel] = {}
        self._selected: str | None = None  # used when there is no file to save to
        self._lock = threading.Lock()

    @property
    def default(self) -> str:
        return self._available[0]

    def options(self) -> list[ModelOption]:
        return [describe(model_id) for model_id in self._available]

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
        return model_id

    def llm(self) -> ChatModel:
        """The chat model to answer the next question with (created once per model)."""
        model_id = self.current()
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
