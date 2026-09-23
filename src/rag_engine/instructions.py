"""Stores the administrator's instructions for the agent (tone, rules, escalation contacts).

The instructions live in a small JSON file (``INSTRUCTIONS_FILE``). They are added to
the prompt below the fixed grounding rules, so they can shape answers but never
override those rules (see ``prompts.py``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

MAX_INSTRUCTIONS_LENGTH = 4000


@dataclass(frozen=True)
class AgentInstructions:
    text: str = ""
    updated_at: datetime | None = None


class InstructionsStore:
    """Reads and writes the admin's instructions as JSON, safely across threads."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> InstructionsStore:
        settings = settings or get_settings()
        return cls(settings.instructions_file)

    def get(self) -> AgentInstructions:
        """Return the saved instructions, or empty ones if none are saved yet."""
        with self._lock:
            if not self._path.exists():
                return AgentInstructions()
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                updated_at = data.get("updated_at")
                return AgentInstructions(
                    text=str(data.get("text", "")),
                    updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
                )
            except (OSError, ValueError, AttributeError) as exc:
                # A broken file must not take the chat down; the admin can re-save.
                logger.error("Could not read agent instructions from %s: %s", self._path, exc)
                return AgentInstructions()

    def update(self, text: str) -> AgentInstructions:
        """Replace the instructions and record when they changed."""
        text = text.strip()
        if len(text) > MAX_INSTRUCTIONS_LENGTH:
            raise ValueError(
                f"Instructions are too long ({len(text)} characters, "
                f"maximum {MAX_INSTRUCTIONS_LENGTH})."
            )
        instructions = AgentInstructions(text=text, updated_at=datetime.now(timezone.utc))
        payload = {"text": instructions.text, "updated_at": instructions.updated_at.isoformat()}

        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file and swap it in, so readers never see a half-written file.
            tmp_path = self._path.with_name(self._path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._path)
        logger.info("Agent instructions updated (%d characters)", len(text))
        return instructions
