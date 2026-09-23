"""Streamlit entry point: ``streamlit run src/ui/app.py``.

Two pages: the employee chat and the password-protected admin page. Both talk to
the FastAPI backend at ``API_BASE_URL``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts this file's folder on sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.config import UISettings  # noqa: E402
from src.ui.api_client import ApiClient  # noqa: E402
from src.ui.components.admin_page import render_admin_page  # noqa: E402
from src.ui.components.chat_page import render_chat_page  # noqa: E402


@st.cache_resource
def get_api() -> ApiClient:
    settings = UISettings()
    return ApiClient(settings.api_base_url, timeout=settings.request_timeout_seconds)


def chat() -> None:
    render_chat_page(get_api())


def admin() -> None:
    render_admin_page(get_api())


def main() -> None:
    st.set_page_config(page_title="InnovaTech Assistant", page_icon=":material/forum:")
    page = st.navigation(
        [
            st.Page(chat, title="Chat", icon=":material/chat:", default=True),
            st.Page(admin, title="Admin", icon=":material/admin_panel_settings:"),
        ]
    )
    page.run()


main()
