"""Streamlit script used by ``tests/test_ui.py`` (not collected by pytest).

It renders one page against the real FastAPI app, wired with fake models and an
in-memory vector store. The test chooses the page and temp folder via session state.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.ui.components.admin_page import render_admin_page  # noqa: E402
from src.ui.components.chat_page import render_chat_page  # noqa: E402
from tests.ui_support import make_fake_api  # noqa: E402

if "_api" not in st.session_state:
    st.session_state["_api"] = make_fake_api(
        Path(st.session_state["_tmp"]), st.session_state.get("_responses", ["ok [1]"])
    )

api = st.session_state["_api"]
if st.session_state.get("_page") == "admin":
    render_admin_page(api)
else:
    render_chat_page(api)
