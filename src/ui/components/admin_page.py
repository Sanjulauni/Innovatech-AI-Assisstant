"""Password-protected admin page: upload and manage documents, edit agent instructions.

The password is kept in this browser session only and sent to the API with every
admin request; the API decides whether it is correct.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from src.rag_engine.instructions import MAX_INSTRUCTIONS_LENGTH
from src.ui.api_client import ApiClient, ApiError

PASSWORD_KEY = "admin_password"
UPLOADER_KEY = "admin_uploader_version"
FLASH_KEY = "admin_flash"
UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]


def _logout() -> None:
    st.session_state.pop(PASSWORD_KEY, None)


def _handle_error(exc: ApiError) -> None:
    """Show an API error; a 401 means the password changed, so log out."""
    if exc.status_code == 401:
        _logout()
        st.session_state[FLASH_KEY] = [("error", "Your session expired. Please log in again.")]
        st.rerun()
    st.error(exc.message, icon=":material/error:")


def _format_time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def _show_flash() -> None:
    for kind, text in st.session_state.pop(FLASH_KEY, []):
        getattr(st, kind)(text)


# --- Login ---------------------------------------------------------------------------


def _render_login(api: ApiClient) -> None:
    st.title("Admin")
    st.caption("Upload company documents and set how the assistant answers.")
    _show_flash()
    with st.form("admin_login"):
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if not submitted:
        return
    if not password:
        st.warning("Enter the admin password.")
        return
    try:
        api.login(password)
    except ApiError as exc:
        st.error(exc.message, icon=":material/error:")
        return
    st.session_state[PASSWORD_KEY] = password
    st.rerun()


# --- Documents -----------------------------------------------------------------------


def _upload_files(api: ApiClient, password: str, files: list[Any]) -> None:
    results = []
    progress = st.progress(0.0, text="Uploading…")
    for number, file in enumerate(files, start=1):
        progress.progress((number - 1) / len(files), text=f"Indexing {file.name}…")
        try:
            result = api.upload_document(password, file.name, file.getvalue())
        except ApiError as exc:
            if exc.status_code == 401:
                _handle_error(exc)
            results.append(("error", f"{file.name}: {exc.message}"))
        else:
            kind = "success" if result["status"] == "ingested" else "info"
            results.append((kind, result["message"]))
    progress.empty()
    st.session_state[FLASH_KEY] = results
    # A new key empties the file picker after the upload.
    st.session_state[UPLOADER_KEY] = st.session_state.get(UPLOADER_KEY, 0) + 1
    st.rerun()


def _render_documents(api: ApiClient, password: str) -> None:
    st.subheader("Upload documents")
    files = st.file_uploader(
        "PDF, Word, text or Markdown files",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get(UPLOADER_KEY, 0)}",
    )
    if st.button("Upload and index", type="primary", disabled=not files):
        _upload_files(api, password, files)
    _show_flash()

    st.subheader("Knowledge base")
    try:
        documents = api.list_documents(password)
    except ApiError as exc:
        _handle_error(exc)
        return
    if not documents:
        st.info("No documents yet. Upload one above to get started.")
        return

    st.caption(f"{len(documents)} document(s)")
    for document in documents:
        name_col, info_col, action_col = st.columns([5, 3, 2], vertical_alignment="center")
        name_col.markdown(f"**{document['source']}**")
        info_col.caption(
            f"{document['chunk_count']} chunks · added {_format_time(document['ingested_at'])}"
        )
        with action_col.popover("Delete", icon=":material/delete:"):
            st.write(f"Remove **{document['source']}** from the knowledge base?")
            if st.button("Delete", key=f"delete_{document['doc_id']}", type="primary"):
                try:
                    api.delete_document(password, document["doc_id"])
                except ApiError as exc:
                    _handle_error(exc)
                    return
                st.session_state[FLASH_KEY] = [("success", f"Deleted {document['source']}.")]
                st.rerun()


# --- Instructions --------------------------------------------------------------------


def _render_instructions(api: ApiClient, password: str) -> None:
    st.subheader("Agent instructions")
    st.caption(
        "Tell the assistant how to answer: tone, formatting, who to contact for escalations. "
        "These can't make it answer beyond the documents or skip citations."
    )
    try:
        current = api.get_instructions(password)
    except ApiError as exc:
        _handle_error(exc)
        return

    with st.form("instructions_form"):
        text = st.text_area(
            "Instructions",
            value=current["text"],
            height=250,
            max_chars=MAX_INSTRUCTIONS_LENGTH,
            placeholder=(
                "Example:\n- Use a friendly, professional tone.\n"
                "- For payroll questions, suggest contacting payroll@innovatech.example."
            ),
        )
        saved = st.form_submit_button("Save instructions", type="primary")
    st.caption(f"Last updated: {_format_time(current.get('updated_at'))}")

    if saved:
        try:
            api.update_instructions(password, text)
        except ApiError as exc:
            _handle_error(exc)
            return
        st.success("Instructions saved. They apply to the next question.")


# --- Page ----------------------------------------------------------------------------


def render_admin_page(api: ApiClient) -> None:
    password = st.session_state.get(PASSWORD_KEY)
    if not password:
        _render_login(api)
        return

    with st.sidebar:
        if st.button("Log out", icon=":material/logout:"):
            _logout()
            st.rerun()

    st.title("Admin")
    documents_tab, instructions_tab = st.tabs(["Documents", "Instructions"])
    with documents_tab:
        _render_documents(api, password)
    with instructions_tab:
        _render_instructions(api, password)
