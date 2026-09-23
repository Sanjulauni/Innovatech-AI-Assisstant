"""Employee chat page (FR-26 – FR-28, FR-30, UC-01, UC-03)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.ui.api_client import ApiClient, ApiError

MESSAGES_KEY = "chat_messages"


def _format_source(source: dict[str, Any]) -> str:
    page = f", page {source['page']}" if source.get("page") is not None else ""
    return f"**[{source['index']}] {source['source']}{page}**"


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(_format_source(source))
            st.caption(source.get("snippet", ""))


def _history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Earlier messages in the shape the API expects (the server trims them)."""
    return [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]


def _answer(api: ApiClient, messages: list[dict[str, Any]], question: str) -> None:
    history = _history(messages)
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents…"):
                reply = api.chat(question, history)
        except ApiError as exc:
            # Keep the question out of the history so a retry doesn't send it twice.
            messages.pop()
            st.error(exc.message, icon=":material/error:")
            return
        st.markdown(reply["answer"])
        _render_sources(reply["sources"])

    messages.append(
        {"role": "assistant", "content": reply["answer"], "sources": reply["sources"]}
    )


def render_chat_page(api: ApiClient) -> None:
    st.title("InnovaTech Assistant")
    st.caption(
        "Ask about company policies, manuals and guidelines. "
        "Answers come from the company documents, with sources."
    )

    messages: list[dict[str, Any]] = st.session_state.setdefault(MESSAGES_KEY, [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question, e.g. How many days of annual leave do I get?")
    if question:
        _answer(api, messages, question)

    # Drawn last so it is enabled as soon as the first answer arrives.
    with st.sidebar:
        if st.button("Clear chat", icon=":material/delete_sweep:", disabled=not messages):
            messages.clear()
            st.rerun()
