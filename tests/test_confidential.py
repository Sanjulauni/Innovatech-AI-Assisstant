"""Tests for confidential documents: the store, and routing questions to the local model.

The key guarantee: text from a confidential document is never sent to a cloud model,
not even when the local model is missing or fails to start.
"""

import json
import time

import pytest
from langchain_core.documents import Document

from src.data_pipeline.confidential import ConfidentialStore
from src.data_pipeline.vector_store import SearchResult
from src.local_llm import ServerState, ServerStatus
from src.rag_engine.llm_chain import RAGChain, ServiceUnavailableError
from src.rag_engine.model_selector import LocalModel, ModelSelector
from src.rag_engine.models import Answer, ChatMessage, Route
from src.rag_engine.retriever import DocumentRetriever
from tests.fakes import RecordingChatModel, StubLocalServer

CLOUD = "openai/gpt-oss-120b"
LOCAL = "local:gemma"
SOP = "a" * 64  # doc_id of the confidential SOP
POLICY = "b" * 64  # doc_id of a normal document
# The fake embeddings aren't semantic, but identical text always matches best, so tests
# that need one specific document ask with its exact text.
SOP_TEXT = "Server restart SOP: stop the queue, then reboot."
POLICY_TEXT = "Annual leave is 14 days."


# --- Store ----------------------------------------------------------------------------------


def test_store_starts_empty_and_remembers_marks(tmp_path):
    path = tmp_path / "confidential.json"
    store = ConfidentialStore(path)
    assert store.ids() == frozenset()

    store.set(SOP, True)
    store.set(POLICY, True)
    store.set(POLICY, False)

    assert store.is_confidential(SOP) and not store.is_confidential(POLICY)
    assert ConfidentialStore(path).ids() == {SOP}  # survives a restart
    assert json.loads(path.read_text(encoding="utf-8"))["doc_ids"] == [SOP]


def test_store_works_in_memory():
    store = ConfidentialStore()
    store.set(SOP, True)
    assert store.ids() == {SOP}


@pytest.mark.parametrize("content", ["not json", "[]", '{"doc_ids": "x"}', '{"doc_ids": [1]}'])
def test_unreadable_store_refuses_to_start(tmp_path, content):
    path = tmp_path / "confidential.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="won't start"):
        ConfidentialStore(path)


# --- Routing ----------------------------------------------------------------------------------


@pytest.fixture
def docs(repository):
    """One confidential SOP and one normal policy in the vector store."""
    repository.add_document(SOP, [Document(SOP_TEXT, metadata={"source": "SOP.md"})])
    repository.add_document(POLICY, [Document(POLICY_TEXT, metadata={"source": "Leave.md"})])
    store = ConfidentialStore()
    store.set(SOP, True)
    return repository, store


class Models:
    """A selector whose cloud and local models are separate fakes, so tests can see which
    one answered, plus the stub server behind the local model."""

    def __init__(self, tmp_path, local=True, server=None, idle_seconds=None):
        self.cloud = RecordingChatModel(responses=["Cloud answer [1]."] * 5)
        self.local = RecordingChatModel(responses=["Local answer [1]."] * 5)
        self.server = server or StubLocalServer()
        self.selector = ModelSelector(
            [CLOUD],
            lambda model: self.local if model == LOCAL else self.cloud,
            tmp_path / "selection.json",
            local=LocalModel(LOCAL, "Gemma", self.server) if local else None,
            idle_seconds=idle_seconds,
        )


def make_chain(docs, llm, top_k=2):
    repository, store = docs
    return RAGChain(
        DocumentRetriever(repository, top_k=top_k),
        llm,
        history_limit=6,
        local_top_k=1,
        local_history_limit=2,
        confidential=store,
    )


def prompt_text(llm) -> str:
    return "\n".join(m.content for m in llm.prompts[-1])


def test_question_about_a_confidential_document_is_answered_locally(docs, tmp_path):
    models = Models(tmp_path)
    chain = make_chain(docs, models.selector)

    answer = chain.ask("How do I restart the server? What is the SOP?")

    assert answer.private is True
    assert answer.answer == "Local answer [1]."
    assert models.cloud.prompts == []  # the cloud model never saw it
    assert "Server restart SOP" in prompt_text(models.local)
    assert prompt_text(models.local).count("<document ") == 1  # local_top_k
    assert answer.sources[0].confidential is True
    assert "wait" in models.server.calls


def test_normal_questions_still_use_the_selected_cloud_model(docs, tmp_path):
    repository, store = docs
    store.set(SOP, False)
    models = Models(tmp_path)
    chain = make_chain(docs, models.selector)

    answer = chain.ask("How much annual leave?")

    assert answer.private is False
    assert answer.answer == "Cloud answer [1]."
    assert models.local.prompts == [] and models.server.calls == []
    assert all(not s.confidential for s in answer.sources)


def test_without_a_local_model_confidential_questions_are_refused(docs, tmp_path):
    models = Models(tmp_path, local=False)
    chain = make_chain(docs, models.selector)

    with pytest.raises(ServiceUnavailableError, match="no local model is set up"):
        chain.ask("What is the server restart SOP?")
    assert models.cloud.prompts == []


def test_if_the_local_model_fails_confidential_questions_are_refused(docs, tmp_path):
    server = StubLocalServer(error="The local model could not be started.")
    models = Models(tmp_path, server=server)
    chain = make_chain(docs, models.selector)

    with pytest.raises(ServiceUnavailableError, match="could not be started"):
        chain.ask("What is the server restart SOP?")
    assert models.cloud.prompts == []


def test_a_bare_model_never_gets_confidential_text(docs):
    cloud = RecordingChatModel(responses=["x"])
    with pytest.raises(ServiceUnavailableError):
        make_chain(docs, cloud).ask("What is the server restart SOP?")
    assert cloud.prompts == []


def test_stream_announces_the_private_route_first(docs, tmp_path):
    models = Models(tmp_path)
    chain = make_chain(docs, models.selector)

    route, *pieces, answer = list(chain.stream("What is the server restart SOP?"))

    assert route == Route(local=True, private=True)
    assert "".join(pieces) == "Local answer [1]."
    assert isinstance(answer, Answer) and answer.private


def test_when_the_local_model_is_selected_everything_is_local(docs, tmp_path):
    models = Models(tmp_path)
    models.selector.select(LOCAL)
    chain = make_chain(docs, models.selector)

    normal = chain.ask(POLICY_TEXT)  # the local model's top-k is 1: only the policy
    private = chain.ask(SOP_TEXT)

    assert (normal.private, private.private) == (False, True)
    assert models.cloud.prompts == []


def test_confidential_answers_are_left_out_of_cloud_history(docs, tmp_path):
    repository, store = docs
    store.set(SOP, False)  # so this question is answered by the cloud model
    models = Models(tmp_path)
    chain = make_chain(docs, models.selector, top_k=1)
    history = [
        ChatMessage("user", "What is the restart SOP?"),
        ChatMessage("assistant", "Stop the queue, then reboot [1].", confidential=True),
        ChatMessage("user", "Thanks. Unrelated: is there a gym?"),
        ChatMessage("assistant", "I couldn't find this information in the company documents."),
    ]

    chain.ask("How much annual leave?", history)

    sent = prompt_text(models.cloud)
    assert "Stop the queue" not in sent
    assert "What is the restart SOP?" not in sent  # its question is dropped too
    assert "is there a gym?" in sent


def test_confidential_history_is_kept_for_the_local_model(docs, tmp_path):
    models = Models(tmp_path)
    chain = make_chain(docs, models.selector)
    history = [
        ChatMessage("user", "What is the restart SOP?"),
        ChatMessage("assistant", "Stop the queue, then reboot [1].", confidential=True),
    ]

    chain.ask(SOP_TEXT, history)  # a question that matches the SOP itself

    assert "Stop the queue" in prompt_text(models.local)


def test_marks_apply_from_the_next_question(docs, tmp_path):
    repository, store = docs
    store.set(SOP, False)
    models = Models(tmp_path)
    chain = make_chain(docs, models.selector)
    assert chain.ask("What is the server restart SOP?").private is False

    store.set(SOP, True)

    assert chain.ask("What is the server restart SOP?").private is True


# --- Starting and stopping the local model for confidential questions -------------------


def test_local_model_started_on_demand_stops_after_idle_time(docs, tmp_path):
    models = Models(tmp_path, idle_seconds=0.2)
    chain = make_chain(docs, models.selector)

    chain.ask("What is the server restart SOP?")
    assert models.server.calls == ["wait"]

    deadline = time.monotonic() + 5
    while "stop" not in models.server.calls and time.monotonic() < deadline:
        time.sleep(0.05)
    assert models.server.calls == ["wait", "stop"]


def test_each_confidential_question_restarts_the_idle_countdown(docs, tmp_path):
    models = Models(tmp_path, idle_seconds=0.5)
    chain = make_chain(docs, models.selector)

    for _ in range(3):
        chain.ask("What is the server restart SOP?")
        time.sleep(0.3)
    assert "stop" not in models.server.calls  # never idle for 0.5 s in a row


def test_selected_local_model_is_not_stopped_when_idle(docs, tmp_path):
    models = Models(tmp_path, idle_seconds=0.1)
    chain = make_chain(docs, models.selector)
    chain.ask("What is the server restart SOP?")  # starts the countdown

    models.selector.select(LOCAL)
    time.sleep(0.3)

    assert "stop" not in models.server.calls


def test_reselecting_a_cloud_model_keeps_an_on_demand_local_server(tmp_path):
    models = Models(tmp_path)
    models.selector.select(CLOUD)  # was already the cloud model
    assert models.server.calls == []

    models.selector.select(LOCAL)
    models.selector.select(CLOUD)  # switching away from the local model stops it
    assert models.server.calls == ["start", "stop"]


def test_running_local_model_shows_as_running_in_the_model_list(tmp_path):
    server = StubLocalServer(ServerState(ServerStatus.READY))
    models = Models(tmp_path, server=server)
    assert models.selector.options()[-1].status == "ready"


# --- Only questions that are themselves about confidential documents go local ------------


class ScriptedRetriever:
    """Returns set results for a search with history, the question alone, or with
    confidential documents excluded, and records which searches were made."""

    def __init__(self, with_history, alone, excluded):
        self.results = {"with_history": with_history, "alone": alone, "excluded": excluded}
        self.calls = []

    def retrieve(self, question, history=None, k=None, exclude_doc_ids=()):
        kind = "excluded" if exclude_doc_ids else "with_history" if history else "alone"
        self.calls.append(kind)
        return self.results[kind]


def hit(doc_id, text):
    return SearchResult(Document(text, metadata={"doc_id": doc_id, "source": f"{doc_id[:4]}.md"}),
                        0.8)


CANE = hit(SOP, "The smart cane detects obstacles with an ultrasonic sensor.")
DRESS = hit(POLICY, "Dress code: smart casual.")
AFTER_CANE = [ChatMessage("user", "How does the smart cane detect obstacles?"),
              ChatMessage("assistant", "With an ultrasonic sensor [1].", confidential=True)]


def scripted_chain(retriever, selector):
    store = ConfidentialStore()
    store.set(SOP, True)
    return RAGChain(retriever, selector, history_limit=6, local_top_k=1,
                    local_history_limit=2, confidential=store)


def test_new_topic_after_a_confidential_question_uses_the_cloud_model(tmp_path):
    # The previous (confidential) question drags the cane document into the search,
    # but the new question on its own doesn't match it: it's a change of topic.
    retriever = ScriptedRetriever(with_history=[CANE, DRESS], alone=[DRESS],
                                  excluded=[DRESS])
    models = Models(tmp_path)

    answer = scripted_chain(retriever, models.selector).ask("What is the dress code?", AFTER_CANE)

    assert answer.private is False
    assert answer.answer == "Cloud answer [1]."
    assert models.local.prompts == [] and models.server.calls == []
    sent = prompt_text(models.cloud)
    assert "Dress code" in sent
    assert "ultrasonic" not in sent  # neither the excerpt nor the earlier answer
    assert retriever.calls == ["with_history", "alone", "excluded"]


def test_follow_up_that_matches_the_confidential_document_stays_local(tmp_path):
    retriever = ScriptedRetriever(with_history=[CANE], alone=[CANE], excluded=[DRESS])
    models = Models(tmp_path)

    answer = scripted_chain(retriever, models.selector).ask("What sensor does it use?", AFTER_CANE)

    assert answer.private is True
    assert models.cloud.prompts == []
    assert retriever.calls == ["with_history", "alone"]  # nothing was excluded


def test_no_extra_search_when_no_confidential_document_is_found(tmp_path):
    retriever = ScriptedRetriever(with_history=[DRESS], alone=[CANE], excluded=[])
    models = Models(tmp_path)

    scripted_chain(retriever, models.selector).ask("What is the dress code?", AFTER_CANE)

    assert retriever.calls == ["with_history"]


def test_every_question_is_routed_on_its_own(tmp_path):
    # Confidential, then normal, then confidential again: the model follows each question.
    models = Models(tmp_path)
    confidential = ScriptedRetriever(with_history=[CANE], alone=[CANE], excluded=[DRESS])
    normal = ScriptedRetriever(with_history=[CANE, DRESS], alone=[DRESS], excluded=[DRESS])

    first = scripted_chain(confidential, models.selector).ask("How does the cane work?")
    second = scripted_chain(normal, models.selector).ask("And the dress code?", AFTER_CANE)
    third = scripted_chain(confidential, models.selector).ask("And its sensor?", AFTER_CANE)

    assert [first.private, second.private, third.private] == [True, False, True]
    assert len(models.local.prompts) == 2 and len(models.cloud.prompts) == 1


def test_confidential_excerpt_below_the_local_models_view_does_not_route_locally(tmp_path):
    # The cloud search found the cane document only in 2nd place, but the local model
    # would read just the top 1 (DRESS): answering locally would be slow and pointless.
    retriever = ScriptedRetriever(with_history=[DRESS, CANE], alone=[DRESS, CANE],
                                  excluded=[DRESS])
    models = Models(tmp_path)

    answer = scripted_chain(retriever, models.selector).ask("What is the dress code?")

    assert answer.private is False
    assert models.local.prompts == []
    assert "ultrasonic" not in prompt_text(models.cloud)
    assert retriever.calls == ["alone", "excluded"]  # no history: the first search is "alone"
