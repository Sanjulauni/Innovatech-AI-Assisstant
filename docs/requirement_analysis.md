# 📋 Requirement Analysis — InnovaTech AI Internal Document Assistant

| Item | Detail |
|---|---|
| **Project** | InnovaTech AI — Internal Document Assistant (RAG Application) |
| **Course** | IT 4005 Advanced Software Engineering |
| **Document type** | Software Requirements Specification (SRS) |
| **Version** | 2.0 (as built) |
| **Status** | Updated to match the implemented system |

### Revision History

| Version | Summary |
|---|---|
| 1.0 (Draft) | Initial requirements: fully local LLM via Ollama, Streamlit UI. |
| 2.0 (as built) | Aligned with the implementation. Answers come from a Groq cloud model by default, or from a local GGUF model (llama.cpp) chosen by the admin; documents marked **confidential** are only ever answered by the local model. The UI is a React web app. Endpoints, technologies, statuses and traceability updated; FR-34 – FR-45 and NFR-23 – NFR-25 added. |

### Status Legend

| Status | Meaning |
|---|---|
| ✅ Implemented | Built as specified. |
| 🔄 Changed | Built, but differently from v1.0; the requirement text describes what was built. |
| ⚠️ Partial | Partly built or not formally verified. |
| ❌ Not implemented | Not built in this release. |

---

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the InnovaTech AI Internal Document Assistant and records how each one was implemented. It is the reference for design, implementation, testing and acceptance.

### 1.2 Scope
The system is a self-hosted **Retrieval-Augmented Generation (RAG)** assistant. It:

- ingests Innovatech's internal documents (manuals, technical specs, guidelines, policies, SOPs),
- indexes them in a local vector database, using a local embedding model,
- answers employee questions in natural language, citing the source documents,
- writes answers with a chat model chosen by the administrator: a **Groq** cloud model (default), or a **local model** running on the company's own machine,
- never sends documents marked **confidential** to a cloud model: questions that need them are answered by the local model, or refused.

Full documents are never uploaded to an external service. With a Groq model, only the question and the few most relevant excerpts are sent to write the answer.

### 1.3 Definitions & Acronyms

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation: the system retrieves relevant document passages and gives them to an LLM as context for its answer. |
| **LLM** | Large Language Model. |
| **Embedding** | A numeric vector that represents the meaning of a piece of text. |
| **Chunk** | A small section of a document, sized for embedding and retrieval. |
| **Vector store** | A database that stores embeddings and runs similarity searches over them (here, ChromaDB). |
| **Ingestion** | Loading, splitting, embedding and storing documents. |
| **Hallucination** | An LLM answer that is not supported by the supplied context. |
| **Cloud model** | A chat model hosted by Groq and called over its API. |
| **Local model** | A GGUF chat model run on the API's machine by llama.cpp's `llama-server`. |
| **Confidential document** | A document the administrator has marked so that only the local model may read it. |
| **GGUF** | The model file format used by llama.cpp. |

---

## 2. Problem Statement

Innovatech hires new employees regularly. New hires must absorb a large volume of internal documentation, which slows onboarding and pulls senior staff away from their work to answer repeated questions.

Company security and privacy policy **prohibits uploading proprietary documents to public LLMs** such as ChatGPT. Employees therefore cannot use widely available AI tools to search or summarize internal knowledge.

**Goal:** give employees an AI assistant that answers questions accurately from internal documentation, never uploads documents to an external service, and keeps the most sensitive documents entirely on the company's own machines.

---

## 3. Stakeholders & User Roles

| Stakeholder | Role / Interest |
|---|---|
| **Employee (End User)** | Asks questions about internal documentation and needs fast, accurate, sourced answers. New hires are the primary users. |
| **Knowledge Administrator** | Uploads and removes documents, marks confidential ones, chooses the chat model, and sets the assistant's instructions and upload limit. |
| **IT / Security Team** | Requires that confidential data never leaves the company's machines and that the system meets company security policy. |
| **Management** | Wants shorter onboarding time and less load on senior staff. |
| **Development Team** | Builds and maintains the system. |

---

## 4. Functional Requirements

Priority uses **MoSCoW**: **M**ust, **S**hould, **C**ould, **W**on't (this release).

### 4.1 Document Ingestion

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-01 | The system shall load documents in **PDF, DOCX, TXT and Markdown** formats. | M | ✅ |
| FR-02 | The system shall choose the correct loader for each file type through a **Loader Factory** (`DocumentLoaderFactory`), so new formats can be added without changing existing code. | M | ✅ |
| FR-03 | The system shall split documents into overlapping chunks, with chunk size and overlap set in configuration (default 1,000 / 200 characters). | M | ✅ |
| FR-04 | The system shall generate an embedding for each chunk using a **locally hosted** embedding model (FastEmbed `BAAI/bge-small-en-v1.5`, CPU). | M | ✅ |
| FR-05 | The system shall store chunks, embeddings and metadata (document ID, source name, file name, page number, chunk index, start offset, ingestion timestamp) in ChromaDB, persisted on local disk. | M | ✅ |
| FR-06 | The system shall index every supported file saved in `data/raw/` that is not yet in the knowledge base ("Re-index saved files"). | M | ✅ |
| FR-07 | The system shall let an administrator upload documents through the web UI (several at once, drag and drop). | S | ✅ |
| FR-08 | The system shall detect documents that are already indexed (by SHA-256 content hash) and skip them instead of creating duplicates. A changed file has a new hash and is indexed as a new document. | S | 🔄 |
| FR-09 | The system shall let an administrator delete a document, all of its chunks and its saved file. | S | ✅ |
| FR-10 | The system shall report unsupported or corrupt files with a clear error and continue processing the remaining files. | M | ✅ |

### 4.2 Question Answering (RAG)

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-11 | The system shall accept a natural-language question from the user (up to 2,000 characters). | M | ✅ |
| FR-12 | The system shall retrieve the top-*k* most relevant chunks from the vector store (*k* configurable: default 4 for cloud models, 2 for the local model). | M | ✅ |
| FR-13 | The system shall generate an answer with the **chat model selected by the administrator** (a Groq cloud model or the local model), using only the retrieved chunks as context. | M | 🔄 |
| FR-14 | The system shall return the **source citations** (document name, page and an excerpt) used to produce each answer, numbered as cited in the text (`[1]`). | M | ✅ |
| FR-15 | When the retrieved context does not contain the answer, the system shall say it could not find the information rather than invent one. | M | ✅ |
| FR-16 | The system shall support follow-up questions within a chat session (the last 6 messages for cloud models, 2 for the local model). | S | ✅ |
| FR-17 | The system shall summarize a specified document on request. | C | ❌ |
| FR-18 | The system shall stream the answer to the UI as it is generated. | C | ✅ |

### 4.3 REST API (FastAPI)

All endpoints are served under the `/api` prefix.

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-19 | `POST /api/chat` and `POST /api/chat/stream`: accept a question and the earlier messages of the conversation; return the answer and its sources (the stream sends NDJSON `route`, `token`, `done` and `error` events). | M | 🔄 |
| FR-20 | `POST /api/admin/documents`: upload a document and index it, optionally marking it confidential. | S | 🔄 |
| FR-21 | `GET /api/admin/documents`: list the indexed documents with their metadata and confidential status. | S | 🔄 |
| FR-22 | `DELETE /api/admin/documents/{doc_id}`: remove a document from the index. | S | 🔄 |
| FR-23 | `GET /api/health`: report the status of the API and the vector store, the number of documents, and the current chat model. | M | ✅ |
| FR-24 | The API shall validate every request and response with **Pydantic schemas** and return meaningful HTTP error codes. | M | ✅ |
| FR-25 | The API shall publish auto-generated OpenAPI (Swagger) documentation at `/docs`. | M | ✅ |

### 4.4 User Interface (React web app)

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-26 | The UI shall provide a chat interface for asking questions and viewing answers. | M | 🔄 |
| FR-27 | The UI shall display the source citations with each answer; clicking a `[1]` chip shows that source. | M | ✅ |
| FR-28 | The UI shall keep the chat for the current browser tab (surviving a page reload), allow a new chat to be started, and let the user stop an answer or retry a failed one. | S | ✅ |
| FR-29 | The UI shall provide an admin panel for managing documents, the chat model and the assistant's instructions. | S | 🔄 |
| FR-30 | The UI shall show a progress indicator while an answer is being generated, with a note when the slower local model is answering. | S | ✅ |

### 4.5 Access Control

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-31 | The system shall separate **Employee** and **Administrator** permissions; only administrators can upload, delete or mark documents, or change settings. | S | ✅ |
| FR-32 | Protected endpoints shall require the admin password: a signed, expiring session cookie from `POST /api/admin/login` (web app), or an `X-Admin-Password` header (scripts). | S | 🔄 |
| FR-33 | Integration with the corporate SSO / LDAP directory. | W | ❌ |

### 4.6 Configuration & Model Providers

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-34 | All settings shall be read from environment variables or a `.env` file, with working defaults for everything except the Groq API key. | M | ✅ |
| FR-35 | Invalid configuration (e.g. a missing Groq key, chunk overlap ≥ chunk size, only one of the two local-model paths) shall stop the API at startup with a clear message. | M | ✅ |
| FR-36 | Each embedding model shall use its own vector-store collection, so changing the model never mixes incompatible vectors; saved files can then be re-indexed (FR-06). | S | ✅ |
| FR-37 | Chat and embedding providers shall be created through factories (`LLMFactory`, `EmbeddingFactory`), so adding a provider means adding one builder. | S | ✅ |

### 4.7 Administration

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-38 | The administrator shall choose the chat model from the configured Groq models and the local model; the choice is saved and applies from the next question without a restart. | S | ✅ |
| FR-39 | The administrator shall write instructions for the assistant (tone, format, escalation contacts, up to 4,000 characters). They are added to the prompt but cannot override the grounding and safety rules. | S | ✅ |
| FR-40 | The administrator shall set the maximum upload size, between 1 MB and a configured ceiling (default 20 MB, ceiling 200 MB); it applies to the next upload. | C | ✅ |

### 4.8 Local Model & Confidential Documents

| ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-41 | When `LOCAL_LLM_SERVER` and `LOCAL_LLM_MODEL` are configured, the system shall offer a local GGUF model. The API starts `llama-server` when it is needed and stops it when another model is selected (or after 15 idle minutes if it was started only for confidential questions), and restarts it after a crash. | S | ✅ |
| FR-42 | The admin panel shall show the local model's state (stopped, loading, running, or the reason it failed) and let the administrator retry after an error. | S | ✅ |
| FR-43 | The administrator shall mark documents as **confidential**, when uploading or later from the document list. | S | ✅ |
| FR-44 | If any retrieved chunk comes from a confidential document, the question shall be answered by the local model, whichever model is selected; if no local model is available, the question shall be refused with a clear message. | S | ✅ |
| FR-45 | Answers based on confidential documents shall be marked as answered privately, and their sources shown as confidential. | C | ✅ |

---

## 5. Non-Functional Requirements

### 5.1 Security & Privacy

| ID | Requirement | Status |
|---|---|---|
| NFR-01 | **Full documents shall never be sent to an external AI service.** Documents, embeddings and the vector store stay on the company's machine. With a Groq model, only the question, recent chat messages and the top retrieved excerpts are sent to write the answer. Confidential documents are covered by NFR-23. | 🔄 |
| NFR-02 | Secrets (API keys, admin password) shall be loaded from environment variables (`.env`) and never committed to source control. | ✅ |
| NFR-03 | Uploaded files shall be checked for allowed extension and maximum size (default 20 MB, adjustable by the administrator up to the configured ceiling) before processing. | ✅ |
| NFR-04 | The system shall defend against prompt injection. Retrieved context shall be clearly separated from system instructions in the prompt, and document text is treated as data, never as instructions. | ✅ |
| NFR-05 | Logs shall not record full document contents or sensitive user data. | ✅ |
| NFR-23 | **Confidential document text shall never be sent to an external AI service**, including through the chat history of later questions. If the local model is unavailable, such questions are refused. If the list of confidential documents cannot be read, the API shall refuse to start. | ✅ |
| NFR-24 | Admin sessions shall use a signed, expiring (default 8 hours) HttpOnly, SameSite=strict cookie; a client is blocked after 5 wrong passwords within a minute; passwords are compared in constant time. | ✅ |
| NFR-25 | The local model's server shall listen on `127.0.0.1` only, so it is not reachable from the network. | ✅ |

### 5.2 Performance

| ID | Requirement | Status |
|---|---|---|
| NFR-06 | Retrieval (vector search) shall complete in **< 1 second** for a knowledge base of up to 10,000 chunks. | ⚠️ Not formally measured |
| NFR-07 | An end-to-end answer shall be returned in **< 15 seconds** on the reference hardware (8-core CPU, 16 GB RAM, optional GPU). | ⚠️ See note |
| NFR-08 | The system shall support at least **10 concurrent users** without errors. | ⚠️ Not load-tested |

> **Note on NFR-07.** With Groq models, answers stream back quickly but latency was not formally measured. The local model was measured on the development laptop, which is below the reference hardware (4-core Ryzen 3 PRO 2300U, 6.9 GB RAM, model on a USB drive): about 45 s to load, then about 45 s to the first words and 80 s for a full answer from real documents. To reduce this, the local model gets a smaller prompt (2 excerpts, 2 earlier messages) and runs with `--reasoning off` and `--parallel 1`. Answers stream as they are written (FR-18), and the chat tells the user when the local model is answering.

### 5.3 Accuracy & Quality

| ID | Requirement | Status |
|---|---|---|
| NFR-09 | At least **80%** of the questions in a curated evaluation set shall receive answers judged correct and grounded in the sources. | ❌ No evaluation set yet |
| NFR-10 | Every factual answer shall include at least one source citation. The prompt requires citations; if the model forgets them, all retrieved sources are shown so the answer can still be checked. | ⚠️ Partial |

### 5.4 Usability

| ID | Requirement | Status |
|---|---|---|
| NFR-11 | A new employee shall be able to ask a first question with no training, in under 1 minute (the empty chat offers example questions). | ✅ |
| NFR-12 | Error messages shall be in plain language and suggest what to do next. | ✅ |

### 5.5 Maintainability & Code Quality

| ID | Requirement | Status |
|---|---|---|
| NFR-13 | The code shall follow **Object-Oriented** design and a layered architecture (UI → API → RAG Engine → Data Pipeline). | ✅ |
| NFR-14 | The code shall follow PEP 8, include type hints, and pass linting (`ruff`; the frontend passes ESLint and the TypeScript compiler). | ✅ |
| NFR-15 | Configurable values (model names, chunk size, *k*, paths, limits) shall be kept in `src/config.py` / `.env`, never hard-coded. | ✅ |
| NFR-16 | The LLM and embedding providers shall be swappable through configuration without changing business logic. | ✅ |

### 5.6 Reliability & Testability

| ID | Requirement | Status |
|---|---|---|
| NFR-17 | Unit and integration tests shall cover at least **70%** of the `src/` code. | ✅ 98% (327 backend tests; plus 52 frontend tests) |
| NFR-18 | The CI pipeline (GitHub Actions) shall run linting and tests on every push and pull request. | ❌ Workflow file exists but is empty |
| NFR-19 | The vector store shall persist across application restarts. | ✅ |
| NFR-20 | If a model is unavailable, the system shall fail gracefully with a clear error instead of crashing (rate limits, overload, a local model that cannot start). | ✅ |

### 5.7 Portability

| ID | Requirement | Status |
|---|---|---|
| NFR-21 | The system shall run on Windows, macOS and Linux with Python 3.10+ and Node.js 20.19+. | ⚠️ Developed and tested on Windows only |
| NFR-22 | Setup shall need only `pip install -r requirements.txt`, `npm install` in `frontend/`, and a `.env` file, as documented in the README. | 🔄 No `make` targets (the `Makefile` is empty) |

---

## 6. Use Cases

### UC-01 — Ask a Question
| Field | Description |
|---|---|
| **Actor** | Employee |
| **Precondition** | At least one document has been ingested. |
| **Main flow** | 1. The employee types a question in the chat. 2. The web app sends `POST /api/chat/stream` with the question and earlier messages. 3. The system embeds the question locally and retrieves the top-*k* chunks. 4. The system decides which model answers: the selected one, or the local model if any chunk is from a confidential document, and sends a `route` event. 5. The system builds a prompt from the rules, admin instructions, chunks and history, and calls the model. 6. The answer streams back and ends with its sources. 7. The web app shows the answer with clickable citations. |
| **Alternate flow** | 3a. Nothing is indexed → the system replies that the information is not available, without calling a model. 5a. The chunks don't answer the question → the model replies that it couldn't find the information. |
| **Exception flow** | 4a. A confidential document is involved but no local model is available → the question is refused with a clear message. 5b. The model is unavailable or rate-limited → an `error` event is sent and the chat offers a retry. |
| **Postcondition** | The question and answer appear in the chat, which is kept for the browser tab. |

### UC-02 — Ingest Documents
| Field | Description |
|---|---|
| **Actor** | Knowledge Administrator |
| **Precondition** | The administrator is signed in. |
| **Main flow** | 1. The admin uploads files in the admin panel, optionally ticking *Confidential* (or clicks *Re-index saved files* for files in `data/raw/`). 2. The system validates type and size. 3. The Loader Factory selects a loader for each file. 4. The documents are split into chunks. 5. The chunks are embedded and stored in ChromaDB with metadata. 6. The system reports the result for each file. |
| **Alternate flow** | 2a. Unsupported, oversized or corrupt file → it is rejected with a reason, and the rest continue. 3a. Same content already indexed → it is skipped as a duplicate. |
| **Postcondition** | The new content can be retrieved in questions. |

### UC-03 — View Answer Sources
| Field | Description |
|---|---|
| **Actor** | Employee |
| **Main flow** | 1. The employee receives an answer. 2. The employee clicks a `[1]` citation or expands the sources. 3. The system shows the document name, page and an excerpt of each chunk used, marking confidential ones. |

### UC-04 — Manage Knowledge Base
| Field | Description |
|---|---|
| **Actor** | Knowledge Administrator |
| **Main flow** | 1. The admin opens the document list. 2. The admin deletes a document, or marks or unmarks it as confidential. 3. The system removes the document's chunks and file, or updates its confidential status for the next question. |

### UC-05 — Configure the Assistant
| Field | Description |
|---|---|
| **Actor** | Knowledge Administrator |
| **Main flow** | 1. The admin selects a chat model (a Groq model or the local model). 2. If it is the local model, the system starts it and shows *Loading* until it is ready. 3. The admin edits the assistant's instructions or the maximum upload size. 4. Changes apply from the next question or upload. |
| **Exception flow** | 2a. The local model cannot start → the model list shows the reason; the admin fixes it and clicks the model to retry. |

### Use Case Overview

```text
                ┌──────────────────────────────────────────┐
                │         InnovaTech AI Assistant          │
  Employee ─────┼──► (UC-01 Ask a Question)                │
      │         │           │  «include»                  │
      └─────────┼──► (UC-03 View Answer Sources)           │
                │                                          │
  Knowledge ────┼──► (UC-02 Ingest Documents)              │
  Administrator │                                          │
      ├─────────┼──► (UC-04 Manage Knowledge Base)         │
      └─────────┼──► (UC-05 Configure the Assistant)       │
                └──────────────────────────────────────────┘
```

---

## 7. Technology Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+, TypeScript | Mature AI/ML ecosystem; a typed front end. |
| RAG orchestration | LangChain | Standard abstractions for loaders, splitters, chat models and embeddings. |
| Vector store | ChromaDB (persistent, local) | Lightweight, embedded and free; no external server needed. |
| Chat model (cloud) | Groq free tier: `openai/gpt-oss-120b` (default), `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` | Fast, free hosted inference; no GPU needed; the admin can switch models if one is rate-limited. |
| Chat model (local) | llama.cpp `llama-server` with a GGUF model (e.g. Gemma 4 E2B, 4-bit), via LangChain's OpenAI-compatible client | Runs on the company's machine for confidential documents; reads the GGUF file directly and is started and stopped by the API. |
| Embedding model | FastEmbed `BAAI/bge-small-en-v1.5` | Runs locally on the CPU (about 70 MB), works offline after the first download. |
| Backend API | FastAPI + Uvicorn | Pydantic validation, streaming responses, automatic OpenAPI docs, dependency injection. |
| Frontend | React 19, Vite, Tailwind CSS, TanStack Query | A responsive chat and admin UI with cached server state. |
| Testing | pytest, pytest-cov, FastAPI `TestClient`; Vitest + Testing Library | Unit and integration tests with fake models, so no API key or model is needed. |
| Code quality | ruff; ESLint, TypeScript | Linting and type checking. |
| CI | GitHub Actions | Planned; the workflow is not configured yet (NFR-18). |

> **Why this changed from v1.0.** Version 1.0 planned a fully local stack (Ollama with `llama3.1:8b` and `nomic-embed-text`). A local 8B model is too slow on the available CPU-only hardware, so the implementation keeps retrieval local (FastEmbed and ChromaDB) and uses Groq's free tier to write answers. A small local model (llama.cpp) was then added for documents that must never leave the machine. Per NFR-16, models are chosen through `.env` configuration.

---

## 8. Design Patterns & Architecture Mapping

| Pattern | Where | Purpose |
|---|---|---|
| **Factory** | `data_pipeline/document_loaders.py` (`DocumentLoaderFactory`), `model_factory.py` (`LLMFactory`, `EmbeddingFactory`) | Choose a loader by file extension, or a model by provider; new formats and providers need no changes to existing code (Open/Closed Principle). |
| **Strategy** | `rag_engine/model_selector.py`, the LangChain chat-model and `Embeddings` interfaces | Swap the chat model (cloud or local) per question without changing the pipeline. |
| **Singleton** | `config.py` (`get_settings`) | One shared, validated configuration object. |
| **Repository** | `data_pipeline/vector_store.py` (`VectorStoreRepository`) | Hide ChromaDB behind a clean interface (add, search, list, delete). |
| **Facade** | `rag_engine/llm_chain.py` (`RAGChain.ask` / `stream`) | A single entry point over retrieval, routing, prompting and generation. |
| **Dependency Injection** | `api/dependencies.py`, `api/routes.py` (FastAPI `Depends`) | Decouple endpoints from concrete services; tests inject fake models and an in-memory store. |

### Layered Architecture

| Layer | Package | Responsibility |
|---|---|---|
| Presentation | `frontend/` | React chat and admin web app. |
| API | `src/api/` | REST endpoints, validation, admin sessions, error handling. |
| Business logic | `src/rag_engine/` | Retrieval, confidential routing, prompt construction, model selection, answer generation. |
| Data | `src/data_pipeline/` | Loading, chunking, embedding, vector store, confidential marks, upload limit. |
| Model runtime | `src/model_factory.py`, `src/local_llm.py` | Creating chat and embedding models; starting and stopping the local model's server. |
| Configuration | `src/config.py` | Centralized settings from the environment. |

---

## 9. Constraints & Assumptions

### Constraints
- **C-1:** Full documents must never be uploaded to a public LLM service, and confidential documents must never be sent to one at all.
- **C-2:** The system must run on commodity hardware; a GPU is optional.
- **C-3:** The project must be finished within the IT 4005 course timeline.
- **C-4:** Only open-source or free tools and services may be used (Groq's free tier has rate limits).

### Assumptions
- **A-1:** Documents are in English, text-based, and not scanned images (no OCR needed).
- **A-2:** A Groq API key is available. For the local model, a `llama-server` build and a GGUF model file are available on the API's machine, with about 4 GB of free RAM.
- **A-3:** All users have the same read access to every document. Marking a document confidential controls **where** it is processed, not **who** may ask about it.
- **A-4:** Sample or public company documents are used for development and demos.

---

## 10. Out of Scope (This Release)

- Corporate SSO / LDAP integration (FR-33).
- Per-user accounts, roles and per-document access control (e.g. HR-only documents).
- Document summarization on request (FR-17).
- OCR for scanned PDFs and images.
- Multi-language support.
- Model fine-tuning.
- Mobile application.

---

## 11. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| LLM hallucination gives wrong answers | High | Medium | Strict grounding prompt, "couldn't find" fallback (FR-15), numbered citations (FR-14, NFR-10), low temperature (0.1). |
| Confidential text reaches a cloud model | High | Low | Confidential routing (FR-44), refusal when no local model is available, confidential turns removed from cloud history, API refuses to start if the confidential list is unreadable (NFR-23); covered by automated tests. |
| Local model too slow on CPU-only hardware | Medium | High (observed) | Smaller prompts for the local model, `--reasoning off`, one answer at a time, streaming, and a notice in the chat; use it only when selected or for confidential questions. |
| Groq free-tier rate limits or outages | Medium | Medium | Clear "model is busy" message; the admin can switch to another model. |
| Poor retrieval quality from bad chunking | High | Medium | Tune chunk size and overlap; evaluate with a test question set (NFR-09, not yet done). |
| Sensitive data leaks through logs | High | Low | NFR-05; errors shown to users never include internal details. |
| Prompt injection hidden in documents | Medium | Low | Separate context from instructions (NFR-04); reserved tags in document text are neutralized. |
| Dependency or version conflicts (LangChain changes fast) | Medium | Medium | Pinned versions in `requirements-lock.txt`; a CI pipeline would catch breakage (NFR-18, not yet configured). |

---

## 12. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | All **Must-have** functional requirements are implemented and demonstrated. | ✅ |
| 2 | A PDF, a DOCX and a TXT document can be ingested, and questions about each are answered correctly with citations. | ✅ |
| 3 | A question outside the knowledge base returns a "not found" response rather than an invented answer. | ✅ |
| 4 | Questions that use confidential documents make no call to a cloud model (NFR-23). | ✅ Automated tests; checked with the real local model and a cloud "tripwire" |
| 5 | The test suite passes with ≥ 70% coverage (NFR-17) in CI (NFR-18). | ⚠️ 98% coverage locally; no CI yet |
| 6 | The API's Swagger docs are available at `/docs`. | ✅ |
| 7 | The README contains complete setup and run instructions. | ✅ |

---

## 13. Requirements Traceability Matrix

| Requirement(s) | Module | Test |
|---|---|---|
| FR-01, FR-02, FR-10 | `src/data_pipeline/document_loaders.py` | `tests/test_loaders.py` |
| FR-03 | `src/data_pipeline/text_splitter.py` | `tests/test_loaders.py` |
| FR-05 – FR-10, NFR-03 | `src/data_pipeline/ingestion.py` | `tests/test_ingestion.py` |
| FR-04, FR-05, FR-08, FR-09, NFR-19 | `src/data_pipeline/vector_store.py` | `tests/test_vector_store.py` |
| FR-12, FR-16 | `src/rag_engine/retriever.py` | `tests/test_retriever.py` |
| FR-13 – FR-16, FR-18, NFR-04, NFR-10 | `src/rag_engine/llm_chain.py`, `src/rag_engine/prompts.py` | `tests/test_retriever.py` |
| FR-19 – FR-25, FR-31, FR-32, NFR-20, NFR-24 | `src/api/routes.py`, `src/api/schemas.py`, `src/api/app.py`, `src/api/auth.py` | `tests/test_api.py`, `tests/test_web_api.py` |
| FR-26 – FR-30 | `frontend/src/features/chat/` | `frontend/src/features/chat/*.test.ts(x)` |
| FR-29, FR-38 – FR-40, FR-42, FR-43 | `frontend/src/features/admin/` | `frontend/src/features/admin/AdminPage.test.tsx` |
| FR-34 – FR-36, NFR-02, NFR-15 | `src/config.py`, `.env.example` | `tests/test_config.py` |
| FR-37, NFR-16 | `src/model_factory.py` | `tests/test_model_factory.py` |
| FR-38, FR-41 | `src/rag_engine/model_selector.py` | `tests/test_model_selector.py` |
| FR-39 | `src/rag_engine/instructions.py` | `tests/test_instructions.py` |
| FR-40 | `src/data_pipeline/upload_limit.py` | `tests/test_upload_limit.py` |
| FR-41, NFR-25 | `src/local_llm.py` | `tests/test_local_llm.py` (runs a fake `llama-server` process) |
| FR-43 – FR-45, NFR-23 | `src/data_pipeline/confidential.py`, `src/rag_engine/llm_chain.py` | `tests/test_confidential.py`, `tests/test_web_api.py` |
| NFR-17 | whole `src/` package | `pytest --cov=src` (98%) |
| NFR-18 | `.github/workflows/ci.yml` | Not configured yet |
