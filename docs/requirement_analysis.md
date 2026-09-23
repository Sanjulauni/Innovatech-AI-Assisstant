# 📋 Requirement Analysis — InnovaTech AI Internal Document Assistant

| Item | Detail |
|---|---|
| **Project** | InnovaTech AI — Internal Document Assistant (RAG Application) |
| **Course** | IT 4005 Advanced Software Engineering |
| **Document type** | Software Requirements Specification (SRS) |
| **Version** | 1.0 (Draft) |
| **Status** | For review |

---

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the InnovaTech AI Internal Document Assistant. It is the reference for design, implementation, testing and acceptance, and it is the first SDLC phase of the project.

### 1.2 Scope
The system is a secure, self-hosted **Retrieval-Augmented Generation (RAG)** assistant. It:

- ingests Innovatech's internal documents (manuals, technical specs, guidelines, policies),
- indexes them in a local vector database,
- answers employee questions in natural language, citing the source documents,
- runs entirely inside the company's network, so proprietary data is never sent to a public LLM service.

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

---

## 2. Problem Statement

Innovatech hires new employees regularly. New hires must absorb a large volume of internal documentation, which slows onboarding and pulls senior staff away from their work to answer repeated questions.

Company security and privacy policy **prohibits uploading proprietary documents to public LLMs** such as ChatGPT. Employees therefore cannot use widely available AI tools to search or summarize internal knowledge.

**Goal:** give employees an AI assistant that answers questions accurately from internal documentation while keeping all data inside the company's network.

---

## 3. Stakeholders & User Roles

| Stakeholder | Role / Interest |
|---|---|
| **Employee (End User)** | Asks questions about internal documentation and needs fast, accurate, sourced answers. New hires are the primary users. |
| **Knowledge Administrator** | Uploads, updates and removes documents, and keeps the knowledge base current. |
| **IT / Security Team** | Requires that no data leaves the network and that the system meets company security policy. |
| **Management** | Wants shorter onboarding time and less load on senior staff. |
| **Development Team** | Builds and maintains the system. |

---

## 4. Functional Requirements

Priority uses **MoSCoW**: **M**ust, **S**hould, **C**ould, **W**on't (this release).

### 4.1 Document Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The system shall load documents in **PDF, DOCX, TXT and Markdown** formats. | M |
| FR-02 | The system shall choose the correct loader for each file type through a **Loader Factory**, so new formats can be added without changing existing code. | M |
| FR-03 | The system shall split documents into overlapping chunks, with chunk size and overlap set in configuration. | M |
| FR-04 | The system shall generate an embedding for each chunk using a **locally hosted** embedding model. | M |
| FR-05 | The system shall store chunks, embeddings and metadata (source file name, page number, chunk index, ingestion timestamp) in ChromaDB, persisted on local disk. | M |
| FR-06 | The system shall support bulk ingestion of every document in `data/raw/`. | M |
| FR-07 | The system shall let an administrator upload individual documents through the API/UI. | S |
| FR-08 | The system shall detect documents that are already indexed (by content hash) and skip or re-index them instead of creating duplicates. | S |
| FR-09 | The system shall let an administrator delete a document and all of its chunks from the vector store. | S |
| FR-10 | The system shall report unsupported or corrupt files with a clear error and continue processing the remaining files. | M |

### 4.2 Question Answering (RAG)

| ID | Requirement | Priority |
|---|---|---|
| FR-11 | The system shall accept a natural-language question from the user. | M |
| FR-12 | The system shall retrieve the top-*k* most relevant chunks from the vector store (*k* configurable, default 4). | M |
| FR-13 | The system shall generate an answer with a **locally hosted LLM**, using only the retrieved chunks as context. | M |
| FR-14 | The system shall return the **source citations** (document name and page) used to produce each answer. | M |
| FR-15 | When the retrieved context does not contain the answer, the system shall say it could not find the information rather than invent one. | M |
| FR-16 | The system shall support follow-up questions within a chat session (conversation memory). | S |
| FR-17 | The system shall summarize a specified document on request. | C |
| FR-18 | The system shall stream the answer to the UI token by token. | C |

### 4.3 REST API (FastAPI)

| ID | Requirement | Priority |
|---|---|---|
| FR-19 | `POST /query`: accept a question (and optional session ID); return the answer and its sources. | M |
| FR-20 | `POST /ingest`: upload one or more documents and index them. | S |
| FR-21 | `GET /documents`: list the indexed documents with their metadata. | S |
| FR-22 | `DELETE /documents/{id}`: remove a document from the index. | S |
| FR-23 | `GET /health`: report the status of the API, the vector store and the LLM. | M |
| FR-24 | The API shall validate every request and response with **Pydantic schemas** and return meaningful HTTP error codes. | M |
| FR-25 | The API shall publish auto-generated OpenAPI (Swagger) documentation. | M |

### 4.4 User Interface (Streamlit)

| ID | Requirement | Priority |
|---|---|---|
| FR-26 | The UI shall provide a chat interface for asking questions and viewing answers. | M |
| FR-27 | The UI shall display the source citations under each answer. | M |
| FR-28 | The UI shall keep the chat history for the current session and allow it to be cleared. | S |
| FR-29 | The UI shall provide a document upload panel for administrators. | S |
| FR-30 | The UI shall show a loading indicator while an answer is being generated. | S |

### 4.5 Access Control

| ID | Requirement | Priority |
|---|---|---|
| FR-31 | The system shall separate **Employee** and **Administrator** permissions; only administrators can ingest or delete documents. | S |
| FR-32 | The API shall require an API key (or token) on protected endpoints. | S |
| FR-33 | Integration with the corporate SSO / LDAP directory. | W |

---

## 5. Non-Functional Requirements

### 5.1 Security & Privacy

| ID | Requirement |
|---|---|
| NFR-01 | **No document content, prompt or query shall be sent to any external or public AI service.** The LLM and embedding models shall run on company infrastructure. |
| NFR-02 | Secrets (API keys, configuration) shall be loaded from environment variables (`.env`) and never committed to source control. |
| NFR-03 | Uploaded files shall be checked for allowed extension and maximum size (default 20 MB) before processing. |
| NFR-04 | The system shall defend against prompt injection. Retrieved context shall be clearly separated from system instructions in the prompt. |
| NFR-05 | Logs shall not record full document contents or sensitive user data. |

### 5.2 Performance

| ID | Requirement |
|---|---|
| NFR-06 | Retrieval (vector search) shall complete in **< 1 second** for a knowledge base of up to 10,000 chunks. |
| NFR-07 | An end-to-end answer shall be returned in **< 15 seconds** on the reference hardware (8-core CPU, 16 GB RAM, optional GPU). |
| NFR-08 | The system shall support at least **10 concurrent users** without errors. |

### 5.3 Accuracy & Quality

| ID | Requirement |
|---|---|
| NFR-09 | At least **80%** of the questions in a curated evaluation set shall receive answers judged correct and grounded in the sources. |
| NFR-10 | Every factual answer shall include at least one source citation. |

### 5.4 Usability

| ID | Requirement |
|---|---|
| NFR-11 | A new employee shall be able to ask a first question with no training, in under 1 minute. |
| NFR-12 | Error messages shall be in plain language and suggest what to do next. |

### 5.5 Maintainability & Code Quality

| ID | Requirement |
|---|---|
| NFR-13 | The code shall follow **Object-Oriented** design and a layered architecture (UI → API → RAG Engine → Data Pipeline). |
| NFR-14 | The code shall follow PEP 8, include type hints, and pass linting (`ruff` / `flake8`). |
| NFR-15 | Configurable values (model names, chunk size, *k*, paths) shall be kept in `src/config.py` / `.env`, never hard-coded. |
| NFR-16 | The LLM and embedding providers shall be swappable through configuration without changing business logic. |

### 5.6 Reliability & Testability

| ID | Requirement |
|---|---|
| NFR-17 | Unit and integration tests shall cover at least **70%** of the `src/` code. |
| NFR-18 | The CI pipeline (GitHub Actions) shall run linting and tests on every push and pull request. |
| NFR-19 | The vector store shall persist across application restarts. |
| NFR-20 | If the LLM is unavailable, the system shall fail gracefully with a clear error instead of crashing. |

### 5.7 Portability

| ID | Requirement |
|---|---|
| NFR-21 | The system shall run on Windows, macOS and Linux with Python 3.10+. |
| NFR-22 | Setup shall need only `pip install -r requirements.txt`, a `.env` file and the documented `make` targets. |

---

## 6. Use Cases

### UC-01 — Ask a Question
| Field | Description |
|---|---|
| **Actor** | Employee |
| **Precondition** | At least one document has been ingested; the LLM service is running. |
| **Main flow** | 1. The employee types a question in the chat UI. 2. The UI sends `POST /query`. 3. The system embeds the question and retrieves the top-*k* chunks. 4. The system builds a prompt from the chunks and question and calls the local LLM. 5. The system returns the answer and its sources. 6. The UI displays the answer with its citations. |
| **Alternate flow** | 3a. No relevant chunks are found → the system replies that the information is not available in the knowledge base. |
| **Exception flow** | 4a. The LLM is unreachable → the API returns `503` and the UI shows a friendly error. |
| **Postcondition** | The question and answer appear in the session's chat history. |

### UC-02 — Ingest Documents
| Field | Description |
|---|---|
| **Actor** | Knowledge Administrator |
| **Precondition** | The administrator is authenticated. |
| **Main flow** | 1. The admin uploads files in the UI (or places them in `data/raw/`). 2. The system validates type and size. 3. The Loader Factory selects a loader for each file. 4. The documents are split into chunks. 5. The chunks are embedded and stored in ChromaDB with metadata. 6. The system reports how many files and chunks were ingested. |
| **Alternate flow** | 2a. Unsupported or corrupt file → it is skipped, reported, and the rest continue. 3a. Document already indexed → it is skipped, or re-indexed if its content changed. |
| **Postcondition** | The new content can be retrieved in queries. |

### UC-03 — View Answer Sources
| Field | Description |
|---|---|
| **Actor** | Employee |
| **Main flow** | 1. The employee receives an answer. 2. The employee expands "Sources". 3. The system shows the document name, page and a snippet of each chunk used. |

### UC-04 — Manage Knowledge Base
| Field | Description |
|---|---|
| **Actor** | Knowledge Administrator |
| **Main flow** | 1. The admin opens the document list (`GET /documents`). 2. The admin selects a document to delete. 3. The system removes all of its chunks from the vector store (`DELETE /documents/{id}`). |

### Use Case Overview

```text
                ┌──────────────────────────────────────┐
                │      InnovaTech AI Assistant         │
  Employee ─────┼──► (UC-01 Ask a Question)            │
      │         │           │  «include»              │
      └─────────┼──► (UC-03 View Answer Sources)       │
                │                                      │
  Knowledge ────┼──► (UC-02 Ingest Documents)          │
  Administrator │                                      │
      └─────────┼──► (UC-04 Manage Knowledge Base)     │
                └──────────────────────────────────────┘
```

---

## 7. Technology Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Mature AI/ML ecosystem. |
| RAG orchestration | LangChain | Standard abstractions for loaders, splitters, retrievers and chains. |
| Vector store | ChromaDB (persistent, local) | Lightweight, embedded and free; no external server needed. |
| LLM runtime | **Ollama** (default model: `llama3.1:8b`) | Runs open-weight models locally, satisfying NFR-01. |
| Embedding model | `nomic-embed-text` via Ollama *(alt: `sentence-transformers/all-MiniLM-L6-v2`)* | Runs locally with good retrieval quality. |
| Backend API | FastAPI + Uvicorn | Async, Pydantic validation, automatic OpenAPI docs. |
| Frontend | Streamlit | Fast to build a chat UI in pure Python. |
| Testing | pytest, pytest-cov, FastAPI `TestClient` | Unit and integration testing with coverage reports. |
| Code quality | ruff / black | Linting and formatting. |
| CI | GitHub Actions | Runs lint and tests automatically on every push and pull request. |

> **Note:** The model choices above are defaults. Per NFR-16, any locally hosted LLM or embedding model can replace them through `.env` configuration.

---

## 8. Design Patterns & Architecture Mapping

| Pattern | Where | Purpose |
|---|---|---|
| **Factory** | `data_pipeline/document_loaders.py` | Choose a loader by file extension; new formats need no changes to existing code (Open/Closed Principle). |
| **Strategy** | `data_pipeline/text_splitter.py`, embedding provider | Swap chunking or embedding strategies at runtime. |
| **Singleton** | `config.py` (settings), vector store client | One shared configuration object and database connection. |
| **Repository** | `data_pipeline/vector_store.py` | Hide ChromaDB behind a clean interface (add, search, delete). |
| **Facade** | `rag_engine/llm_chain.py` | Offer a single `ask(question)` entry point over retrieval, prompting and generation. |
| **Dependency Injection** | `api/routes.py` (FastAPI `Depends`) | Decouple endpoints from concrete services and make testing easier. |

### Layered Architecture

| Layer | Package | Responsibility |
|---|---|---|
| Presentation | `src/ui/` | Streamlit chat and upload interface. |
| API | `src/api/` | REST endpoints, validation, error handling. |
| Business logic | `src/rag_engine/` | Retrieval, prompt construction, LLM chain. |
| Data | `src/data_pipeline/` | Loading, chunking, embedding, persistence. |
| Configuration | `src/config.py` | Centralized settings from the environment. |

---

## 9. Constraints & Assumptions

### Constraints
- **C-1:** Public LLM APIs (OpenAI, Gemini, etc.) must not be used for any document or query data.
- **C-2:** The system must run on commodity hardware; a GPU is optional.
- **C-3:** The project must be finished within the IT 4005 course timeline.
- **C-4:** Only open-source or free tools may be used.

### Assumptions
- **A-1:** Documents are in English, text-based, and not scanned images (no OCR needed).
- **A-2:** Ollama (or an equivalent local runtime) is installed on the host machine.
- **A-3:** All users have the same read access to every document (no per-document permissions in this release).
- **A-4:** Sample or synthetic company documents will be used for development and demos.

---

## 10. Out of Scope (This Release)

- Corporate SSO / LDAP integration (FR-33).
- Per-document access control (e.g. HR-only documents).
- OCR for scanned PDFs and images.
- Multi-language support.
- Model fine-tuning.
- Mobile application.

---

## 11. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| LLM hallucination gives wrong answers | High | Medium | Strict grounding prompt, "I don't know" fallback (FR-15), mandatory citations (NFR-10). |
| Local LLM too slow on CPU-only hardware | Medium | High | Use a small quantized model; make the model configurable; stream responses (FR-18). |
| Poor retrieval quality from bad chunking | High | Medium | Tune chunk size and overlap; evaluate with a test question set (NFR-09). |
| Sensitive data leaks through logs or external calls | High | Low | NFR-01 and NFR-05; review code for outbound network calls. |
| Prompt injection hidden in documents | Medium | Low | Separate context from instructions (NFR-04); instruct the model to treat context as data. |
| Dependency or version conflicts (LangChain changes fast) | Medium | Medium | Pin versions in `requirements.txt`; CI catches breakage. |

---

## 12. Acceptance Criteria

The release is accepted when:

1. ✅ All **Must-have** functional requirements are implemented and demonstrated.
2. ✅ A PDF, a DOCX and a TXT document can be ingested, and questions about each are answered correctly with citations.
3. ✅ A question outside the knowledge base returns a "not found" response rather than an invented answer.
4. ✅ Network inspection confirms no outbound calls to external AI services (NFR-01).
5. ✅ The test suite passes in CI with ≥ 70% coverage (NFR-17, NFR-18).
6. ✅ The API's Swagger docs are available at `/docs`.
7. ✅ The README contains complete setup and run instructions.

---

## 13. Requirements Traceability Matrix

| Requirement(s) | Module | Test |
|---|---|---|
| FR-01, FR-02, FR-10 | `src/data_pipeline/document_loaders.py` | `tests/test_loaders.py` |
| FR-03 | `src/data_pipeline/text_splitter.py` | `tests/test_loaders.py` |
| FR-04, FR-05, FR-08, FR-09 | `src/data_pipeline/vector_store.py` | `tests/test_retriever.py` |
| FR-12 | `src/rag_engine/retriever.py` | `tests/test_retriever.py` |
| FR-13, FR-15, FR-16, NFR-04 | `src/rag_engine/llm_chain.py`, `src/rag_engine/prompts.py` | `tests/test_retriever.py` |
| FR-19 – FR-25, FR-32 | `src/api/routes.py`, `src/api/schemas.py`, `src/api/app.py` | `tests/test_api.py` |
| FR-26 – FR-30 | `src/ui/app.py`, `src/ui/components/` | Manual UI testing |
| NFR-02, NFR-15, NFR-16 | `src/config.py`, `.env.example` | `tests/test_api.py` (config loading) |
| NFR-17, NFR-18 | `.github/workflows/ci.yml`, `Makefile` | CI pipeline |
