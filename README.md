# 🤖 InnovaTech AI — Internal Document Assistant (RAG Application)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Framework](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![Backend](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi)
![Frontend](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react)
![Database](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20llama.cpp-f55036?style=flat-square)
![Course](https://img.shields.io/badge/Course-IT%204005%20Advanced%20Software%20Engineering-purple?style=flat-square)

---

## 📌 Executive Summary

**Innovatech** is a fast-growing tech company onboarding new employees regularly. New hires are often overwhelmed by large volumes of internal documentation, manuals, technical specs, and guidelines. Due to strict security and privacy policies regarding proprietary company data, uploading these documents to public LLMs (e.g., ChatGPT) is strictly prohibited.

**InnovaTech AI** solves this challenge by implementing an internal, secure **Retrieval-Augmented Generation (RAG) AI assistant**. Built using **LangChain**, **FastAPI**, and **ChromaDB**, the assistant ingests and indexes private company documents locally and answers employees' questions from them, citing its sources. Employees interact via a web interface to receive accurate, context-aware answers, and the admin chooses whether answers are written by a cloud model or by a model running entirely on the company's own machine.

> **Privacy note.** Documents are stored, indexed and searched **locally**: embeddings run on
> your own CPU and the vector database is on disk. What happens to a question depends on the
> chat model the admin selects:
> - **Groq model (cloud):** the question and the few most relevant excerpts are sent to the
>   Groq API to write the answer. Full documents are never uploaded.
> - **Local model (optional):** a GGUF model runs on the same machine with llama.cpp, so
>   **nothing leaves the machine at all**. See [Local model](#-local-model-optional).
> - **Confidential documents:** documents the admin marks confidential (e.g. SOPs) are only
>   ever read by the local model, **even when a Groq model is selected**. See
>   [Confidential documents](#-confidential-documents).

---

## ✨ Features

**For employees**
- 💬 Chat in natural language; answers **stream** in as they are written
- 📎 Every answer cites its sources; click a `[1]` chip to see the file, page and excerpt
- 🧠 Follow-up questions keep the conversation's context
- 🔒 Answers based on confidential documents are marked as answered privately
- ⏹️ Stop an answer, retry a failed one, or start a new chat; the chat survives a page reload
- 🚫 Says *"I couldn't find this information in the company documents."* instead of guessing
- 🌗 Light and dark themes

**For admins** (password-protected)
- 📤 Upload PDF, Word (.docx), text and Markdown files (drag and drop, several at once)
- 🔁 Duplicate uploads are detected and skipped; documents can be deleted
- 📏 **Set the maximum file size** for uploads (default 20 MB); takes effect on the next upload
- 🔒 **Mark documents as confidential** (on upload, or later from the list): questions about
  them are answered only by the local model
- ✍️ Write instructions for the assistant (tone, format, escalation contacts). They shape
  answers but can't override the grounding rules.
- 🤖 **Choose the chat model**: Groq free-tier models, or a **local GGUF model** that the
  API starts and stops itself; takes effect on the next question
- 🔄 Re-index saved files, e.g. after changing the embedding model

**Safety**
- Prompt-injection defense: document text is treated as data, never as instructions
- Admin sessions use a signed, expiring HttpOnly cookie; login is throttled after
  5 wrong passwords per minute
- File type and size checks on upload
- The local model's server listens on `127.0.0.1` only, so it isn't reachable from the network
- Confidential documents never reach a cloud model: such questions are refused if the local
  model is unavailable

---

## 🏗️ System Architecture & Workflow

The application follows the standard Software Development Life Cycle (SDLC) tailored for AI systems, adhering strictly to **Object-Oriented Programming (OOP)** and enterprise design patterns.

```text
                [React web app (Vite)]
                          │
                          ▼
              [REST API layer (FastAPI)]
       /api/chat · /api/chat/stream · /api/admin/*
                          │
                          ▼
            [RAG engine (LangChain) — RAGChain]
     ┌──────────────┬─────┴────────┬──────────────────┐
     ▼              ▼              ▼                  ▼
 [Retriever]    [Prompts]    [Instructions]    [Model selector]
     │        (rules + admin   (JSON file)       (admin's choice)
     │         + context)                              │
     ▼                                                 ▼
 [ChromaDB] ◄── [Local embeddings]        [Groq chat model] or [Local model]
 (on disk)       FastEmbed · CPU            (cloud API)     llama-server · GGUF
     ▲
     │
 [Ingestion: Loader Factory → Text Splitter → Embeddings]
   PDF · DOCX · TXT · MD
```

| Layer | Package | Responsibility |
|---|---|---|
| Presentation | `frontend/` (React) | Chat and admin interfaces |
| API | `src/api/` | REST endpoints, validation, auth, error handling |
| Business logic | `src/rag_engine/` | Retrieval, prompts, model selection, answer generation |
| Data | `src/data_pipeline/` | Loading, chunking, embedding, vector store |
| Configuration | `src/config.py` | All settings from `.env` |

**Design patterns:** Factory (document loaders, model factories), Repository (vector store),
Facade (`RAGChain.ask` / `stream`), Singleton (settings), Dependency Injection (FastAPI `Depends`).

---

## 🧰 Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.10+ (developed on 3.12), TypeScript |
| RAG orchestration | LangChain |
| Chat model | [Groq](https://console.groq.com) free tier: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`; optionally a local GGUF model via [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) `BAAI/bge-small-en-v1.5`, runs locally on CPU |
| Vector store | ChromaDB (persistent, local) |
| Backend | FastAPI + Uvicorn |
| Frontend | React 19, Vite, Tailwind CSS, TanStack Query |
| Testing | pytest, Vitest + Testing Library; ruff, ESLint, TypeScript |

---

## 📁 Project Structure

```text
Innovatech-AI-Assisstant/
├── src/
│   ├── config.py              # settings from .env
│   ├── model_factory.py       # Groq and local chat models, local embeddings
│   ├── local_llm.py           # starts/stops llama-server for the local model
│   ├── api/                   # FastAPI app, routes, schemas, auth
│   ├── rag_engine/            # RAG chain, prompts, retriever, model selector, instructions
│   └── data_pipeline/         # loaders, text splitter, vector store, ingestion
├── frontend/                  # React web app (Vite)
├── tests/                     # pytest suite (fake models, no API calls)
├── data/
│   ├── raw/                   # uploaded files (git-ignored)
│   ├── vector_db/             # ChromaDB (git-ignored)
│   ├── confidential_documents.json  # which documents are confidential (git-ignored)
│   ├── models/                # downloaded embedding model (git-ignored)
│   └── logs/                  # llama-server output (git-ignored)
├── docs/requirement_analysis.md  # original requirements specification (SRS)
├── .env.example
└── requirements.txt
```

---

## 🚀 Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 20.19+ or 22.12+** (for the React web app)
- A free **Groq API key**: create one at <https://console.groq.com/keys>

### 1. Get the code and create a virtual environment

```bash
git clone https://github.com/Sanjulauni/Innovatech-AI-Assisstant.git
cd Innovatech-AI-Assisstant
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 3. Configure `.env`

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then open `.env` and set at least:

```ini
GROQ_API_KEY=your-groq-key
ADMIN_PASSWORD=choose-a-strong-password   # leave empty to disable the admin page
```

Everything else has working defaults. `.env` is git-ignored; never commit it.

> **Embedding model download.** The first upload or question downloads the local embedding
> model (about 70 MB) to `data/models/`. After that it works offline.

---

## ▶️ Running

### Development (two terminals)

**Terminal 1: API**
```bash
uvicorn src.api.app:app
```

**Terminal 2: web app**
```bash
cd frontend
npm run dev
```

Open **<http://localhost:5173>**. The dev server forwards `/api` requests to the API on port 8000.

### Single server

Build the web app once; the API then serves it too:

```bash
cd frontend
npm run build
cd ..
uvicorn src.api.app:app
```

Open **<http://localhost:8000>**. API docs (Swagger) are at **<http://localhost:8000/docs>**.

---

## 🧭 First Steps

1. Open **Admin** and sign in with your `ADMIN_PASSWORD`.
2. **Documents:** upload your PDF, Word, text or Markdown files.
3. **Model:** pick a chat model. GPT-OSS 120B is the default and gives the best answers; a
   [local model](#-local-model-optional) keeps everything on the machine but is slower.
4. **Instructions (optional):** set the tone and escalation contacts.
5. Go to **Chat** and ask a question. Click `[1]` in an answer to see its source.

> Upgrading from an older version? Documents indexed with the previous embedding model
> aren't searchable by the new one. Click **Admin → Documents → Re-index saved files** once.

---

## 💻 Local model (optional)

Besides the Groq models, the admin can pick a **local model**: a GGUF file run on the
API's machine by llama.cpp's `llama-server`. Questions and document excerpts then never
leave the machine.

1. Download a `llama-server` build for your platform from the
   [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), and an
   instruction-tuned GGUF model (for example Gemma 4 E2B, `Q4_K_M`).
2. Point `.env` at both files (they can be on any drive):

   ```ini
   LOCAL_LLM_SERVER=D:\llama-bin\llama-server.exe
   LOCAL_LLM_MODEL=D:\gemma-4-E2B-it-Q4_K_M.gguf
   LOCAL_LLM_LABEL=Gemma 4 E2B
   ```

3. Restart the API. The model appears under **Admin → Model → On this server**.

You don't start `llama-server` yourself. The API starts it when the local model is
selected (and at startup if it was the last choice), shows *Loading* until the model is
ready, and stops it when another model is selected, so the model only uses memory while it
is in use. If the server crashes, the next question starts it again. It listens on
`127.0.0.1` only, and its output goes to `data/logs/llama-server.log`.

The server is started with `--reasoning off` (answer at once, with no hidden "thinking"
step) and `--parallel 1` (one answer at a time). Options in `LOCAL_LLM_ARGS`, such as
`--threads 4`, are added after these and override them.

> **Hardware and speed.** A 4-bit 2B-class model needs about 4 GB of free RAM. On a CPU the
> model reads the prompt slowly, so the local model gets a smaller prompt (2 excerpts, 2
> earlier messages) and answers without a hidden "thinking" step. On an entry-level laptop
> CPU the first words still take 30–60 seconds. Keep the `.gguf` on an internal SSD rather
> than a USB stick: with little free RAM, the model is re-read from disk while answering.

---

## 🔒 Confidential documents

Some documents, such as SOPs, should never be sent to a cloud service. The admin can mark
them **confidential**: tick *Confidential* before uploading, or click the lock next to a
document in **Admin → Documents**. The change applies from the next question.

When a question is asked, the documents are searched first. If any excerpt found comes
from a confidential document, **that question is answered by the local model**, whichever
model is selected. The chat marks the answer as answered privately, and its sources show
which documents are confidential.

- **No cloud fallback.** If no local model is set up, or it can't be started, the question
  is refused with a message instead. The admin page warns about this when documents are
  marked confidential but no local model is configured.
- **Follow-ups are protected too.** When a later question is answered by a cloud model,
  earlier answers based on confidential documents (and the questions that led to them)
  are left out of the chat history sent with it.
- **Starting on demand.** If a cloud model is selected, the local model is started for the
  first confidential question (the first answer then includes loading time) and stopped
  after `LOCAL_LLM_IDLE_MINUTES` without confidential questions.
- **Stored safely.** The marks are kept in `data/confidential_documents.json`, keyed by each
  file's content hash, so re-indexing or changing the embedding model keeps them. If the
  file can't be read, the API refuses to start rather than treat everything as normal.

---

## ⚙️ Configuration

All settings live in `.env` (see `.env.example` for the full list).

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `GROQ_MODELS` | `openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b` | Models the admin can choose from; the first is the default |
| `LLM_TEMPERATURE` | `0.1` | Lower = more predictable answers |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (re-index after changing it) |
| `ADMIN_PASSWORD` | empty | Admin password; admin is disabled when empty |
| `ADMIN_SESSION_HOURS` | `8` | How long an admin stays signed in |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | How documents are split |
| `RETRIEVER_TOP_K` | `4` | Excerpts retrieved per question |
| `CHAT_HISTORY_LIMIT` | `6` | Earlier messages sent with each question |
| `MAX_UPLOAD_SIZE_MB` | `20` | Largest file accepted, until the admin changes it |
| `MAX_UPLOAD_SIZE_CAP_MB` | `200` | Highest file size limit the admin can set |
| `LOCAL_LLM_SERVER` / `LOCAL_LLM_MODEL` | empty | Paths to `llama-server` and a `.gguf` model; set both to offer the local model |
| `LOCAL_LLM_LABEL` | file name | Name shown for the local model |
| `LOCAL_LLM_PORT` / `LOCAL_LLM_CONTEXT_SIZE` | `8080` / `8192` | Port and context window of the local server |
| `LOCAL_LLM_STARTUP_TIMEOUT` | `180` | Seconds to wait for the local model to load |
| `LOCAL_LLM_TOP_K` / `LOCAL_LLM_HISTORY_LIMIT` | `2` / `2` | Excerpts and earlier messages sent to the local model (smaller prompts answer faster on a CPU) |
| `LOCAL_LLM_ARGS` | empty | Extra `llama-server` options, e.g. `--threads 4` |
| `LOCAL_LLM_IDLE_MINUTES` | `15` | Stop a local model started only for confidential questions after this many idle minutes |
| `CONFIDENTIAL_FILE` | `data/confidential_documents.json` | Which documents are confidential |
| `LOCAL_LLM_LOG_FILE` | `data/logs/llama-server.log` | Where `llama-server`'s output is written |

---

## 🔌 API

Interactive docs: `/docs`.

| Method | Endpoint | Who | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | anyone | Status, document count, current model |
| `POST` | `/api/chat` | anyone | Ask a question → answer + sources |
| `POST` | `/api/chat/stream` | anyone | Same, streamed as NDJSON: `route` (which model answers), `token`s, then `done` or `error` |
| `POST` | `/api/admin/login` · `/logout` | admin | Start / end a session (cookie) |
| `GET` | `/api/admin/session` | admin | Check the session is still valid |
| `GET` · `POST` | `/api/admin/documents` | admin | List / upload documents |
| `DELETE` | `/api/admin/documents/{doc_id}` | admin | Delete a document |
| `PUT` | `/api/admin/documents/{doc_id}/confidential` | admin | Mark / unmark a document as confidential |
| `GET` · `PUT` | `/api/admin/upload-limit` | admin | Current / set the maximum file size |
| `POST` | `/api/admin/documents/reindex` | admin | Index saved files not yet searchable |
| `GET` · `PUT` | `/api/admin/model` | admin | Available models / choose one |
| `GET` · `PUT` | `/api/admin/instructions` | admin | Read / save agent instructions |

Admin endpoints accept the session cookie from `/api/admin/login`, or an
`X-Admin-Password` header.

---

## 🧪 Testing

The tests use fake models and an in-memory vector store, so they make **no API calls**
and need no API key. The local-model tests start a small fake `llama-server` (written in
Python) as a real process, so no model file is needed either.

```bash
# Backend
pytest
ruff check .

# Frontend
cd frontend
npm test
npm run lint
npm run typecheck
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is not set` on startup | Add your key to `.env` and restart the API |
| `llm_provider: Input should be 'groq'` | Your `.env` is from an older version: set `LLM_PROVIDER=groq` or remove the line |
| "The AI model is busy (usage limit reached)" | Groq's free-tier rate limit; wait a minute, or switch to another model in **Admin → Model** |
| Admin page says admin access is turned off | Set `ADMIN_PASSWORD` in `.env` and restart the API |
| Every answer is "I couldn't find this information…" | No documents are indexed yet: upload some, or click **Re-index saved files** |
| First upload or question is slow | The embedding model is being downloaded (once, ~70 MB) |
| Local model shows an error in **Admin → Model** | The message says why (e.g. file not found). More detail is in `data/logs/llama-server.log`. Fix it, then click the model to try again |
| "The local model is still loading" | Loading takes a while on first use; wait and ask again |
| "This question needs confidential documents … no local model is set up" | The question matched a confidential document. Set up the [local model](#-local-model-optional), or unmark the document if it isn't confidential |
| API won't start: "Could not read the confidential documents list" | `data/confidential_documents.json` is damaged. Restore it from a backup, or fix the JSON |
| Local model fails with "out of memory" / exits while loading | Close other programs, or use a smaller quantization (e.g. `Q4_K_S`) or lower `LOCAL_LLM_CONTEXT_SIZE` |
| `http://127.0.0.1:5173` doesn't load | Use `http://localhost:5173` (the dev server listens on `localhost`) |

---

## 📄 License

[MIT](LICENSE)
