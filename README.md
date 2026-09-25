# 🤖 InnovaTech AI — Internal Document Assistant (RAG Application)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Framework](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![Backend](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi)
![Frontend](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react)
![Database](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Groq-f55036?style=flat-square)
![Course](https://img.shields.io/badge/Course-IT%204005%20Advanced%20Software%20Engineering-purple?style=flat-square)

---

## 📌 Executive Summary

**Innovatech** is a fast-growing tech company onboarding new employees regularly. New hires are often overwhelmed by large volumes of internal documentation, manuals, technical specs, and guidelines. Due to strict security and privacy policies regarding proprietary company data, uploading these documents to public LLMs (e.g., ChatGPT) is strictly prohibited.

**InnovaTech AI** solves this challenge by implementing an internal, secure **Retrieval-Augmented Generation (RAG) AI assistant**. Built using **LangChain**, **FastAPI**, and **ChromaDB**, the assistant ingests, indexes, and summarizes private company documents locally within the enterprise boundary. Employees interact via a web interface to receive accurate, context-aware answers without compromising data privacy.

> **Privacy note (current version).** Documents are stored, indexed and searched **locally**:
> embeddings run on your own CPU and the vector database is on disk. To write an answer, the
> question and the few most relevant excerpts are sent to the **Groq API**, a cloud service.
> Full documents are never uploaded to Groq, but for highly confidential material a fully
> local model (e.g. via Ollama) would be needed. That is planned, not built.

---

## ✨ Features

**For employees**
- 💬 Chat in natural language; answers **stream** in as they are written
- 📎 Every answer cites its sources; click a `[1]` chip to see the file, page and excerpt
- 🧠 Follow-up questions keep the conversation's context
- 🚫 Says *"I couldn't find this information in the company documents."* instead of guessing
- 🌗 Light and dark themes

**For admins** (password-protected)
- 📤 Upload PDF, Word (.docx), text and Markdown files (drag and drop, several at once)
- 🔁 Duplicate uploads are detected and skipped; documents can be deleted
- 📏 **Set the maximum file size** for uploads (default 20 MB); takes effect on the next upload
- ✍️ Write instructions for the assistant (tone, format, escalation contacts). They shape
  answers but can't override the grounding rules.
- 🤖 **Choose the chat model** (Groq free-tier models); takes effect on the next question
- 🔄 Re-index saved files, e.g. after changing the embedding model

**Safety**
- Prompt-injection defense: document text is treated as data, never as instructions
- Admin sessions use a signed, expiring HttpOnly cookie; login is throttled after
  5 wrong passwords per minute
- File type and size checks on upload

---

## 🏗️ System Architecture & Workflow

The application follows the standard Software Development Life Cycle (SDLC) tailored for AI systems, adhering strictly to **Object-Oriented Programming (OOP)** and enterprise design patterns.

```text
 [React web app (Vite)]        [Streamlit UI (legacy)]
            │                           │
            └─────────────┬─────────────┘
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
 [ChromaDB] ◄── [Local embeddings]              [Groq chat model]
 (on disk)       FastEmbed · CPU                  (cloud API)
     ▲
     │
 [Ingestion: Loader Factory → Text Splitter → Embeddings]
   PDF · DOCX · TXT · MD
```

| Layer | Package | Responsibility |
|---|---|---|
| Presentation | `frontend/` (React), `src/ui/` (Streamlit) | Chat and admin interfaces |
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
| Chat model | [Groq](https://console.groq.com) free tier: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) `BAAI/bge-small-en-v1.5`, runs locally on CPU |
| Vector store | ChromaDB (persistent, local) |
| Backend | FastAPI + Uvicorn |
| Frontend | React 19, Vite, Tailwind CSS, TanStack Query |
| Testing | pytest, Vitest + Testing Library; ruff, ESLint, TypeScript |

---

## 📁 Project Structure

```text
innovatech-rag-assistant/
├── src/
│   ├── config.py              # settings from .env
│   ├── model_factory.py       # Groq chat models, local embeddings
│   ├── api/                   # FastAPI app, routes, schemas, auth
│   ├── rag_engine/            # RAG chain, prompts, retriever, model selector, instructions
│   ├── data_pipeline/         # loaders, text splitter, vector store, ingestion
│   └── ui/                    # Streamlit UI (legacy)
├── frontend/                  # React web app (Vite)
├── tests/                     # pytest suite (fake models, no API calls)
├── data/
│   ├── raw/                   # uploaded files (git-ignored)
│   ├── vector_db/             # ChromaDB (git-ignored)
│   └── models/                # downloaded embedding model (git-ignored)
├── docs/requirement_analysis.md
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
git clone https://github.com/Sanjulauni/innovatech-rag-assistant.git
cd innovatech-rag-assistant
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
3. **Model:** pick a chat model (GPT-OSS 120B is the default and gives the best answers).
4. **Instructions (optional):** set the tone and escalation contacts.
5. Go to **Chat** and ask a question. Click `[1]` in an answer to see its source.

> Upgrading from an older version? Documents indexed with the previous embedding model
> aren't searchable by the new one. Click **Admin → Documents → Re-index saved files** once.

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

---

## 🔌 API

Interactive docs: `/docs`. All endpoints are also available without the `/api` prefix
(used by the Streamlit UI).

| Method | Endpoint | Who | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | anyone | Status, document count, current model |
| `POST` | `/api/chat` | anyone | Ask a question → answer + sources |
| `POST` | `/api/chat/stream` | anyone | Same, streamed as NDJSON events |
| `POST` | `/api/admin/login` · `/logout` | admin | Start / end a session (cookie) |
| `GET` | `/api/admin/session` | admin | Check the session is still valid |
| `GET` · `POST` | `/api/admin/documents` | admin | List / upload documents |
| `DELETE` | `/api/admin/documents/{doc_id}` | admin | Delete a document |
| `GET` · `PUT` | `/api/admin/upload-limit` | admin | Current / set the maximum file size |
| `POST` | `/api/admin/documents/reindex` | admin | Index saved files not yet searchable |
| `GET` · `PUT` | `/api/admin/model` | admin | Available models / choose one |
| `GET` · `PUT` | `/api/admin/instructions` | admin | Read / save agent instructions |

Admin endpoints accept the session cookie from `/api/admin/login`, or an
`X-Admin-Password` header.

---

## 🧪 Testing

The tests use fake models and an in-memory vector store, so they make **no API calls**
and need no API key.

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
| `http://127.0.0.1:5173` doesn't load | Use `http://localhost:5173` (the dev server listens on `localhost`) |

---

## 📄 License

[MIT](LICENSE)
