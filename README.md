# 🤖 InnovaTech AI — Internal Document Assistant (RAG Application)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Framework](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![Backend](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi)
![Database](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange?style=flat-square)
![Course](https://img.shields.io/badge/Course-IT%204005%20Advanced%20Software%20Engineering-purple?style=flat-square)

---

## 📌 Executive Summary

**Innovatech** is a fast-growing tech company onboarding new employees regularly. New hires are often overwhelmed by large volumes of internal documentation, manuals, technical specs, and guidelines. Due to strict security and privacy policies regarding proprietary company data, uploading these documents to public LLMs (e.g., ChatGPT) is strictly prohibited.

**InnovaTech AI** solves this challenge by implementing an internal, secure **Retrieval-Augmented Generation (RAG) AI assistant**. Built using **LangChain**, **FastAPI**, and **ChromaDB**, the assistant ingests, indexes, and summarizes private company documents locally within the enterprise boundary. Employees interact via a web interface to receive accurate, context-aware answers without compromising data privacy.

---

## 🏗️ System Architecture & Workflow

The application follows the standard Software Development Life Cycle (SDLC) tailored for AI systems, adhering strictly to **Object-Oriented Programming (OOP)** and enterprise design patterns.

```text
[User Interface (Streamlit/Web Client)] 
                   │
                   ▼
       [REST API Layer (FastAPI)]
                   │
                   ▼
        [RAG Pipeline (LangChain)]
   ┌───────────────┼───────────────┐
   ▼               ▼               ▼
[Loaders]   [Text Splitters]  [Embeddings]
(Factory)                      & ChromaDB
                                   │
                                   ▼
                            [Secure LLM]
