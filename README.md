# 🤖 VighnaMitra AI

> A production-ready, multi-modal AI assistant with long-term memory, RAG pipelines, MCP tool integration, and a persistent conversation engine — built on LangGraph, FastAPI, and Streamlit.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Graph Flow](#graph-flow)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [MCP Tool Integration](#mcp-tool-integration)

---

## Overview

**VighnaMitra AI** is a full-stack conversational AI assistant that goes far beyond a basic chatbot. It maintains long-term memory about users across sessions, retrieves context from uploaded documents, integrates with external tools via MCP servers, and automatically summarizes long conversations to stay within token limits — all powered by a stateful LangGraph agent pipeline.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Docker Compose                         │
│                                                              │
│  ┌─────────────┐    HTTP     ┌──────────────────────────┐   │
│  │  Streamlit  │ ──────────► │   FastAPI Backend         │   │
│  │  (app.py)   │             │   (src/routes.py)         │   │
│  │  Port 8501  │             │   Port 8005               │   │
│  └─────────────┘             └──────────┬───────────────┘   │
│                                          │                    │
│              ┌───────────────────────────┼──────────────┐    │
│              │                           │              │    │
│     ┌────────▼──────┐    ┌──────────────▼──┐  ┌───────▼──┐ │
│     │  PostgreSQL    │    │  Ollama (gemma3) │  │  MCP     │ │
│     │  + pgvector   │    │  Port 11434      │  │  Server  │ │
│     │  Port 5442    │    └─────────────────┘  │  Port    │ │
│     │               │                          │  8009    │ │
│     │  - checkpoints│                          └──────────┘ │
│     │  - user memory│                                        │
│     │  - user auth  │                                        │
│     │  - doc meta   │                                        │
│     └───────────────┘                                        │
│                                                              │
│     ┌──────────────────────┐                                 │
│     │  FAISS VectorStore   │                                 │
│     │  (local disk)        │                                 │
│     │  - per user/thread   │                                 │
│     └──────────────────────┘                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Graph Flow

The heart of VighnaMitra is a **LangGraph stateful agent** that routes every message through a carefully designed pipeline:

```
__start__
    │
    ▼
init_SystemMessage          ← Loads user memories from PostgresStore,
    │                         synthesizes them into a concise system prompt
    ▼
system_message_summarizer   ← Compresses system message if it exceeds token budget
    │
    ▼
retrieval_router_node       ← Decides: does this query need retrieval?
    │
    ├── need_retrieval ──► retrieval_info_fetcher_node   ← Plans retrieval parameters
    │                           │               │
    │                           ▼               ▼
    │               retrieve_user_     retriever_node
    │               memory_node        (FAISS RAG)
    │                           │               │
    │                           └───► retrieval_join_node ──► chat_node
    │
    └── (no retrieval) ──► summarize_node ──► chat_node
                                                │
                              ┌─────────────────┤
                              │                 │
                         tools branch       __end__ branch
                              │                 │
                        tools_trace_node    remember_node
                              │                 │
                         tool_node          __end__
                              │
                         (back to chat_node)
```

> The graph image is included in the repo root as `current-chatbot-graph.png`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend / API** | FastAPI + Uvicorn |
| **Agent Framework** | LangGraph |
| **LLMs** | Groq (llama-3.3-70b, gpt-oss-120b, qwen3-32b) + Ollama (gemma3:4b) |
| **Embeddings** | HuggingFace `BAAI/bge-large-en-v1.5` |
| **Vector Store (docs)** | FAISS (per user/thread, local disk) |
| **Vector Store (memory)** | PostgreSQL + pgvector via LangGraph AsyncPostgresStore |
| **Checkpointing** | LangGraph AsyncPostgresSaver (PostgreSQL) |
| **Tool Integration** | MCP (Model Context Protocol) via `langchain-mcp-adapters` |
| **Auth** | bcrypt password hashing |
| **Package Manager** | uv |
| **Containerization** | Docker + Docker Compose |

---

## Features

### 🧠 Long-Term User Memory
- Automatically extracts and stores reusable facts about the user (preferences, goals, skills, projects, habits, etc.) after each conversation turn
- Memories are categorized into 18 types: `personal`, `habit`, `interests`, `goals`, `skills`, `dislikes`, `preferences`, `learning_style`, `projects`, `tools`, `constraints`, `knowledge_level`, `career`, `education`, `behavior`, `decisions`, `context`, `health`
- Memories are semantically indexed in PostgreSQL via pgvector and retrieved per-query
- A memory synthesizer compresses all retrieved memories into a ≤650-character context-aware summary injected into the system prompt

### 📄 Document RAG (Retrieval-Augmented Generation)
- Users can upload **PDF** or **TXT** files per conversation thread
- Documents are chunked and indexed into a per-user, per-thread **FAISS** vector store
- A multi-stage retrieval planner decides what to search, how many chunks to fetch, and which retrieval strategy (`similarity` or `mmr`) to use
- RAG results are consolidated by a secondary LLM pass before being fed to the main model

### 🔁 Conversation Summarization
- Automatically summarizes conversation history when token count exceeds ~1800 tokens or message count exceeds 20
- Summaries are stored in graph state and used to maintain context without overflowing the context window
- System message summarization also runs when the system prompt grows too large

### 🛠️ MCP Tool Integration
- Connects to MCP servers (HTTP transport) to expose external tools to the LLM
- Tools are loaded once at startup and cached globally for efficiency
- Currently ships with an **Expense Tracker** MCP server
- Users can add custom online MCP servers from the Streamlit sidebar at runtime

### 🔀 Smart Retrieval Routing
- A lightweight router LLM (`gemma3:4b`) decides before each response whether retrieval is needed
- If yes, a second planner generates typed, optimized retrieval queries for both memory and documents
- This avoids unnecessary vector store lookups for simple general-knowledge queries

### 🔐 User Authentication
- Full signup/login flow with username, email, date-of-birth, and password
- Passwords hashed with **bcrypt**; never stored in plaintext
- Session state managed via Streamlit `st.session_state`

### 💬 Multi-Thread Conversations
- Each user can have multiple independent conversation threads
- Thread IDs are namespaced to the user (`{username}_{uuid}`)
- Full chat history is restored from the LangGraph checkpointer on load
- Sidebar shows all past threads and allows switching between them

---

## Project Structure

```
.
├── app.py                          # Streamlit frontend
├── docker-compose.yml
├── Dockerfile.app
├── Dockerfile.backend
├── pyproject.toml
├── uv.lock
│
├── init-db/
│   ├── 01-extensions.sql           # Enable pgvector
│   └── 02-tables.sql               # accounts_info, uploaded_documents
│
├── data/
│   └── vectorstore/                # FAISS indexes (auto-created, per user/thread)
│
└── src/
    ├── routes.py                   # FastAPI endpoints
    ├── state.py                    # LangGraph state schema (ChatBotState)
    ├── encrypt.py                  # bcrypt + SHA-256 helpers
    ├── user_auth.py                # DB-backed auth functions
    │
    ├── LLMs/
    │   └── load_llm.py             # LLM factory functions (Groq + Ollama)
    │
    ├── chatbots/
    │   ├── chatbot_graphs.py       # Graph assembly + compilation
    │   ├── nodes.py                # All LangGraph node implementations
    │   └── node_conditions.py      # Routing logic (retrieval_router)
    │
    ├── configs/
    │   ├── config_methods.py       # MCP config load/update helpers
    │   └── mcpServers_config.json  # MCP server registry
    │
    └── rag/
        ├── DocumentsLoader.py      # PDF / TXT / URL loaders + chunking
        ├── retrievers.py           # FAISS vectorstore create/load/update
        └── embeddings.py
```

---

## Setup & Installation

### Prerequisites

- Docker & Docker Compose
- A `ollama_data` Docker volume (created externally, used for model persistence):
  ```bash
  docker volume create ollama_data
  ```

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd vighna-mitra-ai
```

### 2. Create `.env` file

```bash
cp .env.example .env
# Fill in the required values (see Environment Variables below)
```

### 3. Pull the Ollama model (first run only)

```bash
docker compose run --rm ollama ollama pull gemma3:4b
```

### 4. Build and start all services

```bash
docker compose up --build
```

### 5. Open the app

```
http://localhost:8501
```

> On first launch, the backend compiles the LangGraph graph and runs `store.setup()` + `checkpointer.setup()` to initialize PostgreSQL tables automatically.

---

## Environment Variables

Create a `.env` file in the project root:

```env
# PostgreSQL connection string
DB_POSTGRES_URL=postgresql://postgres:postgres@postgres:5432/postgres

# Groq API key (for llama-3.3-70b, gpt-oss-120b, qwen3-32b)
GROQ_API_KEY=your_groq_api_key_here
```

---

## API Reference

All endpoints are served by the FastAPI backend on port `8005`.

### `POST /chat`
Send a message and receive the assistant's response.

**Request body:**
```json
{
  "message": "What is my backend framework?",
  "user_id": "alice",
  "thread_id": "alice_abc123"
}
```

**Response:**
```json
{
  "response": {
    "message": "Based on your projects, you typically use FastAPI...",
    "trace": ["Chat Node", "Remember Node"]
  }
}
```

---

### `GET /chat/history?thread_id=<id>`
Fetch the full message history for a thread.

---

### `GET /thread_ids?user_id=<id>`
List all conversation thread IDs for a user.

---

### `GET /is_chat_empty?thread_id=<id>`
Check if a thread has any messages.

---

### `POST /upload`
Upload a PDF or TXT file to the current thread's vector store.

**Form data:** `file`, `thread_id`, `user_id`

---

## How It Works

### System Prompt Initialization (`init_SystemMessage`)

On every new conversation, the node:
1. Queries the PostgresStore for all 18 memory categories for the user
2. Runs a memory synthesis prompt to compress them into a ≤650-character dense summary
3. Injects this summary into a structured system prompt alongside the current date and user ID

This means the assistant "knows" the user from message one, without ever loading irrelevant context.

---

### Retrieval Routing

The `retrieval_router_node` uses a small local model (`gemma3:4b`) to classify the query:

- **No retrieval needed** → goes directly to `summarize_node` → `chat_node`
- **Retrieval needed** → goes to `retrieval_info_fetcher_node` which generates typed retrieval plans for memory and/or documents → fans out to parallel retrieval nodes → joins back before `chat_node`

This two-stage approach keeps retrieval costs low for simple queries while ensuring rich context for complex ones.

---

### Memory Extraction (`remember_node`)

After every assistant response, this node:
1. Fetches the user's existing memories (up to 35) from PostgresStore
2. Asks the LLM to decide if the latest human message contains any new reusable long-term information
3. If yes, deduplicates and stores only genuinely new atomic facts

Memories are stored with a `type` field for filtered semantic search, and a `date` field for temporal reasoning.

---

## MCP Tool Integration

VighnaMitra supports connecting to **MCP servers** for external tool access.

### Built-in: Expense Tracker

The project ships with a local MCP server (`mcp-server-expense-tracker`) that exposes expense tracking tools to the LLM. The server config lives in:

```json
// src/configs/mcpServers_config.json
{
  "expense_tracker": {
    "url": "http://mcp-server:8009/mcp",
    "transport": "http"
  }
}
```

### Adding a custom online MCP server

From the Streamlit sidebar:
1. Select **connectors** → **online**
2. Enter the server name and URL
3. The config is stored in session state and passed to the backend on the next chat request

### Adding a server programmatically

```python
from src.configs.config_methods import update_config_local, ToolConfigLocal

await update_config_local("my_server", ToolConfigLocal(
    command="uv",
    args=["run", "my_mcp_server.py"],
    transport="stdio"
))
```

---

## Notes

- The `ollama_data` volume is declared `external: true` in `docker-compose.yml` — create it once with `docker volume create ollama_data` before first run.
- FAISS vector stores are persisted to `./data/vectorstore/` on the host via the backend container's working directory. Back this up if you want to preserve uploaded document indexes.
- The graph is compiled once at FastAPI startup (`lifespan`) and reused for all requests. If you change node logic, restart the backend container.