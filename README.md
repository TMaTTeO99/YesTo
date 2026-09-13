# YesTo — The AI that never says no.

## About this project

This is a personal project built to explore and implement the agentic patterns from Antonio Gulli's *Agentic Design Patterns*. It's under active development — some parts are still being refactored, and not every corner has had the same level of polish. Treat it as a working exploration of multi-agent architectures with LangGraph, not a production-hardened system.

YesTo is a **multi-agent conversational assistant** built on [LangGraph](https://langchain-ai.github.io/langgraph/). It routes every user message through a reflection loop and, when the request requires real data instead of general knowledge, hands it off to a **plan-and-execute sub-graph** that dispatches work to specialized agents — a database agent, a RAG (knowledge-base) agent, and a web-search agent — before synthesizing everything into a single answer.

It supports text and voice input, persists conversation state per session in PostgreSQL (via LangGraph checkpointing), and can retrieve information both from an internal document knowledge base (PGVector) and the live web.

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Database setup](#database-setup)
- [Running the app](#running-the-app)
- [Testing](#testing)
- [Development commands — what they are and why](#development-commands--what-they-are-and-why)
- [Fine-tuning (optional)](#fine-tuning-optional)
- [Known limitations](#known-limitations)

## How it works

Every user turn goes through two graphs, both built with `langgraph.graph.StateGraph`.

### 1. Main graph — routing & reflection ([graph.py](graph.py))

```
START → router → [domanda | incomprensibile]

domanda → critica_domanda ─┬─(correggi_domanda)→ domanda   (retry loop, up to 5 attempts)
                            └─(clean)→ clean

clean ─┬─(go_to_end)→ END
       └─(go_to_planning)→ planning_agent → END

incomprensibile → END
```

- **router** ([nodes/router.py](nodes/router.py)) classifies the input as `domanda` (a real request) or `incomprensibile` (gibberish), using conversation history for context.
- **domanda** ([nodes/qa.py](nodes/qa.py)) tries to answer directly from general knowledge/chat history. If the request needs real data (DB records, document content, web/real-time info), it flags `needs_tools=True` and the turn is routed to planning instead.
- **critica_domanda** critiques the draft answer (an LLM-as-judge check) and can send it back for a correction — capped at 5 attempts to avoid infinite loops.
- **clean** resets per-turn state and decides whether to hand off to the planning sub-graph.

### 2. Planning sub-graph — plan & execute ([agents/planning_graph.py](agents/planning_graph.py))

```
START → planning_init → planner_checker ─┬─(good_plan)→ tools_supervisor
                                          └─(bad_plan)→ replanner

tools_supervisor → replanner ─┬─(continue_execution)→ tools_supervisor
                               └─(go_to_final_response)→ merge_tools_output → END
```

- **planning_init** ([nodes/planning.py](nodes/planning.py)) breaks the request down into the minimum number of sub-tasks needed to answer it.
- **planner_checker** sanity-checks the generated plan before execution starts.
- **tools_supervisor** ([agents/executor_graph.py](agents/executor_graph.py)) is itself a small graph: for each sub-task it picks one specialized executor —
  - `rag_agent` — searches the internal knowledge base (default choice for anything document/knowledge related),
  - `db_agent` — lists tables, inspects schema, creates tables,
  - `web_search_agent` — used only when the user explicitly asks for it, or as a fallback once RAG/DB have already failed for that request.
- **replanner** inspects the last tool result: on success it either continues the plan or (via an early-stop check) decides the goal is already met; on failure it asks the LLM for a corrective plan.
- **merge_tools_output** synthesizes every tool result collected so far into one final natural-language answer, in the same language as the user's request.

### Tools ([tools/](tools/))

| Tool | File | Purpose |
|---|---|---|
| `elenco_tabelle_db`, `create_table`, `find_table_info` | `tools/db_tools.py` | List/inspect/create PostgreSQL tables. Identifiers and column types are validated against an allowlist before any DDL runs. |
| `search_knowledge_base` | `tools/rag_tools.py` | Similarity search over ingested documents (PGVector). |
| `web_search_tool` | `tools/web_tools.py` | DuckDuckGo web search for external/real-time information. |

Every tool returns a JSON payload with `success`, `receipt`, `summary`, and `details` — the graph uses `success` for control flow and `summary`/`details` when composing the final answer.

### Memory

- **Turn-level state**: LangGraph checkpointing persists `AgentState` per `session_id` (mapped to a LangGraph `thread_id`) in PostgreSQL, so a conversation can be resumed across runs.
- **Chat history compaction**: once a conversation exceeds `MAX_CHAT_HISTORY` messages, older turns are summarized (`chains/summary_chain.py`) instead of being sent to the model in full.
- **Document knowledge base**: PDFs/text files ingested via `/ingest` are chunked and embedded into a PGVector collection (`memory/vector_store.py`), separate from the conversational checkpoint tables.

## Project structure

```
agents/      LangGraph sub-graph assembly (planning_graph, executor_graph)
chains/      Prompt + LLM (+ structured output) pipelines — one per decision point
nodes/       Graph node functions (the logic that runs at each graph step)
tools/       @tool-decorated functions callable by the executor agents
memory/      PGVector-backed vector store for conversation memory and document RAG
state.py     TypedDict schemas shared across the graphs
config.py    Central singleton for the LLM, DB engine, web-search client, Whisper model
main.py      CLI entry point (text/voice conversation loop, /ingest command)
tests/       pytest unit tests for routing and node logic
```

## Requirements

- Python 3.10+
- A PostgreSQL instance with the [pgvector](https://github.com/pgvector/pgvector) extension available
- [Ollama](https://ollama.com/) running locally with a `llama3.1` model pulled (or another chat model — see [Configuration](#configuration))
- `libportaudio2` (system package, required by `sounddevice` for the voice-input feature)

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

System dependency for voice input:

```bash
sudo apt-get install libportaudio2
```

Pull the base LLM in Ollama (only needed once):

```bash
ollama pull llama3.1
```

## Configuration

Copy `.env.example` to `.env` and fill in the values:

| Variable | Required | Description |
|---|---|---|
| `DB_USER` | yes | PostgreSQL user |
| `DB_PASSWD` | yes | PostgreSQL password |
| `DB_HOST` | yes | PostgreSQL host (e.g. `localhost`) |
| `DB_PORT` | yes | PostgreSQL port (default `5432`) |
| `DB_NAME` | yes | PostgreSQL database name |
| `DEBUG` | no | `True`/`False` — enables verbose step-by-step logging of the graph execution |
| `TEST_FINE_TUNING` | no | `True`/`False` — switches the LLM from `llama3.1` to a locally fine-tuned Ollama model (see [Fine-tuning](#fine-tuning-optional)) |

## Database setup

YesTo uses **one PostgreSQL database for two things**: LangGraph's own checkpoint tables (conversation state) and two PGVector collections (conversation-memory embeddings and document embeddings for RAG). Both are created automatically at startup — the only manual step is making sure PostgreSQL has the `vector` extension available.

If you don't already have a Postgres instance, the quickest way to get one on a fresh machine is Docker:

```bash
docker run -d \
  --name yesto-postgres \
  -e POSTGRES_USER=<db_user> \
  -e POSTGRES_PASSWORD=<db_password> \
  -e POSTGRES_DB=<db_name> \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

The `pgvector/pgvector` image ships the extension already compiled, which avoids the manual install step below. If you're using a plain `postgres` image (or a Postgres instance you don't control the image for) instead, install the extension inside the container:

```bash
docker exec -it <container_name> bash
apt-get update && apt-get install -y postgresql-16-pgvector
```

Then, from inside `psql` (or `docker exec -it <container_name> psql -U <db_user> -d <db_name>`), enable the extension in the target database — this only needs to run once per database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Verify it's active:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

Point `.env` at this instance (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWD`, `DB_NAME`) and you're ready to run the app — `main.py` calls `PostgresSaver.setup()` and `init_rag()` on startup, which create every table/collection they need.

## Running the app

```bash
python3 main.py
```

You'll be asked for a **session ID** (this becomes the LangGraph `thread_id`, so reusing the same ID resumes that conversation's memory), then dropped into a loop:

- type a message to talk to the agent
- `/ingest <path/to/file.pdf|.txt>` — index a document into the RAG knowledge base
- `/voice` — record from the microphone and transcribe it with Whisper instead of typing
- `1` — exit

## Testing

```bash
source env/bin/activate
pytest tests/ --cov=. --cov-report=term-missing
```

## Development commands — what they are and why

These are the exact commands used while developing YesTo day to day, kept here so the same environment can be reproduced on any machine — not just mine.

**Setting up Postgres with pgvector from scratch** (see [Database setup](#database-setup) above for the full explanation of *why* each step is needed):
```bash
docker run -d --name yesto-postgres -e POSTGRES_USER=<db_user> -e POSTGRES_PASSWORD=<db_password> -e POSTGRES_DB=<db_name> -p 5432:5432 pgvector/pgvector:pg16
docker exec -it yesto-postgres bash
apt-get update && apt-get install -y postgresql-16-pgvector
docker exec -it yesto-postgres psql -U <db_user> -d <db_name>
```
```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

**Installing the RAG-specific Python dependencies** — these are already pinned in `requirements.txt`, but if you're adding RAG support to a bare environment by hand, this is the minimal set:
```bash
pip3 install pgvector langchain-postgres sentence-transformers
```

**Enabling voice input** — `sounddevice` (used by `main.py`'s `/voice` command) links against PortAudio, which isn't a Python package:
```bash
sudo apt-get install libportaudio2
```

**Running tests with coverage** — used before every commit to make sure a change to one chain/node didn't silently break another:
```bash
pytest tests/ --cov=. --cov-report=term-missing
```

## Fine-tuning (optional)

`config.py` can load a locally fine-tuned Ollama model instead of the base `llama3.1` by setting `TEST_FINE_TUNING=True` in `.env` — this points `ChatOllama` at whatever model tag is configured in `config.py`. The training/validation datasets and the fine-tuning pipeline itself are intentionally not part of this repository (see `.gitignore`); this flag exists purely to A/B test a locally-trained checkpoint against the base model on the same agent graph.

## Known limitations

- `agents/planning_graph.py` sends the final merged answer straight to `END` without re-checking it against the original request — a later retry loop there is a planned improvement, not yet implemented.
- `tools/browser_tools.py` (Playwright-based page browsing) is scaffolded but currently disabled/commented out — not wired into any executor.
- Web search results, RAG results, and DB results all go through the same LLM-as-judge critique for hallucination checking; this catches confidently-wrong answers but is not a substitute for a deterministic grounding check.
