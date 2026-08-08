# Greek Mythology RAG Agent

A question-answering agent over five public-domain Greek mythology texts. It answers
only from those texts, names the source it used, and says so when the answer isn't
there.

Three front-ends over one shared core: a terminal REPL, a FastAPI service, and a
Streamlit chat UI.

```
Streamlit UI  ─┐
               ├─→  rag_core  ─→  LangGraph agent ⇄ Chroma
Terminal REPL ─┤                        │
FastAPI       ─┘                     Groq LLM
```

## Agent, not pipeline

Classic RAG embeds and retrieves on every query, whether or not it helps. Here
retrieval is a **tool the model chooses to call**, which buys three things:

- **It can skip retrieval.** "What can you do?" doesn't need a search of Homer.
- **It writes its own query.** Asked *"how did the hero get home?"*, it searches
  `Odysseus return Ithaca` — the query is a model output, not the raw input.
- **It can search again.** The graph loops, so multi-hop questions work.

```
START → llm ──(no tool call)──→ END
         │
    (tool call)
         ↓
    retriever ──→ back to llm
```

`should_continue` checks the last message for `tool_calls`: present → retrieve, absent
→ that's the final answer.

## Two implementation notes

**Source filtering happens inside the search.** Users can scope to one text. Filtering
*after* retrieval silently returns fewer than `k` chunks, so the filter is passed into
the vector search instead and Chroma applies it during similarity search:

```python
search_kwargs = {"k": 5}
if source:
    search_kwargs["filter"] = {"source": source}
```

**Scope is per-request, not global.** An earlier version held the selected source in a
module-level global — fine for one terminal user, broken for two concurrent web users.
The retriever tool is now built per request via a closure, so one user's selection can't
affect another's. Compiling a graph is cheap; the model and embeddings are already
loaded.

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq `openai/gpt-oss-20b` | Free tier, fast, supports tool calling — which the design requires |
| Embeddings | `all-MiniLM-L6-v2`, local | Free and offline; embeddings run over every chunk, so a hosted API is where cost accumulates |
| Vector store | Chroma, on disk | Persists between runs, supports metadata filtering |
| Orchestration | LangGraph | Explicit state machine rather than nested `if`/`while` |
| API | FastAPI | Typed request/response, automatic OpenAPI docs |
| UI | Streamlit | Chat interface, no frontend build |

**Corpus:** 3,947 chunks from The Odyssey (Homer), Hesiod & the Homeric Hymns,
Bulfinch's *The Age of Fable*, Kingsley's *The Heroes*, and Hawthorne's *Tanglewood
Tales* — all public domain, via Project Gutenberg. Chunked at 1000 characters with 200
of overlap, retrieving the top 5.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env      # free key: https://console.groq.com
```

**Streamlit UI** — builds the vector store from `sources/` on first run, which takes a
minute:

```bash
streamlit run streamlit_app.py
```

**Terminal REPL** — `/add sources/odyssey.txt` to index a file, `/use 1` to scope to it:

```bash
python RAG_Agent.py
```

**API:**

```bash
uvicorn api:app --reload --port 8000
```

To point the UI at the API instead of running retrieval in-process, set `API_URL`:

```bash
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness — also reports chunk count and loaded sources |
| `GET` | `/sources` | Available sources |
| `POST` | `/chat` | `{question, source?, history?}` → `{answer, source_used, tool_called}` |

`/health` reports the chunk count deliberately: a container that booted without its data
returns `"status": "empty"` rather than failing later on a user's question.

## Deploying

Runs on Streamlit Community Cloud from this repo — set `GROQ_API_KEY` in the app's
**Secrets** as a root-level TOML key, which Streamlit also exposes as an environment
variable. The UI runs retrieval in-process when `API_URL` is unset, so no separate API
process is needed.

Free HuggingFace Spaces no longer runs CPU compute Spaces on personal accounts, so that
route needs a paid plan.

## Docker

> **Written but never built.** The Dockerfile is statically checked only — the
> development machine didn't have the ~15 GB of free disk a PyTorch image needs. Expect
> to iterate on the first real build.

The vector store is baked in at build time, so the container boots ready to answer.

```bash
docker build -t mythology-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here mythology-rag
```

The key is injected at runtime, never baked into the image. Expect ~1.5–2 GB, dominated
by PyTorch; the Dockerfile takes the CPU-only wheel to avoid ~2 GB of unused CUDA
libraries.

## Known limitations

- **Broad questions retrieve poorly.** "Summarise this book" retrieves 5 semi-arbitrary
  chunks, so the answer reflects those chunks, not the book. Needs hierarchical or
  map-reduce summarisation.
- **No reranking.** Retrieving 20 and reranking to the best 5 with a cross-encoder would
  measurably improve precision.
- **Pure semantic search.** Hybrid BM25 + vector would catch exact terms and rare proper
  nouns that embeddings blur.
- **Citation consistency varies.** The 20B model usually names the source file but
  sometimes says "Document 3". A larger model (`gpt-oss-120b`, also free on Groq)
  follows the instruction more reliably.
- **No evaluation set.** Quality is judged by inspection, so there's no way to know
  whether a change helped.
- **Collection is still named `attention_paper`** — a leftover from this project's
  earlier life as a research-paper Q&A bot.

## Next steps

1. Evaluation set — everything else is guesswork without it
2. Cross-encoder reranking
3. Web or arXiv search tools, so it can find sources rather than only read given ones
