---
title: Greek Mythology RAG Agent
emoji: 🏛️
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.60.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# Greek Mythology RAG Agent

A question-answering agent over five public-domain Greek mythology texts. It answers
**only** from those texts, names the source it used, and says so plainly when the
answer isn't there.

Three front-ends over one shared core: a terminal REPL, a FastAPI service, and a
Streamlit chat UI.

```
Streamlit UI  ─┐
               ├─→  FastAPI  ─→  rag_core  ─→  LangGraph agent ⇄ Chroma
Terminal REPL ─┘                                      │
                                                   Groq LLM
```

---

## What makes it an *agent*, not a pipeline

This is the design decision worth understanding, and it's what separates this from a
standard RAG tutorial.

**Classic RAG is a fixed pipeline.** Every query is embedded, retrieved, and stuffed
into the prompt — always, whether or not retrieval helps.

**Here, retrieval is a tool the model chooses to call.** The consequences:

- **It can skip retrieval.** "Hello, what can you do?" is answered directly; searching
  Homer for a greeting wastes a call and returns nothing useful.
- **It writes its own search query.** Asked *"how did the hero get home?"*, it can search
  for `Odysseus return Ithaca` rather than the user's literal words. The retrieval query
  is a model output, not the raw input.
- **It can search more than once.** The graph loops back, so multi-hop questions work.

The graph is four lines of routing logic:

```
START → llm ──(no tool call)──→ END
         │
    (tool call)
         ↓
    retriever ──→ back to llm
```

`should_continue` inspects the last message for `tool_calls`. Present → retrieve.
Absent → the model has its final answer.

## Filtering happens *inside* the search, not after it

Users can scope questions to one text. The naive implementation retrieves globally and
filters the results afterwards — which silently returns fewer than `k` chunks, and
leaks content when the filter is applied loosely.

Instead, the source name is written into each chunk's metadata at ingestion, and the
filter is passed **into** the vector search:

```python
search_kwargs = {"k": 5}
if source:
    search_kwargs["filter"] = {"source": source}
```

Chroma applies it during the similarity search, so you always get the top `k` chunks
*within* that source. Verified: scoped to one text, a deliberate search for another
text's content returns zero of its chunks.

## Per-request scope, not global state

The first version held the selected source in a module-level global. Fine for one
terminal user; broken the moment two web users are concurrent — one user's selection
would silently change the other's results.

The retriever tool is now built per request via a closure, so scope is request-local:

```python
def _make_retriever_tool(source):
    @tool
    def retriever_tool(query: str) -> str:
        ...  # `source` captured here, not read from a global
    return retriever_tool
```

Compiling a graph is cheap — the model and embeddings are already in memory — so
per-request construction costs effectively nothing.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq `openai/gpt-oss-20b` | Free tier, fast, supports tool calling — which the design requires |
| Embeddings | `all-MiniLM-L6-v2`, local | Free and offline. Embeddings run over every chunk, so a hosted API is where cost accumulates |
| Vector store | Chroma, on disk | Persists between runs; supports metadata filtering |
| Orchestration | LangGraph | Explicit state machine — nodes and conditional edges rather than nested `if`/`while` |
| API | FastAPI | Typed request/response, automatic OpenAPI docs |
| UI | Streamlit | Chat interface in ~120 lines, no frontend build |

**Corpus:** 3,947 chunks from The Odyssey (Homer), Hesiod & the Homeric Hymns,
Bulfinch's *The Age of Fable*, Kingsley's *The Heroes*, and Hawthorne's
*Tanglewood Tales*. All public domain, via Project Gutenberg.

**Chunking:** 1000 characters with 200 of overlap. The overlap matters — without it a
sentence split across a boundary loses its meaning in both halves.

---

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env      # free key: https://console.groq.com
```

Add sources (the vector store starts empty):

```bash
python RAG_Agent.py
```
```
/add sources/odyssey.txt
/use 1
```

**Terminal:**
```bash
python RAG_Agent.py
```

**API:**
```bash
uvicorn api:app --reload --port 8000
```

**UI** (needs the API running):
```bash
streamlit run streamlit_app.py
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness — also reports chunk count and loaded sources |
| `GET` | `/sources` | Available sources |
| `POST` | `/chat` | `{question, source?, history?}` → `{answer, source_used, tool_called}` |

`/health` deliberately reports the chunk count: a container that booted without its
data returns `"status": "empty"` instead of failing later on a user's question.

---

## Docker

> **Not yet built or tested.** The Dockerfile is written and statically checked, but
> the image has never been built — the development machine lacked the ~15 GB of free
> disk a PyTorch image needs. Expect to iterate on the first real build.

The vector store and embedding model are baked in at build time, so the container
boots ready to answer — no embedding on startup, and the source texts don't ship.

```bash
docker build -t mythology-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here mythology-rag
```

The API key is injected at runtime, never baked into the image.

**Deploying:** works on HuggingFace Spaces (Docker SDK, port 7860) or Render. Add
`GROQ_API_KEY` as a secret in the host's settings.

**Image size:** ~1.5–2 GB, dominated by PyTorch. The Dockerfile installs the CPU-only
wheel, which avoids roughly 2 GB of unused CUDA libraries. Dropping to a hosted
embedding API would cut it to ~200 MB, at the cost of the fully-local property.

---

## Known limitations

Worth stating plainly:

- **Broad questions retrieve poorly.** "Summarise this book" retrieves 5 semi-arbitrary
  chunks, so the answer reflects those chunks, not the book. Naive RAG answers
  *"what's in what I retrieved."* Real fix: hierarchical or map-reduce summarisation.
- **No reranking.** Retrieving 20 and reranking with a cross-encoder to the best 5
  would measurably improve precision.
- **Pure semantic search.** Hybrid BM25 + vector search would catch exact terms and
  rare proper nouns that embeddings blur.
- **Citation consistency.** The 20B model sometimes cites "Document 3" instead of the
  source name. A larger model (`gpt-oss-120b`, also free on Groq) follows the
  instruction more reliably.
- **No evaluation set.** Quality is judged by inspection. A question/answer set with
  retrieval-recall and faithfulness metrics is the honest next step — without it,
  there's no way to know whether a change helped.

## Next steps

1. Evaluation set — everything else is guesswork without it
2. Cross-encoder reranking
3. arXiv / web search tools, so it can *find* sources rather than only read given ones
