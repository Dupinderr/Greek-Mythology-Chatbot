# Greek Mythology RAG Agent

A question-answering agent over five public-domain Greek mythology texts. It answers
only from those texts, names the source it used, and says so when the answer isn't
there.

```
Streamlit UI  ─┐
               ├─→  rag_core  ─→  LangGraph agent ⇄ Chroma
Terminal REPL ─┤                        │
FastAPI       ─┘                     Groq LLM
```

Three front-ends over one shared core, so retrieval behaviour is identical in all of
them.

## Agent, not pipeline

Classic RAG retrieves on every query whether or not it helps. Here retrieval is a
**tool the model chooses to call**:

- **It can skip retrieval** — "what can you do?" needs no search of Homer.
- **It writes its own query** — asked *"how did the hero get home?"*, it searches
  `Odysseus return Ithaca`. The query is a model output, not the raw input.
- **It can search again** — the graph loops, so multi-hop questions work.

```
START → llm ──(no tool call)──→ END
         │
    (tool call)
         ↓
    retriever ──→ back to llm
```

`should_continue` checks the last message for `tool_calls`: present → retrieve, absent
→ that's the final answer.

## Two design notes

**Filtering happens inside the search.** Questions can be scoped to one text. Filtering
*after* retrieval silently returns fewer than `k` chunks, so the filter goes into the
vector search and Chroma applies it during similarity search:

```python
search_kwargs = {"k": 5}
if source:
    search_kwargs["filter"] = {"source": source}
```

**Scope is per-request.** An earlier version held the selected source in a module-level
global — fine for one terminal user, wrong for two concurrent web users. The retriever
tool is now built per request via a closure, so one user's selection can't affect
another's.

## Stack

| Layer | Choice |
|---|---|
| LLM | Groq `openai/gpt-oss-20b` — free tier, supports tool calling |
| Embeddings | `all-MiniLM-L6-v2`, run locally — free and offline |
| Vector store | Chroma on disk — persists, and filters on metadata |
| Orchestration | LangGraph — explicit state machine |
| API | FastAPI |
| UI | Streamlit |

**Corpus:** 3,947 chunks from The Odyssey (Homer), Hesiod & the Homeric Hymns,
Bulfinch's *The Age of Fable*, Kingsley's *The Heroes*, and Hawthorne's *Tanglewood
Tales* — public domain, via Project Gutenberg. Chunked at 1000 characters with 200 of
overlap, retrieving the top 5.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env      # free key: https://console.groq.com
```

**Chat UI** — builds the vector store from `sources/` on first run, which takes a minute:

```bash
streamlit run streamlit_app.py
```

**Terminal REPL** — `/add sources/odyssey.txt` to index a file, `/use 1` to scope to it:

```bash
python RAG_Agent.py
```

**API:**

```bash
uvicorn api:app --port 8000
```

| Method | Path | |
|---|---|---|
| `GET` | `/health` | Reports chunk count and loaded sources, so a process that started without its data is obvious immediately |
| `GET` | `/sources` | Available sources |
| `POST` | `/chat` | `{question, source?, history?}` → `{answer, source_used, tool_called}` |

Set `API_URL` to route the UI through the API instead of retrieving in-process:

```bash
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

## Deploying

Runs on Streamlit Community Cloud from this repo. Add `GROQ_API_KEY` as a root-level key
in the app's **Secrets** — Streamlit exposes those as environment variables. With
`API_URL` unset the UI retrieves in-process, so no second process is needed.

## Known limitations

- **Broad questions retrieve poorly.** "Summarise this book" retrieves 5 semi-arbitrary
  chunks, so the answer reflects those chunks rather than the book. Needs map-reduce
  summarisation.
- **No reranking.** Retrieving 20 and cutting to the best 5 with a cross-encoder would
  measurably improve precision.
- **Pure semantic search.** Hybrid BM25 + vector would catch exact terms and rare proper
  nouns that embeddings blur.
- **Citations vary.** The 20B model usually names the source file but sometimes says
  "Document 3". `gpt-oss-120b`, also free on Groq, follows the instruction more reliably.
- **No evaluation set.** Quality is judged by inspection, so there's no way to know
  whether a change helped.

## Next steps

1. Evaluation set — everything else is guesswork without it
2. Cross-encoder reranking
3. Web search as a second tool, so it can find sources rather than only read given ones
