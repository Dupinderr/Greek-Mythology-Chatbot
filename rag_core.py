"""
Shared RAG logic: vector store, source management, and the LangGraph agent.

Imported by the CLI (RAG_Agent.py), the API (api.py) and the Streamlit UI, so
none of these hold their own copy of the retrieval logic.
"""

from dotenv import load_dotenv
import os

# ~/.cache is root-owned on this machine, so huggingface_hub can't read or write it.
# huggingface_hub resolves this at import time, so it must be set before the imports below.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"))

from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_core.tools import tool

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PERSIST_DIRECTORY = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "attention_paper"

MAX_SOURCES = 5
TOP_K = 5

base_llm = ChatGroq(
    model=MODEL,
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

SYSTEM_PROMPT = """
You are a Greek mythology assistant. The user has added classical texts and
retellings — Homer, Hesiod and others — as your sources.

Use the retriever tool for any question about the myths: gods, heroes,
monsters, places, events, or what a particular text says.

Ground every factual claim in the retrieved passages. Each retrieved document
is labelled with its source name; always name that source in your answer.

If the sources do not cover something, say so plainly instead of guessing.
Never invent quotations, characters, or events.

Greetings and questions about what you can do need no retrieval — just answer
naturally and mention which sources you have.
"""


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------

def list_sources():
    """
    Names of the documents currently in the vector store.
    """

    data = vectorstore._collection.get(include=["metadatas"])

    names = {
        m["source"]
        for m in data["metadatas"]
        if m.get("source")
    }

    return sorted(names)


def chunk_count():
    return vectorstore._collection.count()


def add_source(path):
    """
    Load, chunk and embed one document. Returns (ok, message).
    """

    path = os.path.expanduser(path.strip().strip('"').strip("'"))

    if not path.lower().endswith((".pdf", ".txt")):
        return False, "Only .pdf and .txt files are supported."

    if not os.path.exists(path):
        return False, f"File not found: {path}"

    name = os.path.basename(path)

    existing = list_sources()

    if name in existing:
        return False, f"'{name}' is already added."

    if len(existing) >= MAX_SOURCES:
        return False, (
            f"Limit of {MAX_SOURCES} sources reached. "
            f"Delete {PERSIST_DIRECTORY} to start over."
        )

    if path.lower().endswith(".pdf"):
        pages = PyPDFLoader(path).load()
    else:
        # Plain text has no pages, so this is a single document.
        pages = TextLoader(path, encoding="utf-8").load()

    chunks = text_splitter.split_documents(pages)

    # Loaders record whatever path was passed in, so normalise it to the file
    # name — that is what both the source filter and citations match on.
    for chunk in chunks:
        chunk.metadata["source"] = name

    vectorstore.add_documents(chunks)

    return True, f"Added '{name}' — {len(chunks)} chunks."


def remove_source(name):
    """
    Drop every chunk belonging to one source. Returns (ok, message).
    """

    if name not in list_sources():
        return False, f"'{name}' is not a known source."

    vectorstore._collection.delete(where={"source": name})

    return True, f"Removed '{name}'."


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]


def _make_retriever_tool(source):
    """
    Build a retriever tool scoped to one source (or all, when source is None).

    The scope is captured per call rather than read from a global, so that
    concurrent web users can't overwrite each other's selection.
    """

    @tool
    def retriever_tool(query: str) -> str:
        """
        Retrieve relevant passages from the user's sources.
        """

        search_kwargs = {"k": TOP_K}

        if source:
            search_kwargs["filter"] = {"source": source}

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found."

        results = []

        for i, doc in enumerate(docs):

            name = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")

            label = f"{name}, page {page + 1}" if page is not None else name

            results.append(
                f"Document {i+1} [{label}]\n\n{doc.page_content}"
            )

        return "\n\n".join(results)

    return retriever_tool


def build_agent(source=None, on_tool_call=None):
    """
    Compile an agent graph scoped to one source (or all sources).

    Compiling is cheap — the model and embeddings are already loaded — so this
    is done per request to keep the source filter request-local.
    """

    retriever_tool = _make_retriever_tool(source)

    tools = [retriever_tool]
    tools_dict = {t.name: t for t in tools}

    llm = base_llm.bind_tools(tools)

    def call_llm(state: AgentState):

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

        return {"messages": [llm.invoke(messages)]}

    def take_action(state: AgentState):

        results = []

        for t in state["messages"][-1].tool_calls:

            if on_tool_call:
                on_tool_call(t["name"], source or "all sources")

            if t["name"] not in tools_dict:
                result = "Tool not found."
            else:
                result = tools_dict[t["name"]].invoke(t["args"])

            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=t["name"],
                    content=str(result)
                )
            )

        return {"messages": results}

    def should_continue(state: AgentState):

        last_message = state["messages"][-1]

        return (
            hasattr(last_message, "tool_calls")
            and len(last_message.tool_calls) > 0
        )

    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            True: "retriever_agent",
            False: END,
        },
    )

    graph.add_edge("retriever_agent", "llm")

    graph.set_entry_point("llm")

    return graph.compile()


def ask(question, source=None, history=None, on_tool_call=None):
    """
    Answer one question. `history` is an optional list of {role, content}
    dicts holding earlier turns, so callers can support multi-turn chat.
    """

    agent = build_agent(source=source, on_tool_call=on_tool_call)

    messages = []

    for turn in history or []:

        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})

    return result["messages"][-1].text
