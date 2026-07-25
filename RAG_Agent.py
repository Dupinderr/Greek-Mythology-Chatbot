"""
Terminal chat interface for the Greek mythology RAG agent.

All retrieval and agent logic lives in rag_core, which the API and Streamlit
UI share, so behaviour stays identical across the three front-ends.
"""

import rag_core
from rag_core import MAX_SOURCES, add_source, ask, list_sources

# None means "search across every source".
selected_source = None


HELP = """
Commands:
  /add <path>       add a .pdf or .txt source (max 5)
  /list             show sources and which one is selected
  /use <number>     ask questions against that source only
  /use all          ask questions against every source
  /help             show this message
  exit              quit

Anything else is treated as a question.
"""


def show_sources():

    sources = list_sources()

    if not sources:
        print("\nNo sources yet. Add one with:  /add path/to/book.txt")
        return

    print(f"\nSources ({len(sources)}/{MAX_SOURCES}):")

    for i, name in enumerate(sources, start=1):

        marker = "*" if name == selected_source else " "

        print(f"  {marker} {i}. {name}")

    if selected_source is None:
        print("\n  * searching across all sources")


def greet():
    """
    Welcome message shown once at startup.
    """

    print("=" * 50)
    print("GREEK MYTHOLOGY RAG AGENT")
    print("=" * 50)

    sources = list_sources()

    print("\nHello! I'm your guide to Greek mythology.")

    if not sources:

        print("\nI don't have any texts loaded yet, so there's nothing")
        print("for me to read from.")
        print("\nAdd one to get started:")
        print("  /add path/to/book.txt")
        print("\nType /help to see everything I can do.")
        return

    print("\nI can answer questions from these texts:\n")

    for i, name in enumerate(sources, start=1):
        print(f"  {i}. {name}")

    print("\nAsk me about the gods, the heroes, the monsters, or any")
    print("story told in them. I'll quote the source I used.")
    print("\nType /help for commands.")
    print("\nSo — what would you like to know about today?")


def handle_command(text):
    """
    Run a /command. Returns True if the input was a command.
    """

    global selected_source

    if not text.startswith("/"):
        return False

    parts = text.split(maxsplit=1)

    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command == "/help":
        print(HELP)

    elif command == "/list":
        show_sources()

    elif command == "/add":

        if not argument:
            print("\nUsage: /add path/to/book.txt")
        else:
            print("\nEmbedding — this can take a moment...")
            ok, message = add_source(argument)
            print("\n" + message)

    elif command == "/use":

        sources = list_sources()

        if argument.lower() == "all":
            selected_source = None
            print("\nNow searching across all sources.")

        elif argument.isdigit() and 1 <= int(argument) <= len(sources):
            selected_source = sources[int(argument) - 1]
            print(f"\nNow searching '{selected_source}' only.")

        else:
            print("\nUsage: /use <number>  or  /use all")
            show_sources()

    else:
        print(f"\nUnknown command: {command}")
        print(HELP)

    return True


def run():

    greet()

    history = []

    while True:

        question = input("\nQuestion: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if handle_command(question):
            continue

        if not list_sources():
            print("\nNo sources yet. Add one with:  /add path/to/book.txt")
            continue

        answer = ask(
            question=question,
            source=selected_source,
            history=history,
            on_tool_call=lambda name, scope: print(f"\n🔍 Calling tool: {name} ({scope})"),
        )

        print("\nAnswer:\n")
        print(answer)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    run()
