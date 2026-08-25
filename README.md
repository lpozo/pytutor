# pytutor

An interactive Python tutor that grounds its answers in the official Python
documentation. Chat from the terminal (TUI) or ask one-off questions; answers
are synthesized by a local LLM over a local vector index of the docs.

## Requirements

- Python 3.13+ managed with `uv`.
- A local [Ollama](https://ollama.com) server with two models:
  - an embedding model (default `nomic-embed-text`)
  - a chat model (default `llama3.2`)

## Install

```console
uv sync
```

## Update the docs and build the index

Downloads the Python docs for a version and rebuilds the vector index:

```console
uv run pytutor -u 3.14
```

To rebuild the index from already-downloaded docs, delete `~/.pytutor/storage/`
and run `pytutor` again (it will auto-detect missing docs and rebuild).

## Chat

```console
uv run pytutor
```

- `Ctrl+R` starts a new conversation.
- `Ctrl+C` quits.

## Ask a single question

```console
uv run pytutor -a "What is a list comprehension?"
```

## Configuration

Settings live in `~/.pytutor/settings.json` (not the package root). Edit them
to change the embedding/chat models, retrieval `top_k`, embed batch size, or
the Ollama host:

```json
{
  "embedding_model": "nomic-embed-text",
  "chat_model": "llama3.2",
  "top_k": 5,
  "embed_batch_size": 512,
  "ollama_host": "http://localhost:11434"
}
```

On first run, `data/`, `storage/` and `settings.json` from the package root are
migrated into `~/.pytutor/`. Set the `PYTUTOR_HOME` env var to use a different
app directory (useful for testing).
