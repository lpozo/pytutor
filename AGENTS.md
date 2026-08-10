# AGENTS

## Project Snapshot
- Purpose: `pytutor` is an interactive Python tutor (TUI chatbot) that grounds its
  answers in the official Python documentation via a local llama-index vector index.
- Python: 3.13+ (workspace root [.python-version](.python-version)).
- Package manager: `uv` (workspace member of ask-pydoc; shared venv/lockfile).

## Fast Start Commands
- Install deps: `uv sync`
- CLI help: `uv run pytutor --help`
- Download docs + build index: `uv run pytutor -u 3.14`
- Chat (TUI): `uv run pytutor`
- One-shot question: `uv run pytutor -a "What is a list comprehension?"`
- Rebuild index from existing docs: `uv run pytutor -r`

## Architecture Map
- TUI chat app: [pytutor/app.py](pytutor/app.py)
- CLI entry and flows: [pytutor/cli.py](pytutor/cli.py)
- LLM chat synthesis (Ollama, streaming): [pytutor/tutor.py](pytutor/tutor.py)
- Index load/build + persistence + meta fingerprint: [pytutor/index.py](pytutor/index.py)
- Retrieval service: [pytutor/retrieval.py](pytutor/retrieval.py)
- Python docs download/extract: [pytutor/docs.py](pytutor/docs.py)
- Settings loader: [pytutor/config.py](pytutor/config.py)

## Agent Working Rules
- Keep CLI flags stable in [pytutor/cli.py](pytutor/cli.py):
  - `--update-pydoc/-u` is exclusive; do not break this contract.
  - `-u` downloads docs + rebuilds index; `-r` rebuilds from existing `data/`.
- In [pytutor/index.py](pytutor/index.py), preserve persistence semantics:
  - Load from `storage/` when present unless rebuilding.
  - Abort cleanly when `data/` is missing.
  - Always keep `meta.json` (embedding model + docs version) in sync with the index.
- In [pytutor/tutor.py](pytutor/tutor.py), the streaming protocol yields
  `("token", text)` then `("sources", chunks)`; keep consumers of this contract working.
- The TUI in [pytutor/app.py](pytutor/app.py) must stay responsive: never call
  blocking retrieval on the UI thread (use `asyncio.to_thread` / workers).

## Generated Artifacts and Safety
- User data and settings live in the user space under `~/.pytutor/`:
  - `~/.pytutor/data/` — downloaded Python docs
  - `~/.pytutor/storage/` — persisted vector index + `meta.json`
  - `~/.pytutor/settings.json` — user settings
- The app directory is overridable via the `PYTUTOR_HOME` env var (useful for
  tests/CI). Paths are centralized in [pytutor/config.py](pytutor/config.py);
  do not hardcode `data/` or `storage/` elsewhere.
- On first run, legacy `data/`, `storage/` and `settings.json` from the package
  root are migrated into the user space by `config.prepare_user_space()`.
- Deleting `~/.pytutor/storage/` or `~/.pytutor/data/` is recoverable; regenerate
  via `pytutor -u`.

## Environment and Runtime Notes
- Requires a local Ollama server with:
  - the embedding model from [settings.json](settings.json)
    (default `nomic-embed-text`)
  - the chat model from [settings.json](settings.json) (default `llama3.2`)
- `pytutor -u <version>` requires internet access to download docs from
  docs.python.org.

## Testing and Validation
- No test suite configured yet.
- Validate changes with:
  - `uv run pytutor --help`
  - `uv run pytutor -a "What is a list comprehension?"`
  - For doc updates: `uv run pytutor -u 3.13`
