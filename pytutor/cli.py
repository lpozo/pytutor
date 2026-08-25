import click

from pytutor.app import PyTutorApp
from pytutor.config import (
    DATA_DIR,
    STORAGE_DIR,
    SettingsError,
    load_settings,
    prepare_user_space,
)
from pytutor.docs import latest_docs_version, update_python_docs
from pytutor.index import get_index
from pytutor.retrieval import RetrievalService
from pytutor.tutor import Tutor


@click.command("pytutor")
@click.option(
    "--update-pydoc",
    "-u",
    metavar="VERSION",
    help="Download Python docs for VERSION (e.g. 3.14), rebuild the index, and exit.",
)
@click.option(
    "--ask",
    "-a",
    "ask_question",
    metavar="QUESTION",
    help="Ask a single question and print the answer.",
)
def main(update_pydoc, ask_question):
    """An interactive Python tutor grounded in the official Python docs."""
    prepare_user_space()
    if update_pydoc:
        if ask_question:
            raise click.UsageError(
                "--update-pydoc/-u is exclusive and cannot be used with other options."
            )
        update_python_docs(update_pydoc)
        get_index(rebuild=True, docs_version=update_pydoc)
        click.echo("\nDocumentation updated and index rebuilt.")
        click.echo("Run `pytutor` to start a chat session.")
        return

    try:
        settings = load_settings()
    except SettingsError as e:
        raise click.ClickException(str(e))

    if ask_question:
        _ask_one_shot(settings, ask_question)
        return

    PyTutorApp(settings=settings, build_retrieval=build_retrieval).run()


def _ensure_index(status=None, download_progress=None, embed_progress=None):
    """Load the persisted index, or bootstrap it (download docs + build).

    ``status``, ``download_progress`` and ``embed_progress`` are optional
    callbacks used by the TUI to render progress. When omitted the CLI prints
    its own progress bars.
    """
    if STORAGE_DIR.exists() and (STORAGE_DIR / "docstore.json").exists():
        return get_index(quiet=True)

    docs_version = None
    if not DATA_DIR.exists():
        docs_version = latest_docs_version()
        if status is not None:
            status(f"Downloading Python {docs_version} documentation…")
        update_python_docs(docs_version, progress=download_progress)
    if status is not None:
        status("Creating index…")

    return get_index(
        rebuild=True,
        docs_version=docs_version,
        quiet=True,
        embed_progress=embed_progress,
    )


def build_retrieval(status=None, download_progress=None, embed_progress=None):
    """Load or build the index and wrap it in a RetrievalService."""
    index, meta = _ensure_index(
        status=status,
        download_progress=download_progress,
        embed_progress=embed_progress,
    )
    return RetrievalService(index=index, meta=meta), meta


def _ask_one_shot(settings, question):
    import asyncio

    retrieval, meta = build_retrieval()
    tutor = Tutor(settings, retrieval)
    docs_version = meta.get("docs_version")

    async def run():
        parts = []
        sources = []
        async for kind, payload in tutor.stream_answer(
            question, history=[], docs_version=docs_version
        ):
            if kind == "token":
                parts.append(payload)
                click.echo(payload, nl=False)
            else:
                sources = payload
        click.echo()
        source_lines = "\n".join(
            f"- {s['source_path']}" for s in sources if s.get("source_path")
        )
        if source_lines:
            click.echo("\nSources:")
            click.echo(source_lines)

    asyncio.run(run())
