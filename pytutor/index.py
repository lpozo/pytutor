import json
import logging
from pathlib import Path

import click
import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore

from pytutor.config import DATA_DIR, STORAGE_DIR, load_settings
from pytutor.embeddings import OllamaEmbeddingModel

logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

META_FILE = "meta.json"
VECTOR_STORE_BACKEND = "faiss"


def _load_meta(storage_dir: Path) -> dict | None:
    meta_path = storage_dir / META_FILE
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(
    storage_dir: Path, embedding_model: str, docs_version: str, vector_store: str
) -> None:
    meta = {
        "embedding_model": embedding_model,
        "docs_version": docs_version,
        "vector_store": vector_store,
    }
    with open(storage_dir / META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _current_embedding_model() -> str:
    return load_settings()["embedding_model"]


def _embed_model(progress_callback=None) -> OllamaEmbeddingModel:
    settings = load_settings()
    return OllamaEmbeddingModel(
        model_name=settings["embedding_model"],
        host=settings.get("ollama_host", "http://localhost:11434"),
        embed_batch_size=int(settings.get("embed_batch_size", 256)),
        progress_callback=progress_callback,
    )


def _load_index_from_storage(storage_dir: Path, embed_model, backend: str | None):
    """Load the persisted index with the vector store backend it was built with.

    ``backend`` comes from ``meta.json``. The FAISS store persists a binary
    index under the same ``default__vector_store.json`` filename the legacy
    JSON store uses, so the format must be selected explicitly rather than
    sniffed from the files on disk.
    """
    if backend == VECTOR_STORE_BACKEND:
        vector_store = FaissVectorStore.from_persist_dir(str(storage_dir))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=str(storage_dir)
        )
    else:
        storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    return load_index_from_storage(storage_context, embed_model=embed_model)


def get_index(
    storage_dir: Path = STORAGE_DIR,
    data_dir: Path = DATA_DIR,
    rebuild: bool = False,
    docs_version: str | None = None,
    quiet: bool = False,
    embed_progress=None,
):
    """Load or build the document index.

    Returns a tuple ``(index, meta)`` where ``meta`` is a dict carrying the
    embedding model and Python docs version used to build the index.

    ``embed_progress`` is an optional callback ``(processed, total)`` invoked
    as nodes are embedded during a build. When provided, the CLI progress bar
    is suppressed in favor of the callback.
    """
    meta = _load_meta(storage_dir) or {
        "embedding_model": None,
        "docs_version": None,
        "vector_store": None,
    }
    embed_model = _embed_model()

    index_exists = storage_dir.exists() and (storage_dir / "docstore.json").exists()

    if index_exists and not rebuild:
        try:
            if meta.get("embedding_model") != embed_model.model_name and not quiet:
                click.echo(
                    click.style(
                        "Warning: the embedding model in settings.json differs from "
                        "the one used to build the index. Rebuild with `-r` for "
                        "accurate results.",
                        fg="yellow",
                    )
                )
            index = _load_index_from_storage(
                storage_dir, embed_model, meta.get("vector_store")
            )
            if not quiet:
                click.echo(
                    f"Index loaded from storage (docs: {meta.get('docs_version') or 'unknown'})"
                )
            return index, meta
        except Exception as e:
            if not quiet:
                click.echo(click.style(f"Failed to load index: {e}", fg="yellow"))
            else:
                raise RuntimeError(f"Failed to load index: {e}") from e

    if rebuild and index_exists:
        if not quiet:
            click.echo("Rebuilding index...")
    else:
        if not quiet:
            click.echo("Creating new index...")

    if not data_dir.exists():
        if not quiet:
            click.echo(click.style(f"Data directory not found: {data_dir}", fg="red"))
            raise click.Abort()
        raise RuntimeError(
            f"Data directory not found: {data_dir}. "
            "Run `pytutor -u <version>` to download docs and build the index."
        )

    reader = SimpleDirectoryReader(input_dir=str(data_dir), recursive=True)
    documents = reader.load_data()
    if not quiet:
        click.echo(f"Loaded {len(documents)} documents from {data_dir}")

    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        metadata = node.metadata
        for key in ("file_path", "source"):
            raw = metadata.get(key)
            if raw:
                try:
                    metadata[key] = str(Path(raw).relative_to(data_dir))
                except ValueError:
                    pass

    embedding_dim = len(embed_model.get_text_embedding("dimension probe"))

    build_embed_model = embed_model
    if embed_progress is not None:
        embedded = {"count": 0}

        def _on_batch(batch_size: int) -> None:
            embedded["count"] += batch_size
            embed_progress(embedded["count"], len(nodes))

        build_embed_model = _embed_model(progress_callback=_on_batch)
        embed_progress(0, len(nodes))

    vector_store = FaissVectorStore(faiss.IndexFlatIP(embedding_dim))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=embed_progress is None,
        embed_model=build_embed_model,
    )
    index._embed_model = embed_model
    storage_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_dir))

    new_meta = {
        "embedding_model": embed_model.model_name,
        "docs_version": docs_version or meta.get("docs_version"),
        "vector_store": VECTOR_STORE_BACKEND,
    }
    _write_meta(
        storage_dir,
        new_meta["embedding_model"],
        new_meta["docs_version"],
        new_meta["vector_store"],
    )
    if not quiet:
        click.echo("Index created and persisted to storage")

    return index, new_meta
