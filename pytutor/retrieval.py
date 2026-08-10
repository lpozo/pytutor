from pathlib import Path
from typing import Any, Dict, List, Optional

from pytutor.config import DATA_DIR, STORAGE_DIR, load_settings
from pytutor.embeddings import OllamaEmbeddingModel
from pytutor.index import _load_index_from_storage, _load_meta


class RetrievalService:
    def __init__(self, index=None, meta: Optional[dict] = None):
        self.settings = load_settings()
        self.embed_model = OllamaEmbeddingModel(
            model_name=self.settings["embedding_model"],
            host=self.settings.get("ollama_host", "http://localhost:11434"),
        )
        if index is None:
            index, meta = self._load()
        self.index = index
        self.meta = meta or {}

    def _load(self):
        if not STORAGE_DIR.exists() or not (STORAGE_DIR / "docstore.json").exists():
            raise RuntimeError(
                f"Index not found in {STORAGE_DIR}. "
                "Run `pytutor -u <version>` to download docs and build the index."
            )
        meta = _load_meta(STORAGE_DIR) or {}
        index = _load_index_from_storage(
            STORAGE_DIR, self.embed_model, meta.get("vector_store")
        )
        return index, meta

    @staticmethod
    def _relative_source(source_path: str) -> str:
        path = Path(source_path)
        try:
            return str(path.relative_to(DATA_DIR))
        except ValueError:
            pass
        parts = path.parts
        if "data" in parts:
            return str(Path(*parts[parts.index("data") + 1 :]))
        return str(path)

    def search(
        self, query: str, top_k: int = 5, section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results: List[Dict[str, Any]] = []
        for node_with_score in nodes:
            node = node_with_score.node
            metadata = getattr(node, "metadata", {}) or {}
            source_path = metadata.get("file_path") or metadata.get("source")
            if source_path:
                source_path = self._relative_source(source_path)

            if section_filter and source_path:
                if section_filter.lower() not in str(source_path).lower():
                    continue

            results.append(
                {
                    "chunk_id": getattr(node, "node_id", None),
                    "content": node.get_content(),
                    "source_path": source_path,
                    "score": getattr(node_with_score, "score", None),
                }
            )
        return results
