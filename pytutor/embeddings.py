import asyncio
import time

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import ConfigDict, PrivateAttr


class OllamaEmbeddingModel(BaseEmbedding):
    """A llama-index embedding model backed by Ollama's /api/embed endpoint.

    Talks to the Ollama HTTP API directly (instead of the ollama python SDK) so
    we control the host, batch size, and retry behavior. Retries transient
    failures, which can happen when Ollama's model runner is busy or reloading.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _host: str = PrivateAttr()
    _retries: int = PrivateAttr()
    _client: httpx.Client = PrivateAttr()
    _progress_callback = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        host: str = "http://localhost:11434",
        embed_batch_size: int = 256,
        timeout: float = 300.0,
        retries: int = 4,
        progress_callback=None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._host = host.rstrip("/")
        self._retries = retries
        self._client = httpx.Client(timeout=timeout)
        self._progress_callback = progress_callback

    def close(self):
        self._client.close()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": texts}
        last_error = None
        for attempt in range(self._retries):
            try:
                resp = self._client.post(f"{self._host}/api/embed", json=payload)
                resp.raise_for_status()
                embeddings = resp.json()["embeddings"]
                if self._progress_callback is not None:
                    self._progress_callback(len(texts))
                return embeddings
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise RuntimeError(
                        f"Embedding model '{self.model_name}' not found in Ollama. "
                        "Pull it first, e.g. `ollama pull nomic-embed-text`."
                    )
                last_error = e
            except httpx.HTTPError as e:
                last_error = e
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"Ollama embedding failed after {self._retries} retries: {last_error}"
        )

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)
