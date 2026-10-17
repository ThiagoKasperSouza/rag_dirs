from __future__ import annotations

import re
from typing import Protocol, Sequence

from .config import Config

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize text before embedding without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def embed_document_text(path: str, content: str) -> str:
    """The text actually sent to the embedding model.

    The path is prefixed so that semantically similar contents stored under
    different names (e.g. `notes/intro.md` vs `docs/intro.md`) stay separable.
    """
    return f"File: {path}\n\n{content}"


def embed_query_text(query: str) -> str:
    return clean_text(query)


class Embedder(Protocol):
    dimension: int
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Lazy wrapper around fastembed so server start-up stays cheap."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model = None
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return self._config.embed_model

    @property
    def dimension(self) -> int:
        """Vector size, read from the fastembed registry so it costs no inference."""
        if self._dimension is None:
            self._dimension = self._lookup_dimension() or len(self.embed_documents(["dimension probe"])[0])
        return self._dimension

    def _lookup_dimension(self) -> int | None:
        from fastembed import TextEmbedding

        for entry in TextEmbedding.list_supported_models():
            if entry.get("model") == self._config.embed_model and entry.get("dim"):
                return int(entry["dim"])
        return None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._config.embed_model)
        return self._model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = list(model.embed(list(texts), batch_size=self._config.embed_batch_size))
        return [list(map(float, vector)) for vector in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
