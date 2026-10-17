from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_mcp.config import Config  # noqa: E402
from qdrant_mcp.embeddings import (  # noqa: E402
    FastEmbedder,
    clean_text,
    embed_document_text,
    embed_query_text,
)


def test_clean_text_normalises_newlines_without_flattening_paragraphs() -> None:
    assert clean_text("a\r\nb\r\n\r\n\r\n\r\nc  \td") == "a\nb\n\nc d"


def test_document_text_carries_the_path_so_identical_files_stay_distinct() -> None:
    text = embed_document_text("week1/intro.md", "same body")
    assert text == "File: week1/intro.md\n\nsame body"


def test_query_text_is_cleaned() -> None:
    assert embed_query_text("  qdrant\r\n vectors  ") == "qdrant\n vectors"


@pytest.mark.parametrize(
    ("model", "dimension"),
    [
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
    ],
)
def test_dimension_comes_from_the_registry_without_inference(model: str, dimension: int) -> None:
    embedder = FastEmbedder(Config(embed_model=model))
    assert embedder.dimension == dimension
    assert embedder._model is None, "reading the dimension must not load the model"


def test_unknown_model_falls_back_to_a_probe_embedding() -> None:
    class StubModel:
        def embed(self, texts, batch_size=1):
            return [[0.0] * 8 for _ in texts]

    embedder = FastEmbedder(Config(embed_model="not-a-real-model"))
    embedder._lookup_dimension = lambda: None  # type: ignore[method-assign]
    embedder._model = StubModel()

    assert embedder.dimension == 8
