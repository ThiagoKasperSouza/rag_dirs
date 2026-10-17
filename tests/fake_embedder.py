"""Deterministic in-memory embedder so tests need no model download."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

DIMENSION = 64


class FakeEmbedder:
    """Hash-bag-of-words embedding: similar texts get similar vectors."""

    name = "fake-bow-64"

    @property
    def dimension(self) -> int:
        return DIMENSION

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSION
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            vector[digest[0] % DIMENSION] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(f"File: {text}" if text.startswith("/") else text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
