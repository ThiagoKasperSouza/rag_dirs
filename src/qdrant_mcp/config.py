from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_COLLECTION_PREFIX = "vault"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Config:
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_location: str | None = field(default_factory=lambda: os.getenv("QDRANT_LOCATION"))
    collection_prefix: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_PREFIX", DEFAULT_COLLECTION_PREFIX)
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )
    embed_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "32")))
    embed_inference_timeout: float = field(
        default_factory=lambda: float(os.getenv("EMBED_TIMEOUT", "60"))
    )
    max_content_chars: int = field(default_factory=lambda: int(os.getenv("MAX_CONTENT_CHARS", "200000")))
    hybrid_search: bool = field(default_factory=lambda: _env_bool("HYBRID_SEARCH", True))
    default_limit: int = field(default_factory=lambda: int(os.getenv("DEFAULT_SEARCH_LIMIT", "8")))

    def client_kwargs(self) -> dict:
        if self.qdrant_location:
            return {"location": self.qdrant_location, **self._common_client_kwargs()}
        return {"url": self.qdrant_url, **self._common_client_kwargs()}

    def _common_client_kwargs(self) -> dict:
        kwargs: dict = {"timeout": self.embed_inference_timeout * 2}
        if self.qdrant_api_key:
            kwargs["api_key"] = self.qdrant_api_key
        return kwargs
