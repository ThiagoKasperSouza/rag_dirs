from __future__ import annotations

import hashlib
import re
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models

from .config import Config
from .embeddings import Embedder, embed_document_text, embed_query_text

PATH_SEPARATOR = "/"
_SAFE_NAME = re.compile(r"^[^/\\]+$")
_NAMESPACE = uuid.UUID("6f1f0a52-9a2b-5a1e-8c7d-2f4b6e8a0c31")

KEY_PATH = "path"
KEY_NAME = "name"
KEY_DIR = "dir"
KEY_CONTENT = "content"
KEY_METADATA = "metadata"
KEY_CREATED_AT = "created_at"
KEY_UPDATED_AT = "updated_at"
KEY_REVISION = "revision"
KEY_TAGS = "tags"
REGISTRY_SUFFIX = "__dirs"


class VaultError(RuntimeError):
    """Raised for user-correctable problems (bad path, unknown directory...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "dir"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def normalize_relpath(path: str) -> str:
    """Validate a relative path and return it without leading/trailing slashes."""
    cleaned = (path or "").strip().replace("\\", PATH_SEPARATOR)
    cleaned = cleaned.strip(PATH_SEPARATOR)
    if not cleaned:
        return ""
    parts: list[str] = []
    for part in cleaned.split(PATH_SEPARATOR):
        if part in {"", "."}:
            continue
        if part == "..":
            raise VaultError(f"Path '{path}' escapes the directory: '..' is not allowed.")
        if not _SAFE_NAME.match(part) or "\x00" in part:
            raise VaultError(f"Invalid path segment '{part}' in '{path}'.")
        parts.append(part)
    return PATH_SEPARATOR.join(parts)


@dataclass(slots=True)
class DirInfo:
    name: str
    collection: str
    created_at: str
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "dir": self.name,
            "collection": self.collection,
            "created_at": self.created_at,
            "description": self.description,
        }


@dataclass(slots=True)
class SearchHit:
    filepath: str
    score: float
    snippet: str
    updated_at: str
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class Vault:
    """Filesystem-like facade over Qdrant: one collection per top-level directory."""

    def __init__(self, client: QdrantClient, embedder: Embedder, config: Config) -> None:
        self._client = client
        self._embedder = embedder
        self._config = config
        self._prefix = config.collection_prefix
        self._registry = f"{self._prefix}{REGISTRY_SUFFIX}"

    # ------------------------------------------------------------------ paths

    @property
    def registry_collection(self) -> str:
        return self._registry

    def split_path(self, filepath: str) -> tuple[str, str]:
        """`notes/week1/a.md` -> ("notes", "week1/a.md")."""
        cleaned = normalize_relpath(filepath)
        if not cleaned:
            raise VaultError("filepath must not be empty.")
        head, _, tail = cleaned.partition(PATH_SEPARATOR)
        if not tail:
            raise VaultError(
                f"'{filepath}' is a directory, not a file. Use list_files('{cleaned}') or dir_info('{cleaned}')."
            )
        return head, tail

    def join_path(self, dir_name: str, relpath: str) -> str:
        return f"{dir_name}{PATH_SEPARATOR}{normalize_relpath(relpath)}"

    def collection_for(self, dir_name: str) -> str:
        slug = _slug(dir_name)
        suffix = _short_hash(dir_name) if slug != dir_name.strip().lower() else ""
        return f"{self._prefix}-{slug}{'-' + suffix if suffix else ''}"

    def dir_from_collection(self, collection: str) -> str | None:
        cached = self._registry_lookup(collection)
        if cached:
            return cached
        return None

    # ------------------------------------------------------------- registries

    def _registry_lookup(self, collection: str) -> str | None:
        records = self._client.scroll(
            collection_name=self._registry,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="collection", match=models.MatchValue(value=collection))]),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        for point in records[0]:
            payload = point.payload or {}
            if payload.get("collection") == collection:
                return str(payload.get("dir"))
        return None

    def ensure_registry(self) -> None:
        if not self._client.collection_exists(self._registry):
            self._client.create_collection(
                collection_name=self._registry,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )

    def _register_dir(self, dir_name: str, collection: str, description: str = "") -> DirInfo:
        self.ensure_registry()
        point_id = str(uuid.uuid5(_NAMESPACE, f"dir::{dir_name}"))
        now = _now()
        existing = self._client.retrieve(self._registry, ids=[point_id], with_payload=True)
        created_at = (existing[0].payload or {}).get("created_at", now) if existing else now
        self._client.upsert(
            collection_name=self._registry,
            points=[
                models.PointStruct(
                    id=point_id,
                    payload={
                        KEY_DIR: dir_name,
                        "collection": collection,
                        "created_at": created_at,
                        "description": description,
                    },
                    vector=[0.0],
                )
            ],
        )
        return DirInfo(name=dir_name, collection=collection, created_at=created_at, description=description)

    def list_dirs(self) -> list[DirInfo]:
        self.ensure_registry()
        dirs: list[DirInfo] = []
        for point in self._client.scroll(
            collection_name=self._registry, limit=10_000, with_payload=True, with_vectors=False
        )[0]:
            payload = point.payload or {}
            if "dir" not in payload:
                continue
            dirs.append(
                DirInfo(
                    name=str(payload["dir"]),
                    collection=str(payload.get("collection", self.collection_for(str(payload["dir"])))),
                    created_at=str(payload.get("created_at", "")),
                    description=str(payload.get("description", "")),
                )
            )
        return sorted(dirs, key=lambda item: item.name.lower())

    def create_dir(self, dir_name: str, description: str = "") -> DirInfo:
        dir_name = normalize_relpath(dir_name)
        if PATH_SEPARATOR in dir_name:
            raise VaultError("dir must be a single top-level name, not a nested path.")
        collection = self.collection_for(dir_name)
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=self._embedder.dimension, distance=models.Distance.COSINE),
            )
            for field_name in (KEY_DIR, KEY_TAGS):
                with warnings.catch_warnings():
                    # Qdrant's in-memory mode warns that payload indexes are no-ops.
                    warnings.simplefilter("ignore", UserWarning)
                    self._client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
        return self._register_dir(dir_name, collection, description)

    def dir_info(self, dir_name: str) -> dict:
        info = self.create_dir(dir_name)
        return {**info.as_dict(), **self.collection_stats(info.collection)}

    def delete_dir(self, dir_name: str, delete_points: bool = True) -> dict:
        dir_name = normalize_relpath(dir_name)
        if dir_name == "":
            raise VaultError("Refusing to operate on the root directory.")
        collection = self.collection_for(dir_name)
        points = 0
        if self._client.collection_exists(collection):
            points = self.collection_stats(collection)["points"]
            if delete_points:
                self._client.delete_collection(collection_name=collection)
            else:
                self._client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=models.Filter(must=[])),
                )
        self.ensure_registry()
        self._client.delete(
            collection_name=self._registry,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="dir", match=models.MatchValue(value=dir_name))]
                )
            ),
        )
        return {"dir": dir_name, "collection_removed": delete_points, "points_removed": points}

    def collection_stats(self, collection: str) -> dict:
        if not self._client.collection_exists(collection):
            return {"exists": False, "points": 0, "vector_size": None}
        live = self._client.count(collection_name=collection, exact=True).count
        return {
            "exists": True,
            "points": live,
            "vector_size": self._embedder.dimension,
        }

    # ------------------------------------------------------------- read/write

    def _point_id(self, dir_name: str, relpath: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"{dir_name}{PATH_SEPARATOR}{relpath}"))

    def write(
        self,
        name: str,
        filedir: str,
        content: str,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        parent = normalize_relpath(filedir or "")
        if not parent:
            raise VaultError("filedir must name a top-level directory, e.g. filedir='notes'.")
        dir_name, _, rel_dir = parent.partition(PATH_SEPARATOR)
        file_name = normalize_relpath(name)
        if not file_name:
            raise VaultError("name must not be empty.")
        if len(content) > self._config.max_content_chars:
            raise VaultError(
                f"Content is too long ({len(content)} chars, limit {self._config.max_content_chars})."
            )

        relpath = f"{rel_dir}{PATH_SEPARATOR}{file_name}" if rel_dir else file_name
        self.create_dir(dir_name)
        point_id = self._point_id(dir_name, relpath)
        now = _now()
        existing = self._client.retrieve(
            self.collection_for(dir_name), ids=[point_id], with_payload=True, with_vectors=False
        )
        previous = existing[0].payload if existing else None
        merged_metadata = dict((previous or {}).get(KEY_METADATA) or {})
        merged_metadata.update(metadata or {})
        merged_tags = sorted(set(list((previous or {}).get(KEY_TAGS) or []) + list(tags or [])))

        vector = self._embedder.embed_documents([embed_document_text(relpath, content)])[0]
        self._client.upsert(
            collection_name=self.collection_for(dir_name),
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        KEY_DIR: dir_name,
                        KEY_PATH: relpath,
                        KEY_NAME: file_name.rsplit(PATH_SEPARATOR, 1)[-1],
                        KEY_CONTENT: content,
                        KEY_METADATA: merged_metadata,
                        KEY_TAGS: merged_tags,
                        KEY_CREATED_AT: (previous or {}).get(KEY_CREATED_AT, now),
                        KEY_UPDATED_AT: now,
                        KEY_REVISION: int((previous or {}).get(KEY_REVISION, 0)) + 1,
                    },
                )
            ],
        )
        return {
            "filepath": self.join_path(dir_name, relpath),
            "dir": dir_name,
            "created": previous is None,
            "revision": int((previous or {}).get(KEY_REVISION, 0)) + 1,
            "size_chars": len(content),
            "embedding_model": self._embedder.name,
        }

    def batch_write(self, files: list[dict], stop_on_error: bool = False) -> dict:
        results, errors = [], []
        for index, item in enumerate(files):
            try:
                results.append(
                    self.write(
                        name=item["name"],
                        filedir=item["filedir"],
                        content=item.get("content", ""),
                        metadata=item.get("metadata"),
                        tags=item.get("tags"),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported back to the caller
                errors.append({"index": index, "error": str(exc)})
                if stop_on_error:
                    break
        return {"written": len(results), "failed": len(errors), "files": results, "errors": errors}

    def read(self, filepath: str, max_chars: int | None = None, include_content: bool = True) -> dict | None:
        dir_name, relpath = self.split_path(filepath)
        collection = self.collection_for(dir_name)
        if not self._client.collection_exists(collection):
            return None
        records = self._client.retrieve(
            collection_name=collection,
            ids=[self._point_id(dir_name, relpath)],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return self._entry_from_payload(dir_name, records[0].payload or {}, include_content, max_chars)

    def _entry_from_payload(
        self, dir_name: str, payload: dict, include_content: bool, max_chars: int | None
    ) -> dict:
        content = payload.get(KEY_CONTENT, "")
        entry = {
            "filepath": self.join_path(dir_name, str(payload.get(KEY_PATH, ""))),
            "dir": dir_name,
            "name": payload.get(KEY_NAME, ""),
            "size_chars": len(content),
            "created_at": payload.get(KEY_CREATED_AT),
            "updated_at": payload.get(KEY_UPDATED_AT),
            "revision": payload.get(KEY_REVISION, 1),
            "tags": payload.get(KEY_TAGS, []),
            "metadata": payload.get(KEY_METADATA, {}),
        }
        if include_content:
            truncated = max_chars is not None and len(content) > max_chars
            entry["content"] = content[:max_chars] if truncated else content
            entry["truncated"] = truncated
        return entry

    def list_files(self, filedir: str = "", recursive: bool = True, limit: int = 200, offset: int = 0) -> dict:
        """List entries under `filedir` (a top-level dir, or a subpath inside one)."""
        prefix = normalize_relpath(filedir)
        targets = self._scope_targets([prefix] if prefix else [])

        entries: list[dict] = []
        for root_name, sub, collection in targets:
            scroll_filter = models.Filter(
                must=[models.FieldCondition(key=KEY_DIR, match=models.MatchValue(value=root_name))]
            )
            for point in self._client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=10_000,
                with_payload=True,
                with_vectors=False,
            )[0]:
                payload = point.payload or {}
                relpath = str(payload.get(KEY_PATH, ""))
                if sub and not relpath.startswith(sub + PATH_SEPARATOR):
                    continue
                remainder = relpath[len(sub) + 1 :] if sub else relpath
                if not recursive and PATH_SEPARATOR in remainder:
                    continue
                entries.append(self._entry_from_payload(root_name, payload, include_content=False, max_chars=None))

        entries.sort(key=lambda entry: entry["filepath"])
        total = len(entries)
        return {
            "filedir": prefix or "/",
            "recursive": recursive,
            "total": total,
            "entries": entries[offset : offset + limit],
        }

    def delete(self, filepath: str) -> dict:
        dir_name, relpath = self.split_path(filepath)
        collection = self.collection_for(dir_name)
        if not self._client.collection_exists(collection):
            raise VaultError(f"File '{filepath}' not found.")
        existed = self.read(filepath, include_content=False) is not None
        self._client.delete(collection_name=collection, points_selector=[self._point_id(dir_name, relpath)])
        return {"filepath": self.join_path(dir_name, relpath), "deleted": existed}

    def move(self, src: str, dst: str) -> dict:
        entry = self.read(src)
        if entry is None:
            raise VaultError(f"Source '{src}' does not exist.")
        target_dir, target_rel = self.split_path(dst)
        _, source_rel = self.split_path(src)
        metadata = entry.get("metadata") or {}
        tags = entry.get("tags") or []
        self.delete(src)
        written = self.write(
            name=target_rel,
            filedir=target_dir,
            content=entry.get("content", ""),
            metadata=metadata,
            tags=tags,
        )
        return {"from": self.join_path(*self.split_path(src)), "to": written["filepath"]}

    def patch(self, filepath: str, content: str | None = None, metadata: dict | None = None, tags: list[str] | None = None) -> dict:
        """Rewrite a file, keeping fields that were not passed in."""
        entry = self.read(filepath)
        if entry is None:
            raise VaultError(f"File '{filepath}' does not exist.")
        dir_name, relpath = self.split_path(filepath)
        return self.write(
            name=relpath,
            filedir=dir_name,
            content=entry.get("content", "") if content is None else content,
            metadata=metadata,
            tags=tags if tags is not None else (entry.get("tags") or []),
        )

    # ----------------------------------------------------------------- search

    def _scope_targets(self, dirs: list[str] | None) -> list[tuple[str, str, str]]:
        """Resolve `dirs` into (dir_name, subpath_inside_dir, collection) triples."""
        if dirs:
            requested = []
            for entry in dirs:
                normalized = normalize_relpath(entry)
                if not normalized:
                    continue
                root, _, sub = normalized.partition(PATH_SEPARATOR)
                requested.append((root, sub))
        else:
            requested = [(info.name, "") for info in self.list_dirs()]

        resolved: list[tuple[str, str, str]] = []
        for root, sub in requested:
            collection = self.collection_for(root)
            if not self._client.collection_exists(collection):
                if dirs:
                    raise VaultError(f"Directory '{root}' does not exist.")
                continue
            resolved.append((root, sub, collection))
        return resolved

    def search(
        self,
        query: str,
        dirs: list[str] | None = None,
        limit: int | None = None,
        min_score: float = 0.0,
        metadata_filter: dict | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        limit = limit or self._config.default_limit
        vector = self._embedder.embed_documents([embed_query_text(query)])[0]
        hits: list[SearchHit] = []
        for root, sub, collection in self._scope_targets(dirs):
            conditions = [
                models.FieldCondition(key=KEY_DIR, match=models.MatchValue(value=root))
            ]
            for key, expected in (metadata_filter or {}).items():
                conditions.append(
                    models.FieldCondition(key=f"{KEY_METADATA}.{key}", match=models.MatchValue(value=expected))
                )
            for tag in tags or []:
                conditions.append(models.FieldCondition(key=KEY_TAGS, match=models.MatchValue(value=tag)))
            # Over-fetch when scoping to a subdirectory: the exact prefix check happens in Python.
            response = self._client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit * 4 if sub else limit,
                query_filter=models.Filter(must=conditions),
                with_payload=True,
            )
            for scored in response.points:
                payload = scored.payload or {}
                relpath = str(payload.get(KEY_PATH, ""))
                if sub and not relpath.startswith(sub + PATH_SEPARATOR):
                    continue
                content = str(payload.get(KEY_CONTENT, ""))
                hits.append(
                    SearchHit(
                        filepath=self.join_path(root, relpath),
                        score=float(scored.score),
                        snippet=content[:400],
                        updated_at=str(payload.get(KEY_UPDATED_AT, "")),
                        metadata=dict(payload.get(KEY_METADATA) or {}),
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit.as_dict() for hit in hits if hit.score >= min_score][:limit]

    def grep(self, pattern: str, dirs: list[str] | None = None, limit: int = 20, ignore_case: bool = True) -> dict:
        """Literal substring search - no embedding, exact matches."""
        needle = pattern.lower() if ignore_case else pattern
        matches: list[dict] = []
        scanned = 0
        for root, sub, collection in self._scope_targets(dirs):
            scroll_filter = models.Filter(
                must=[models.FieldCondition(key=KEY_DIR, match=models.MatchValue(value=root))]
            )
            for point in self._client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=10_000,
                with_payload=True,
                with_vectors=False,
            )[0]:
                payload = point.payload or {}
                relpath = str(payload.get(KEY_PATH, ""))
                if sub and not relpath.startswith(sub + PATH_SEPARATOR):
                    continue
                scanned += 1
                content = str(payload.get(KEY_CONTENT, ""))
                haystack = content.lower() if ignore_case else content
                index = haystack.find(needle)
                if index < 0:
                    continue
                start = max(0, index - 120)
                matches.append(
                    {
                        "filepath": self.join_path(str(payload.get(KEY_DIR, "")), relpath),
                        "line": content.count("\n", 0, index) + 1,
                        "snippet": content[start : index + len(pattern) + 200].replace("\n", " "),
                    }
                )
                if len(matches) >= limit:
                    return {"query": pattern, "scanned": scanned, "matches": matches, "truncated": True}
        return {"query": pattern, "scanned": scanned, "matches": matches, "truncated": False}

    def stats(self) -> dict:
        dirs = []
        total = 0
        for info in self.list_dirs():
            collection_stats = self.collection_stats(info.collection)
            total += collection_stats["points"]
            dirs.append({**info.as_dict(), **collection_stats})
        return {
            "dirs": len(dirs),
            "points": total,
            "embedding_model": self._embedder.name,
            "vector_size": self._embedder.dimension,
            "qdrant": self._config.qdrant_url if not self._config.qdrant_location else self._config.qdrant_location,
            "directories": dirs,
        }
