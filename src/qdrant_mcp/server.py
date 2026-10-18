"""MCP server exposing a Qdrant-backed vault organised as directories."""

from __future__ import annotations

import functools
import logging
import webbrowser
from typing import Any

from mcp.server.mcpserver import MCPServer
from qdrant_client import QdrantClient

from .config import Config
from .embeddings import FastEmbedder
from .render import build_server as build_viewer_server
from .render import export_html as export_vault_html
from .vault import Vault, VaultError

LOGGER = logging.getLogger("qdrant_mcp")

INSTRUCTIONS = """\
Vector store organised as directories: every top-level directory is one Qdrant
collection, and a file is addressed as "<dir>/<subpath>/<name>".

Typical loop:
  1. dir_list()                      - see which directories exist
  2. write(name, filedir, content)   - store or update a file
  3. search(query)                   - semantic lookup across all directories
  4. read(filepath)                  - fetch the full content of a hit

Rules of thumb:
- filedir may nest ("notes/2026/q1"); only the first segment is the directory.
- metadata is merged on write, tags are unioned; pass them to filter later.
- Prefer search() when you do not know the filepath, read() when you do.
- grep() is the exact, no-embedding alternative to search().
"""


def _error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, VaultError):
        return {"ok": False, "error": str(exc)}
    LOGGER.exception("unexpected vault failure")
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_server(
    config: Config | None = None,
    client: QdrantClient | None = None,
    embedder: FastEmbedder | None = None,
) -> MCPServer:
    config = config or Config()
    embedder = embedder or FastEmbedder(config)
    client = client or QdrantClient(**config.client_kwargs())
    vault = Vault(client, embedder, config)
    vault.ensure_registry()

    server = MCPServer("qdrant-vault", instructions=INSTRUCTIONS)

    def guarded(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> dict[str, Any]:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - surfaced to the model as data
                return _error(exc)

        return wrapper

    # ------------------------------------------------------------ directories

    @server.tool()
    @guarded
    def dir_list() -> dict[str, Any]:
        """List every directory in the vault with its point count. Start here."""
        return {"ok": True, "dirs": [info.as_dict() for info in vault.list_dirs()]}

    @server.tool()
    @guarded
    def dir_create(dir: str, description: str = "") -> dict[str, Any]:
        """Create an empty directory (Qdrant collection). Optional one-line description."""
        info = vault.create_dir(dir, description)
        return {"ok": True, **info.as_dict(), **vault.collection_stats(info.collection)}

    @server.tool()
    @guarded
    def dir_info(dir: str) -> dict[str, Any]:
        """Point count and vector size of a single directory."""
        return {"ok": True, **vault.dir_info(dir)}

    @server.tool()
    @guarded
    def dir_delete(dir: str, delete_points: bool = True) -> dict[str, Any]:
        """Delete a directory. delete_points=False clears it but keeps the collection."""
        return {"ok": True, **vault.delete_dir(dir, delete_points)}

    # ------------------------------------------------------------ file system

    @server.tool()
    @guarded
    def write(name: str, filedir: str, content: str, metadata: dict | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        """Store or update one file.

        Args:
            name: file name, may include a subpath ("week1/intro.md").
            filedir: target directory, may nest ("notes/week1"); created on demand.
            content: full text of the file; it is embedded and kept verbatim.
            metadata: arbitrary JSON merged into the file's metadata on update.
            tags: labels unioned across updates, filterable via search(tags=[...]).
        """
        return {"ok": True, **vault.write(name=name, filedir=filedir, content=content, metadata=metadata, tags=tags)}

    @server.tool()
    @guarded
    def batch_write(files: list[dict], stop_on_error: bool = False) -> dict[str, Any]:
        """Store many files at once.

        Each item: {"name": ..., "filedir": ..., "content": ..., "metadata": {...}, "tags": [...]}.
        Embeddings are computed one file at a time, so prefer this over a loop of write().
        """
        return {"ok": True, **vault.batch_write(files, stop_on_error)}

    @server.tool()
    @guarded
    def read(filepath: str, max_chars: int | None = None, include_content: bool = True) -> dict[str, Any]:
        """Fetch a file by its full path ("notes/week1/intro.md").

        Args:
            filepath: "<dir>/<subpath>/<name>".
            max_chars: truncate the returned content (useful for huge files).
            include_content: false returns metadata only - cheaper than read().
        """
        entry = vault.read(filepath, max_chars=max_chars, include_content=include_content)
        if entry is None:
            return {"ok": False, "error": f"not found: {filepath}"}
        return {"ok": True, **entry}

    @server.tool()
    @guarded
    def list_files(filedir: str = "", recursive: bool = True, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        """List files under a directory, without their content.

        Args:
            filedir: directory or subpath; empty lists every file in the vault.
            recursive: false returns only direct children of filedir.
            limit: page size.
            offset: pagination offset, useful together with the returned `total`.
        """
        return {"ok": True, **vault.list_files(filedir=filedir, recursive=recursive, limit=limit, offset=offset)}

    @server.tool()
    @guarded
    def delete(filepath: str) -> dict[str, Any]:
        """Delete a single file."""
        return {"ok": True, **vault.delete(filepath)}

    @server.tool()
    @guarded
    def move(src: str, dst: str) -> dict[str, Any]:
        """Move or rename a file. dst is a full path including the destination directory."""
        return {"ok": True, **vault.move(src, dst)}

    @server.tool()
    @guarded
    def patch(filepath: str, content: str | None = None, metadata: dict | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        """Update a file in place. Omitted fields keep their current value; metadata is merged, tags unioned."""
        return {"ok": True, **vault.patch(filepath, content=content, metadata=metadata, tags=tags)}

    # ---------------------------------------------------------------- search

    @server.tool()
    @guarded
    def search(
        query: str,
        dirs: list[str] | None = None,
        limit: int = 8,
        min_score: float = 0.0,
        metadata_filter: dict | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Semantic search returning snippets; call read() on a hit for the full content.

        Args:
            query: natural-language description of what you are looking for.
            dirs: restrict to these directories or subpaths; empty searches everything.
            limit: maximum number of hits.
            min_score: cosine similarity floor (0.0-1.0, higher = stricter).
            metadata_filter: exact matches on metadata keys, e.g. {"project": "atlas"}.
            tags: require all of these tags.
        """
        hits = vault.search(
            query=query, dirs=dirs, limit=limit, min_score=min_score, metadata_filter=metadata_filter, tags=tags
        )
        return {"ok": True, "query": query, "count": len(hits), "results": hits}

    @server.tool()
    @guarded
    def grep(pattern: str, dirs: list[str] | None = None, limit: int = 20, ignore_case: bool = True) -> dict[str, Any]:
        """Literal substring search (no embeddings) for exact strings, ids or code tokens."""
        return {"ok": True, **vault.grep(pattern, dirs=dirs, limit=limit, ignore_case=ignore_case)}

    @server.tool()
    @guarded
    def stats() -> dict[str, Any]:
        """Vault overview: directories, total points, embedding model and vector size."""
        return {"ok": True, **vault.stats()}

    # ------------------------------------------------------------- human view

    @server.tool()
    @guarded
    def export_html(
        destination: str = "vault.html",
        title: str = "qdrant vault",
        include_content: bool = True,
        max_mb: int = 64,
    ) -> dict[str, Any]:
        """Export the whole vault as ONE self-contained HTML file (no CDN, no build).

        The page has a collapsible directory tree, a filter box and a content pane.

        Args:
            destination: output path; parent directories are created.
            title: document title.
            include_content: false embeds only the tree, for huge vaults.
            max_mb: refuse to build a file larger than this.
        """
        return {"ok": True, **export_vault_html(vault, destination=destination, title=title, include_content=include_content, max_mb=max_mb)}

    @server.tool()
    @guarded
    def serve_html(host: str = "127.0.0.1", port: int = 8765, title: str = "qdrant vault", open_browser: bool = False) -> dict[str, Any]:
        """Serve the same viewer over localhost, loading file contents on demand.

        Blocks until interrupted; use it when the vault is too large for a single file.
        """
        httpd = build_viewer_server(vault, host=host, port=port, title=title)
        if open_browser:
            webbrowser.open(f"http://{host}:{httpd.server_address[1]}/")
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
        return {"ok": True}

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = Config()
    build_server(config).run(transport="stdio")


if __name__ == "__main__":
    main()
