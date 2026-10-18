from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fake_embedder import FakeEmbedder  # noqa: E402
from qdrant_mcp.config import Config  # noqa: E402
from qdrant_mcp.render import build_server, build_snapshot, export_html  # noqa: E402
from qdrant_mcp.vault import Vault  # noqa: E402

DOCS = [
    ("notes/week1/intro.md", "notes", "# Intro\n\nUse `write` then `search`. See [docs](https://example.com).\n"),
    ("notes/week1/deep/api.md", "notes", "```python\nprint('hi')\n```\n\n- alpha\n- beta\n"),
    ("notes/week2/kafka.md", "notes", "Kafka escreveu o romance Amerika.\n"),
    ("docs/readme.md", "docs", "Readme com acentuação: ação, coração.\n"),
]


@pytest.fixture()
def vault() -> Vault:
    instance = Vault(QdrantClient(":memory:"), FakeEmbedder(), Config(collection_prefix="render"))
    for filepath, dir_name, content in DOCS:
        instance.write(name=filepath.split("/", 1)[1], filedir=dir_name, content=content)
    return instance


def test_snapshot_lists_every_file_sorted_with_content(vault: Vault) -> None:
    snapshot = build_snapshot(vault)
    assert [record["path"] for record in snapshot.files] == sorted(record["path"] for record in snapshot.files)
    assert [record["path"] for record in snapshot.files] == [
        "docs/readme.md",
        "notes/week1/deep/api.md",
        "notes/week1/intro.md",
        "notes/week2/kafka.md",
    ]
    assert snapshot.files[0]["content"] == DOCS[3][2]
    assert snapshot.total_chars == sum(len(content) for _, _, content in DOCS)


def test_snapshot_without_content_carries_only_metadata(vault: Vault) -> None:
    snapshot = build_snapshot(vault, include_content=False)
    assert all("content" not in record for record in snapshot.files)
    assert snapshot.total_chars == 0


def test_export_writes_a_single_dependency_free_file(vault: Vault, tmp_path) -> None:
    destination = tmp_path / "nested" / "vault.html"
    result = export_html(vault, destination=str(destination), title="meu vault")

    assert result["files"] == 4
    assert result["mode"] == "static"
    page = destination.read_text(encoding="utf-8")

    assert page.startswith("<!doctype html>")
    assert "meu vault" in page
    # No external resources of any kind.
    assert not re.search(r'(src|href)\s*=\s*"https?://', page)
    assert "//cdn" not in page and "fonts.googleapis" not in page
    assert page.count("<script") == 2


def test_exported_page_embeds_data_safely(vault: Vault, tmp_path) -> None:
    destination = tmp_path / "vault.html"
    export_html(vault, destination=str(destination))

    page = destination.read_text(encoding="utf-8")
    payload = json.loads(
        re.search(r'<script id="vault-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    )

    assert payload["mode"] == "static"
    assert set(payload["files"]) == {path for path, _, _ in DOCS}
    assert payload["files"]["notes/week1/intro.md"]["content"] == DOCS[0][2]
    # The page receives a flat listing and rebuilds the hierarchy in JS.
    assert payload["tree"]["dirs"] == {}
    assert {"path", "name", "dir", "size"} <= set(payload["tree"]["files"][0])
    assert "function buildTree" in page


def test_embedded_payload_cannot_break_out_of_the_script_tag(vault: Vault, tmp_path) -> None:
    payload = "</script><script>alert(1)</script>"
    vault.write(name="evil.md", filedir="notes", content=payload)

    destination = tmp_path / "vault.html"
    export_html(vault, destination=str(destination))
    page = destination.read_text(encoding="utf-8")

    assert "</script><script>alert(1)" not in page
    assert "<script id=\"vault-data\" type=\"application/json\">" in page


def test_export_refuses_vaults_over_the_limit(vault: Vault, tmp_path) -> None:
    with pytest.raises(ValueError, match="export limit"):
        export_html(vault, destination=str(tmp_path / "big.html"), max_mb=0)


def test_served_page_and_api(vault: Vault) -> None:
    httpd = build_server(vault, host="127.0.0.1", port=0)
    Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"

    try:
        page = urllib.request.urlopen(f"{base}/").read().decode()
        assert 'id="vault-data"' in page
        assert "Readme com acentuação" not in page, "api mode must not embed contents"
        assert not re.search(r'(src|href)\s*=\s*"https?://', page)

        document = json.loads(urllib.request.urlopen(f"{base}/api/files?path=docs/readme.md").read())
        assert document["content"] == DOCS[3][2]
        assert document["path"] == "docs/readme.md"
        assert document["size"] == len(DOCS[3][2])

        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/api/files?path=docs/nope.md")
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_builds_a_threading_server(vault: Vault) -> None:
    httpd = build_server(vault, host="127.0.0.1", port=0)
    try:
        assert isinstance(httpd, ThreadingHTTPServer)
        assert httpd.server_address[1] > 0
    finally:
        httpd.server_close()