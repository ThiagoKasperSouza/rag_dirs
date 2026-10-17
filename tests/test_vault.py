from __future__ import annotations

import os
import sys

import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fake_embedder import FakeEmbedder  # noqa: E402
from qdrant_mcp.config import Config  # noqa: E402
from qdrant_mcp.vault import Vault, VaultError  # noqa: E402


@pytest.fixture()
def vault() -> Vault:
    config = Config(collection_prefix="test")
    return Vault(QdrantClient(":memory:"), FakeEmbedder(), config)


NOTES = [
    ("intro.md", "notes/week1", "Qdrant stores dense vectors in collections. Filters narrow results."),
    ("bruno.md", "notes/week1", "Bruno is a city in southern Italy, famous for its old town."),
    ("kafka.md", "notes/week2", "Kafka wrote the novel Amerika and short stories about bureaucracy."),
]


def test_write_then_read_roundtrip(vault: Vault) -> None:
    result = vault.write(name="intro.md", filedir="notes/week1", content=NOTES[0][2])
    assert result["filepath"] == "notes/week1/intro.md"
    assert result["created"] is True
    assert result["revision"] == 1

    entry = vault.read("notes/week1/intro.md")
    assert entry is not None
    assert entry["content"] == NOTES[0][2]
    assert entry["dir"] == "notes"


def test_write_creates_directory_and_is_idempotent(vault: Vault) -> None:
    vault.write(name="a.md", filedir="notes", content="first")
    again = vault.write(name="a.md", filedir="notes", content="second")

    assert again["created"] is False
    assert again["revision"] == 2
    assert [info.name for info in vault.list_dirs()] == ["notes"]
    assert vault.stats()["points"] == 1


def test_same_name_in_different_dirs_are_separate_files(vault: Vault) -> None:
    vault.write(name="intro.md", filedir="notes", content="notes version")
    vault.write(name="intro.md", filedir="docs", content="docs version")

    assert vault.read("notes/intro.md")["content"] == "notes version"
    assert vault.read("docs/intro.md")["content"] == "docs version"
    assert vault.stats()["points"] == 2


def test_metadata_merges_and_tags_union(vault: Vault) -> None:
    vault.write(name="a.md", filedir="notes", content="v1", metadata={"project": "atlas"}, tags=["draft"])
    vault.write(name="a.md", filedir="notes", content="v2", metadata={"author": "ana"}, tags=["final"])

    entry = vault.read("notes/a.md")
    assert entry["metadata"] == {"project": "atlas", "author": "ana"}
    assert entry["tags"] == ["draft", "final"]
    assert entry["created_at"] == vault.read("notes/a.md")["created_at"]


def test_read_missing_file_returns_none(vault: Vault) -> None:
    assert vault.read("notes/nope.md") is None


def test_rejects_paths_that_escape_the_directory(vault: Vault) -> None:
    with pytest.raises(VaultError):
        vault.write(name="../escape.md", filedir="notes", content="x")
    with pytest.raises(VaultError):
        vault.write(name="a.md", filedir="", content="x")


def test_list_files_scopes_and_pagination(vault: Vault) -> None:
    for name, filedir, content in NOTES:
        vault.write(name=name, filedir=filedir, content=content)

    everything = vault.list_files()
    assert everything["total"] == 3

    nested = vault.list_files(filedir="notes/week1", recursive=True)
    assert [entry["filepath"] for entry in nested["entries"]] == [
        "notes/week1/bruno.md",
        "notes/week1/intro.md",
    ]

    flat = vault.list_files(filedir="notes/week1", recursive=False)
    assert {entry["filepath"] for entry in flat["entries"]} == {
        "notes/week1/bruno.md",
        "notes/week1/intro.md",
    }
    assert vault.list_files(filedir="notes", recursive=False)["total"] == 0

    page = vault.list_files(limit=2)
    assert len(page["entries"]) == 2
    assert page["total"] == 3


def test_list_files_on_unknown_directory_raises(vault: Vault) -> None:
    with pytest.raises(VaultError, match="does not exist"):
        vault.list_files(filedir="ghost")


def test_delete_and_move(vault: Vault) -> None:
    vault.write(name="a.md", filedir="notes", content="hello", tags=["keep"])
    vault.write(name="b.md", filedir="notes", content="bye")

    moved = vault.move("notes/a.md", "archive/2026/a.md")
    assert moved["to"] == "archive/2026/a.md"
    assert vault.read("notes/a.md") is None
    assert vault.read("archive/2026/a.md")["tags"] == ["keep"]

    assert vault.delete("notes/b.md")["deleted"] is True
    assert vault.delete("notes/b.md")["deleted"] is False


def test_patch_keeps_untouched_fields(vault: Vault) -> None:
    vault.write(name="a.md", filedir="notes", content="original", metadata={"a": 1}, tags=["t1"])
    vault.patch("notes/a.md", metadata={"b": 2})

    entry = vault.read("notes/a.md")
    assert entry["content"] == "original"
    assert entry["metadata"] == {"a": 1, "b": 2}
    assert entry["tags"] == ["t1"]


def test_search_finds_semantically_close_content(vault: Vault) -> None:
    for name, filedir, content in NOTES:
        vault.write(name=name, filedir=filedir, content=content)

    hits = vault.search("novel about a character named K in Prague")
    assert hits
    assert hits[0]["filepath"] == "notes/week2/kafka.md"


def test_search_scoped_to_directory(vault: Vault) -> None:
    for name, filedir, content in NOTES:
        vault.write(name=name, filedir=filedir, content=content)

    hits = vault.search("bureaucracy or city", dirs=["notes/week1"], limit=5)
    assert {hit["filepath"].split("/", 1)[0] for hit in hits} == {"notes"}
    assert all(hit["filepath"].startswith("notes/week1/") for hit in hits)


def test_search_filters_by_metadata_and_tags(vault: Vault) -> None:
    vault.write(name="a.md", filedir="notes", content="shared words about qdrant", metadata={"p": "atlas"}, tags=["x"])
    vault.write(name="b.md", filedir="notes", content="shared words about qdrant", metadata={"p": "zeta"}, tags=["y"])

    assert [hit["filepath"] for hit in vault.search("qdrant", metadata_filter={"p": "zeta"})] == ["notes/b.md"]
    assert [hit["filepath"] for hit in vault.search("qdrant", tags=["x"])] == ["notes/a.md"]


def test_search_min_score_filters_results(vault: Vault) -> None:
    vault.write(name="a.md", filedir="notes", content="qdrant vectors")
    assert vault.search("qdrant vectors", min_score=0.99) == []


def test_grep_is_exact_and_case_insensitive(vault: Vault) -> None:
    for name, filedir, content in NOTES:
        vault.write(name=name, filedir=filedir, content=content)

    result = vault.grep("bruno")
    assert [match["filepath"] for match in result["matches"]] == ["notes/week1/bruno.md"]
    assert result["matches"][0]["line"] == 1
    assert vault.grep("BRUNO")["matches"]


def test_batch_write_reports_partial_failure(vault: Vault) -> None:
    result = vault.batch_write(
        [
            {"name": "a.md", "filedir": "notes", "content": "ok"},
            {"name": "b.md", "filedir": "", "content": "bad"},
            {"name": "c.md", "filedir": "notes", "content": "also ok"},
        ]
    )
    assert result["written"] == 2
    assert result["failed"] == 1
    assert vault.read("notes/c.md") is not None


def test_directory_lifecycle(vault: Vault) -> None:
    created = vault.create_dir("empty", description="nothing here")
    assert created.name == "empty"
    assert vault.dir_info("empty")["points"] == 0

    vault.write(name="a.md", filedir="empty", content="x")
    assert vault.dir_info("empty")["points"] == 1

    removed = vault.delete_dir("empty")
    assert removed["points_removed"] == 1
    assert [info.name for info in vault.list_dirs()] == []


def test_collection_names_do_not_collide_across_similar_dirs(vault: Vault) -> None:
    vault.write(name="a.md", filedir="my notes", content="one")
    vault.write(name="a.md", filedir="my-notes", content="two")

    assert vault.read("my notes/a.md")["content"] == "one"
    assert vault.read("my-notes/a.md")["content"] == "two"
    assert {info.collection for info in vault.list_dirs()} == {
        vault.collection_for("my notes"),
        vault.collection_for("my-notes"),
    }
