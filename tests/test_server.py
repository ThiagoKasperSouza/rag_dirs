from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fake_embedder import FakeEmbedder  # noqa: E402
from qdrant_mcp.config import Config  # noqa: E402
from qdrant_mcp.server import build_server  # noqa: E402

EXPECTED_TOOLS = {
    "dir_list",
    "dir_create",
    "dir_info",
    "dir_delete",
    "write",
    "batch_write",
    "read",
    "list_files",
    "delete",
    "move",
    "patch",
    "search",
    "grep",
    "stats",
}


@pytest.fixture()
def server():
    config = Config(collection_prefix="srvtest")
    return build_server(config, client=QdrantClient(":memory:"), embedder=FakeEmbedder())


def call(server, tool: str, **arguments) -> dict:
    result = asyncio.run(server.call_tool(tool, arguments))
    assert result.is_error is False, f"{tool} raised: {result.content}"
    return result.structured_content


def test_every_expected_tool_is_registered(server) -> None:
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert EXPECTED_TOOLS <= names


def test_write_read_search_roundtrip_through_the_protocol(server) -> None:
    written = call(server, "write", name="intro.md", filedir="notes/week1", content="qdrant stores vectors")
    assert written["ok"] is True
    assert written["filepath"] == "notes/week1/intro.md"

    read = call(server, "read", filepath="notes/week1/intro.md")
    assert read["content"] == "qdrant stores vectors"

    found = call(server, "search", query="where are vectors stored")
    assert found["results"][0]["filepath"] == "notes/week1/intro.md"

    assert call(server, "dir_list")["dirs"][0]["dir"] == "notes"
    assert call(server, "stats")["points"] == 1


def test_read_of_missing_file_is_reported_as_data_not_an_exception(server) -> None:
    result = call(server, "read", filepath="notes/missing.md")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_user_errors_are_returned_as_data(server) -> None:
    result = call(server, "write", name="../escape.md", filedir="notes", content="x")
    assert result["ok"] is False
    assert ".." in result["error"]


def test_write_schema_documents_the_essentials(server) -> None:
    schema = next(tool.input_schema for tool in asyncio.run(server.list_tools()) if tool.name == "write")
    assert set(schema["required"]) == {"name", "filedir", "content"}
    assert {"metadata", "tags"} <= set(schema["properties"])
    assert json.dumps(schema).count("filedir") >= 1


def test_tools_declare_a_structured_output_schema(server) -> None:
    tools = asyncio.run(server.list_tools())
    assert all(tool.output_schema is not None for tool in tools), "structured output lets clients skip re-parsing"

