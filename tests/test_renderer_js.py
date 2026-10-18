from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_mcp.render import Snapshot, render_page  # noqa: E402

NODE = shutil.which("node")


def _renderer_source() -> str:
    """Extract esc/inline/render from the page so they can run outside a browser."""
    page = render_page(Snapshot(files=[], total_chars=0), title="t")
    js = re.findall(r"<script>(.*?)</script>", page, re.S)[-1]
    start = js.index("function esc(")
    end = js.index("function bytes(")
    return js[start:end]


def _run_node(body: str) -> dict:
    script = _renderer_source() + "\n" + body
    completed = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return {key: value.replace("\n", "") for key, value in json.loads(completed.stdout).items()}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_renderer_escapes_html_in_documents():
    out = _run_node(
        "const html = render('<script>alert(1)</script> & <img src=x onerror=y>');"
        "console.log(JSON.stringify({html}));"
    )
    assert "<script>" not in out["html"]
    assert "<img" not in out["html"]
    assert "&lt;script&gt;" in out["html"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_renderer_blocks_script_urls_but_keeps_safe_links():
    out = _run_node(
        "const bad = render('[clique](javascript:alert(1))');"
        "const good = render('[docs](https://example.com/x)');"
        "console.log(JSON.stringify({bad, good}));"
    )
    assert "javascript:" not in out["bad"]
    assert "<a" not in out["bad"]
    assert out["good"] == '<p><a href="https://example.com/x" rel="noreferrer">docs</a></p>'


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_renderer_handles_the_markdown_subset():
    out = _run_node(
        "console.log(JSON.stringify({"
        "head: render('# Titulo'),"
        "list: render('- a\\n- b'),"
        "ordered: render('1. a\\n2. b'),"
        "quote: render('> nota'),"
        "code: render('```\\ncode <b>\\n```'),"
        "inline: render('**b** e *i* e `c`')"
        "}));"
    )
    assert out["head"] == "<h1>Titulo</h1>"
    assert out["list"] == "<ul><li>a</li><li>b</li></ul>"
    assert out["ordered"] == "<ol><li>a</li><li>b</li></ol>"
    assert out["quote"] == "<blockquote><p>nota</p></blockquote>"
    assert out["code"] == "<pre><code>code &lt;b&gt;</code></pre>"
    assert out["inline"] == "<p><strong>b</strong> e <em>i</em> e <code>c</code></p>"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_renderer_closes_open_lists_and_quotes():
    out = _run_node(
        "console.log(JSON.stringify({"
        "mixed: render('- a\\ntexto\\n- b'),"
        "afterQuote: render('> nota\\ntexto')"
        "}));"
    )
    assert out["mixed"] == "<ul><li>a</li></ul><p>texto</p><ul><li>b</li></ul>"
    assert out["afterQuote"] == "<blockquote><p>nota</p></blockquote><p>texto</p>"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_page_contains_no_external_resource():
    page = render_page(Snapshot(files=[], total_chars=0), title="t")
    assert not re.search(r'(src|href)\s*=\s*"(?!#)[a-z]+:', page)
    assert "@import" not in page