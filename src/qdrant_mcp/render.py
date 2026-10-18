"""Self-contained HTML viewer for the vault: directory tree + content pane.

No external dependencies: no CDN, no web fonts, no build step. The same page
works in two modes:

- static  - one file on disk with every document embedded in a <script> block
- api     - served by a local HTTP server, content fetched per file on demand
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .vault import Vault

PAGE_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #ffffff; --bg-soft: #f7f8fa; --bg-code: #f2f3f5;
  --fg: #1b1f24; --fg-muted: #6b7280; --border: #e3e6ea; --accent: #2f6feb;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171a; --bg-soft: #1b1f24; --bg-code: #22272e;
    --fg: #e6e8ea; --fg-muted: #9aa3ad; --border: #2c333b; --accent: #6ea8ff;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font-family: var(--sans); background: var(--bg); color: var(--fg); }
#app { display: grid; grid-template-columns: minmax(240px, 24rem) 1fr; height: 100vh; }
aside { border-right: 1px solid var(--border); background: var(--bg-soft); display: flex; flex-direction: column; min-width: 0; }
aside header { padding: .75rem; border-bottom: 1px solid var(--border); }
aside footer { padding: .6rem .75rem; border-top: 1px solid var(--border); color: var(--fg-muted); font-size: .78rem; }
#filter { width: 100%; padding: .45rem .55rem; border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg); color: var(--fg); font: inherit; font-size: .85rem; }
#tree { overflow: auto; flex: 1; padding: .4rem .35rem 1rem; }
.node { display: block; }
.row { display: flex; align-items: center; gap: .3rem; padding: .18rem .3rem; border-radius: 5px;
  cursor: pointer; white-space: nowrap; font-size: .86rem; user-select: none; }
.row:hover { background: var(--bg-code); }
.row.active { background: var(--bg-code); color: var(--accent); font-weight: 600; }
.twisty { width: 1rem; text-align: center; color: var(--fg-muted); font-size: .7rem; }
.icon { color: var(--fg-muted); }
.size { margin-left: auto; color: var(--fg-muted); font-size: .72rem; }
.count { color: var(--fg-muted); font-size: .72rem; margin-left: auto; }
main { overflow: auto; min-width: 0; }
main > header { position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--border);
  padding: .7rem 1.2rem; display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
.path { font-family: var(--mono); font-size: .88rem; }
.meta { color: var(--fg-muted); font-size: .78rem; }
.btn { border: 1px solid var(--border); background: var(--bg-soft); color: var(--fg); border-radius: 6px;
  padding: .2rem .55rem; font: inherit; font-size: .78rem; cursor: pointer; }
.btn:hover { border-color: var(--accent); color: var(--accent); }
.tag { border: 1px solid var(--border); border-radius: 999px; padding: 0 .5rem; font-size: .72rem; color: var(--fg-muted); }
#content { padding: 1.2rem 1.4rem 4rem; }
#content pre { background: var(--bg-code); padding: .8rem 1rem; border-radius: 8px; overflow: auto;
  font-family: var(--mono); font-size: .85rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
#content h1, #content h2, #content h3 { line-height: 1.25; margin: 1.4rem 0 .5rem; }
#content h1 { font-size: 1.5rem; } #content h2 { font-size: 1.25rem; } #content h3 { font-size: 1.05rem; }
#content p { line-height: 1.6; margin: .6rem 0; }
#content code { background: var(--bg-code); padding: .1rem .3rem; border-radius: 4px; font-family: var(--mono); font-size: .85em; }
#content pre code { background: none; padding: 0; }
#content ul, #content ol { line-height: 1.6; }
#content blockquote { margin: .6rem 0; padding-left: .9rem; border-left: 3px solid var(--border); color: var(--fg-muted); }
#content a { color: var(--accent); }
.empty { color: var(--fg-muted); padding: 3rem 1rem; text-align: center; }
@media (max-width: 760px) { #app { grid-template-columns: 1fr; } aside { max-height: 40vh; } }
</style>
</head>
<body>
<div id="app">
  <aside>
    <header><input id="filter" type="search" placeholder="filtrar arquivos..." autocomplete="off"></header>
    <div id="tree"></div>
    <footer id="stats"></footer>
  </aside>
  <main>
    <header>
      <span class="path" id="crumb"></span>
      <span class="meta" id="meta"></span>
      <span id="tags"></span>
      <button class="btn" id="copy">copiar conteúdo</button>
    </header>
    <div id="content"><p class="empty">Selecione um arquivo na árvore.</p></div>
  </main>
</div>
<script id="vault-data" type="application/json">__DATA__</script>
<script>
(function () {
  var CONFIG = __DATA__, API = CONFIG.mode === "api", FILES = CONFIG.files || {};
  var tree = document.getElementById("tree"), content = document.getElementById("content"),
      crumb = document.getElementById("crumb"), meta = document.getElementById("meta"),
      tags = document.getElementById("tags"), filterInput = document.getElementById("filter"),
      stats = document.getElementById("stats"), active = null, source = "";

  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  // Minimal markdown subset: fenced code, headings, lists, quotes, inline spans.
  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, function (_, c) { return "<code>" + c + "</code>"; })
      .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\\*([^*\\n]+)\\*/g, "$1<em>$2</em>")
      .replace(/\\[([^\\]]+)\\]\\(([^)\\s]+)\\)/g, function (_, label, href) {
        var scheme = /^([a-z][a-z0-9+.-]*):/i.exec(href);
        if (scheme && !/^(https?|mailto)$/i.test(scheme[1])) { return label; }
        return '<a href="' + href + '" rel="noreferrer">' + label + "</a>";
      });
  }

  function render(text) {
    var lines = String(text || "").split("\\n"), html = [], list = null, quote = false;
    function closeList() { if (list) { html.push("</" + list + ">"); list = null; } }
    function closeQuote() { if (quote) { html.push("</blockquote>"); quote = false; } }
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (/^```/.test(line)) {
        closeList(); closeQuote();
        var buf = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
        html.push("<pre><code>" + esc(buf.join("\\n")) + "</code></pre>");
        continue;
      }
      var heading = /^(#{1,3})\\s+(.*)$/.exec(line);
      if (heading) {
        closeList(); closeQuote();
        var level = heading[1].length;
        html.push("<h" + level + ">" + inline(heading[2]) + "</h" + level + ">");
        continue;
      }
      var quoteMatch = /^>\\s?(.*)$/.exec(line);
      if (quoteMatch) {
        closeList();
        if (!quote) { html.push("<blockquote>"); quote = true; }
        html.push("<p>" + inline(quoteMatch[1]) + "</p>");
        continue;
      }
      closeQuote();
      var bullet = /^[-*]\\s+(.*)$/.exec(line), ordered = /^\\d+[.)]\\s+(.*)$/.exec(line);
      if (bullet || ordered) {
        var tag = bullet ? "ul" : "ol";
        if (list !== tag) { closeList(); html.push("<" + tag + ">"); list = tag; }
        html.push("<li>" + inline((bullet || ordered)[1]) + "</li>");
        continue;
      }
      closeList();
      if (line.trim() === "") { continue; }
      html.push("<p>" + inline(line) + "</p>");
    }
    closeList(); closeQuote();
    return html.join("\\n");
  }

  function bytes(n) {
    n = Number(n) || 0;
    if (n < 1024) { return n + " B"; }
    if (n < 1048576) { return (n / 1024).toFixed(1) + " KB"; }
    return (n / 1048576).toFixed(1) + " MB";
  }

  function buildTree(entries) {
    var root = { name: "", path: "", dirs: {}, files: [] };
    entries.forEach(function (file) {
      var node = root, parts = file.path.split("/");
      parts.forEach(function (part, index) {
        var isFile = index === parts.length - 1;
        if (isFile) { node.files.push(Object.assign({}, file, { name: part })); return; }
        node.dirs[part] = node.dirs[part] || { name: part, path: node.path ? node.path + "/" + part : part, dirs: {}, files: [] };
        node = node.dirs[part];
      });
    });
    return root;
  }

  function matches(file, query) {
    if (!query) { return true; }
    var haystack = file.path.toLowerCase() + " " + (file.preview || "");
    return haystack.indexOf(query) !== -1;
  }

  function collect(node, query, out) {
    Object.keys(node.dirs).sort().forEach(function (key) {
      collect(node.dirs[key], query, out);
    });
    node.files.forEach(function (file) {
      if (matches(file, query)) { out.push(file); }
    });
    return out;
  }

  function renderTree() {
    var query = filterInput.value.trim().toLowerCase();
    var files = collect(CONFIG.tree, query, []);
    var root = buildTree(files);
    var seen = 0;
    tree.innerHTML = "";

    function draw(node, depth, parent) {
      var keys = Object.keys(node.dirs).sort();
      keys.forEach(function (key) {
        var dir = node.dirs[key];
        var wrapper = document.createElement("div");
        wrapper.className = "node";
        var row = document.createElement("div");
        row.className = "row";
        row.style.paddingLeft = (depth * 0.75 + 0.3) + "rem";
        row.innerHTML = '<span class="twisty"></span><span class="icon">&#9707;</span><span>' + esc(dir.name) + "</span>";
        var kids = document.createElement("div");
        wrapper.appendChild(row);
        wrapper.appendChild(kids);
        parent.appendChild(wrapper);
        var open = Boolean(sessionStorage.getItem("open:" + dir.path) || !query);
        if (open) { row.querySelector(".twisty").textContent = "\\u25be"; }
        row.addEventListener("click", function () {
          var closed = kids.style.display === "none";
          kids.style.display = closed ? "" : "none";
          row.querySelector(".twisty").textContent = closed ? "\\u25be" : "\\u25b8";
          if (closed) { sessionStorage.setItem("open:" + dir.path, "1"); }
          else { sessionStorage.removeItem("open:" + dir.path); }
        });
        if (!open) { kids.style.display = "none"; }
        draw(dir, depth + 1, kids);
      });
      node.files.sort(function (a, b) { return a.name.localeCompare(b.name); }).forEach(function (file) {
        seen++;
        var row = document.createElement("div");
        row.className = "row" + (active === file.path ? " active" : "");
        row.style.paddingLeft = (depth * 0.75 + 1.35) + "rem";
        row.innerHTML = '<span class="icon">&#128196;</span><span>' + esc(file.name) + '</span><span class="size">' + bytes(file.size) + "</span>";
        row.addEventListener("click", function () { openFile(file.path); });
        parent.appendChild(row);
      });
    }

    draw(root, 0, tree);
    stats.textContent = seen + (seen === 1 ? " arquivo" : " arquivos") +
      (query ? (seen === 1 ? " correspondente a \\u201c" : " correspondentes a \\u201c") + query + "\\u201d" : "") +
      (API ? " \\u00b7 served" : " \\u00b7 static");
    if (!seen) { tree.innerHTML = '<p class="empty">nada encontrado</p>'; }
  }

  function openFile(path) {
    active = path;
    var entry = FILES[path];
    crumb.textContent = path;
    meta.textContent = "";
    tags.innerHTML = "";
    renderTree();
    if (!entry) {
      if (!API) { content.innerHTML = '<p class="empty">conteúdo não embutido</p>'; return; }
      content.innerHTML = '<p class="empty">carregando...</p>';
      fetch("/api/files?path=" + encodeURIComponent(path))
        .then(function (response) { return response.json(); })
        .then(function (data) { paint(path, data); })
        .catch(function () { content.innerHTML = '<p class="empty">falha ao carregar</p>'; });
      return;
    }
    paint(path, entry);
  }

  function paint(path, entry) {
    source = entry.content || "";
    var size = entry.size || entry.size_chars || source.length;
    crumb.textContent = path;
    meta.textContent = (entry.updated_at || "") + " \\u00b7 " + bytes(size) +
      " \\u00b7 rev " + (entry.revision || 1) +
      (entry.metadata && Object.keys(entry.metadata).length ? " \\u00b7 " + JSON.stringify(entry.metadata) : "");
    tags.innerHTML = (entry.tags || []).map(function (tag) { return '<span class="tag">' + esc(tag) + "</span>"; }).join("");
    content.innerHTML = render(source);
  }

  document.getElementById("copy").addEventListener("click", function () {
    if (navigator.clipboard) { navigator.clipboard.writeText(source); }
  });

  filterInput.addEventListener("input", renderTree);
  renderTree();

  var initial = location.hash.replace("#", "");
  if (initial && FILES[initial]) { openFile(initial); }
})();
</script>
</body>
</html>
"""


@dataclass(slots=True)
class Snapshot:
    files: list[dict]
    total_chars: int

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "total_chars": self.total_chars,
            "count": len(self.files),
        }


def build_snapshot(vault: Vault, include_content: bool = True, preview_chars: int = 4000) -> Snapshot:
    """Flat, sorted listing of every file, optionally with its content."""
    listing = vault.list_files(limit=1_000_000)
    files: list[dict] = []
    total = 0
    for entry in listing["entries"]:
        record = {
            "path": entry["filepath"],
            "name": entry["name"],
            "dir": entry["dir"],
            "size": entry["size_chars"],
            "updated_at": entry["updated_at"],
            "revision": entry["revision"],
            "tags": entry.get("tags", []),
            "metadata": entry.get("metadata", {}),
        }
        if include_content:
            full = vault.read(entry["filepath"]) or {}
            content = full.get("content", "")
            record["content"] = content
            record["preview"] = content[:preview_chars].lower()
            total += len(content)
        files.append(record)
    files.sort(key=lambda record: record["path"])
    return Snapshot(files=files, total_chars=total)


def _json_for_html(payload: dict) -> str:
    """JSON safe to embed inside a <script> block."""
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_page(
    tree: Snapshot,
    title: str,
    embedded: dict | None = None,
    served: bool = False,
) -> str:
    """Build the full HTML document.

    `embedded` carries the file contents for the static mode; when None the page
    is in API mode and loads contents from /api/files.
    """
    payload = {
        "mode": "api" if served else "static",
        "tree": {"name": "", "path": "", "dirs": {}, "files": tree.files},
        "files": embedded or {},
        "count": len(tree.files),
    }
    return PAGE_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", _json_for_html(payload))


def export_html(
    vault: Vault,
    destination: str,
    title: str = "qdrant vault",
    include_content: bool = True,
    max_mb: int = 64,
) -> dict:
    """Write a single self-contained HTML file with the whole vault embedded."""
    snapshot = build_snapshot(vault, include_content=include_content)
    size_mb = snapshot.total_chars / 1_048_576
    if size_mb > max_mb:
        raise ValueError(
            f"Vault has {size_mb:.1f} MB of content, above the {max_mb} MB export limit. "
            "Use include_content=false or serve the viewer instead."
        )
    path = os.path.abspath(os.path.expanduser(destination))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    embedded: dict[str, Any] = {}
    if include_content:
        embedded = {record["path"]: record for record in snapshot.files}
    page = render_page(snapshot, title=title, embedded=embedded)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return {
        "path": path,
        "files": len(snapshot.files),
        "size_bytes": os.path.getsize(path),
        "content_embedded": include_content,
        "mode": "static",
    }


def build_server(vault: Vault, host: str = "127.0.0.1", port: int = 8765, title: str = "qdrant vault"):
    """Local HTTP server: page from disk-less template, contents via /api/files."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    def handler_factory(*_args, **_kwargs):
        class Handler(BaseHTTPRequestHandler):
            server_version = "qdrant-mcp-viewer"

            def log_message(self, *_args) -> None:  # keep stdout clean for MCP
                pass

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                parsed = urlparse(self.path)
                if parsed.path in ("/", "/index.html"):
                    page = render_page(build_snapshot(vault, include_content=False), title=title, served=True)
                    self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/files":
                    wanted = parse_qs(parsed.query).get("path", [None])[0]
                    if not wanted:
                        self._send(400, b'{"error":"path is required"}', "application/json")
                        return
                    entry = vault.read(wanted)
                    if entry is None:
                        self._send(404, b'{"error":"not found"}', "application/json")
                        return
                    record = {
                        "path": entry["filepath"],
                        "name": entry["name"],
                        "content": entry.get("content", ""),
                        "size": entry["size_chars"],
                        "updated_at": entry.get("updated_at", ""),
                        "revision": entry.get("revision", 1),
                        "tags": entry.get("tags", []),
                        "metadata": entry.get("metadata", {}),
                    }
                    self._send(200, _json_for_html(record).encode("utf-8"), "application/json")
                    return
                self._send(404, b"not found", "text/plain; charset=utf-8")

        return Handler

    return ThreadingHTTPServer((host, port), handler_factory())


__all__ = ["Snapshot", "build_server", "build_snapshot", "export_html", "render_page"]