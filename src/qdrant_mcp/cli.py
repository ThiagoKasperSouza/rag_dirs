"""Terminal entry point for the vault viewer: export a static file or serve it."""

from __future__ import annotations

import argparse
import logging

from qdrant_client import QdrantClient

from .config import Config
from .embeddings import FastEmbedder
from .render import build_server, export_html
from .vault import Vault


def build_vault(config: Config | None = None) -> Vault:
    config = config or Config()
    return Vault(QdrantClient(**config.client_kwargs()), FastEmbedder(config), config)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(prog="qdrant-vault", description="Viewer HTML do vault Qdrant")
    parser.add_argument("--url", default=None, help="QDRANT_URL")
    parser.add_argument("--model", default=None, help="EMBED_MODEL")
    subparsers = parser.add_subparsers(dest="command")

    export = subparsers.add_parser("export", help="gera um único arquivo HTML")
    export.add_argument("-o", "--output", default="vault.html")
    export.add_argument("--title", default="qdrant vault")
    export.add_argument("--no-content", action="store_true", help="embute só a árvore")

    serve = subparsers.add_parser("serve", help="sereve o viewer em localhost")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--title", default="qdrant vault")

    args = parser.parse_args()
    config = Config()
    if args.url:
        config.qdrant_url = args.url
    if args.model:
        config.embed_model = args.model
    vault = build_vault(config)

    if args.command == "serve":
        print(f"viewer em http://{args.host}:{args.port}/")
        httpd = build_server(vault, host=args.host, port=args.port, title=args.title)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return

    destination = args.output if args.command == "export" else "vault.html"
    result = export_html(
        vault,
        destination=destination,
        title=getattr(args, "title", "qdrant vault"),
        include_content=not getattr(args, "no_content", False),
    )
    print(f"{result['path']} ({result['files']} arquivos, {result['size_bytes'] / 1024:.1f} KB)")


if __name__ == "__main__":
    main()