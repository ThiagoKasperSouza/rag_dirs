# qdrant-mcp

Servidor MCP que expõe um banco vetorial **Qdrant** com semântica de **sistema de
arquivos**: cada diretório de primeiro nível é uma *collection* do Qdrant e cada
arquivo é um *point* dentro dela.

```
vault-notes         -> collection `vault-notes`
  week1/intro.md    -> point, payload.path = "week1/intro.md"
  week1/bruno.md
  week2/kafka.md
vault-docs
  readme.md         -> point, payload.path = "readme.md"
```

Um `filepath` é sempre `"<dir>/<subcaminho>/<nome>"`. Só o primeiro segmento
define a collection; o resto é o caminho interno do arquivo. O conteúdo do
arquivo é o **payload** (guardado verbatim) e o **embedding** do texto é o
vetor — por isso `read` devolve o texto exato e `search` devolve o trecho.

## Tools

| Tool | Assinatura | Para quê |
|---|---|---|
| `write` | `write(name, filedir, content, metadata?, tags?)` | grava/atualiza um arquivo (idempotente por caminho) |
| `read` | `read(filepath, max_chars?, include_content?)` | lê o conteúdo completo de um arquivo |
| `search` | `search(query, dirs?, limit, min_score, metadata_filter?, tags?)` | busca semântica, retorna snippet + score |
| `grep` | `grep(pattern, dirs?, limit, ignore_case)` | busca literal exata, sem embedding |
| `list_files` | `list_files(filedir, recursive, limit, offset)` | lista arquivos sem conteúdo |
| `delete` | `delete(filepath)` | remove um arquivo |
| `move` | `move(src, dst)` | move/renomeia preservando metadata e tags |
| `patch` | `patch(filepath, content?, metadata?, tags?)` | atualiza campos isolados sem reescrever o resto |
| `batch_write` | `batch_write(files[], stop_on_error?)` | grava vários arquivos |
| `dir_list` | `dir_list()` | lista diretórios com contagem de pontos |
| `dir_create` | `dir_create(dir, description?)` | cria diretório/collection vazia |
| `dir_info` | `dir_info(dir)` | pontos e tamanho do vetor de um diretório |
| `dir_delete` | `dir_delete(dir, delete_points?)` | apaga o diretório |
| `stats` | `stats()` | visão geral do vault |

Semântica que importa:

- `write` **sobrescreve** o mesmo caminho e incrementa `revision`; `metadata` é
  **mesclado** e `tags` são **união** entre versões. `created_at` é preservado.
- `filedir` cria o diretório sob demanda, com subcaminhos (`notes/2026`).
- Caminhos com `..` são rejeitados — nada escapa do diretório.
- Erros (dir inexistente, arquivo ausente) voltam como `{"ok": false, "error": ...}`,
  nunca como exceção: o modelo consegue se recuperar.

## Rodando

```bash
# 1. Qdrant
docker compose up -d

# 2. dependências
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 3. testes (Qdrant em memória, sem download de modelo)
.venv/bin/python -m pytest -q

# 4. servidor
.venv/bin/qdrant-mcp
```

## Registro no opencode

```jsonc
{
  "mcp": {
    "qdrant-vault": {
      "type": "local",
      "command": ["/your/path/to/rag_dirs/.venv/bin/qdrant-mcp"],
      "environment": {
        "QDRANT_URL": "http://localhost:6333",
        "EMBED_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
      },
      "enabled": true
    }
  }
}
```

## Configuração

| Variável | Padrão | Descrição |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | endpoint do Qdrant |
| `QDRANT_API_KEY` | — | auth, se habilitada |
| `QDRANT_COLLECTION_PREFIX` | `vault` | prefixo das collections (`vault-notes`) |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | qualquer modelo do fastembed |
| `MAX_CONTENT_CHARS` | `200000` | limite por arquivo |
| `DEFAULT_SEARCH_LIMIT` | `8` | `limit` padrão da busca |

> Trocar `EMBED_MODEL` com dados já gravados exige recriar as collections
> (dimensões e espaçamento vetorial mudam).

## Decisões de projeto

- **Collection por diretório** dá isolamento real: busca por diretório é uma
  query só, listar é um scroll de uma collection, e apagar um diretório é
  `delete_collection`.
- **Point id = uuid5("dir/path")** — reescrever o mesmo arquivo é *upsert* no
  mesmo ponto, sem duplicar.
- **Registro de diretórios** em uma collection própria (`vault__dirs`) guarda
  nome original ↔ collection, então nomes com acento/espaço (`minhas notas`)
  voltam corretamente, e o mapping é reconstruível.
- **Prefixo de subdiretório é filtrado em Python**, não por índice: o Qdrant não
  faz match de prefixo em índice `keyword`, então a busca sobre-fetches (4×) e
  descarta o resto. `dir` e `tags` são filtros exatos de verdade.
- **Embeddings locais via fastembed** — sem chave de API e sem dado saindo da
  máquina. O caminho do arquivo entra no texto embeddings (`File: <path>\n\n<conteúdo>`)
  para que conteúdos iguais em pastas diferentes fiquem distinguíveis.
