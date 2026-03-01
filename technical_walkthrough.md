# Personal Knowledge Base RAG — Technical Walkthrough

*2026-03-01T23:04:15Z by Showboat 0.6.1*
<!-- showboat-id: 192c3921-1b26-419b-a2b7-b3afc8ed3732 -->

Local RAG tool that indexes two personal note collections (Evernote/Yarle and Obsidian) into a ChromaDB vector database and answers natural-language queries using a locally-running Ollama LLM. Images go through a two-pass cloud vision pipeline (Gemini 2.5 Flash then Claude Sonnet via Bedrock). An MCP server exposes the index as a tool Claude can call directly from any conversation.

## Project Structure

```bash
ls -1 /home/wook/Documents/ever/*.py /home/wook/Documents/ever/CLAUDE.md
```

```output
/home/wook/Documents/ever/CLAUDE.md
/home/wook/Documents/ever/config.py
/home/wook/Documents/ever/index.py
/home/wook/Documents/ever/mcp_server.py
/home/wook/Documents/ever/migrate_add_dates.py
/home/wook/Documents/ever/note_date_utils.py
/home/wook/Documents/ever/search.py
/home/wook/Documents/ever/test_vision.py
/home/wook/Documents/ever/web.py
```

Responsibilities by file:

| File | Role |
|---|---|
| `config.py` | All constants — paths, model names, chunking params |
| `index.py` | Indexing pipeline: scan → parse → chunk → embed → store |
| `note_date_utils.py` | Lightweight helper: extract `note_date` from filename/frontmatter/mtime |
| `migrate_add_dates.py` | One-off migration to backfill `note_date` onto existing chunks |
| `mcp_server.py` | FastMCP server exposing `search_notes` as a Claude tool |
| `search.py` | CLI search interface |
| `web.py` | Flask browser UI |
| `test_vision.py` | Utility to compare cloud vision models on sample images |

Both `search.py` and `web.py` share the same retrieval logic: encode query → ChromaDB cosine search → format context → Ollama LLM synthesis.

## Configuration (config.py)

All tunable parameters live in a single file. Key design choices:

- `DEFAULT_SOURCES = ["obsidian"]` — daily mode processes only the active notes source; `--backfill` expands to all sources
- `MIN_IMAGE_SIZE_BYTES = 20KB` — filters out icons and thumbnails
- `MAX_IMAGE_AGE_DAYS = 30` — daily mode only touches recent images; `--backfill` disables this
- Fixed Gemini delays were removed in favour of adaptive retry on 429s (see indexing section)

<details>
<summary>config.py</summary>

```python
"""Configuration for the RAG knowledge base tool."""

import os

# Source directories
SOURCES = {
    "obsidian": "/home/wook/Documents/obsidiangit",
    "yarle": "/home/wook/Documents/evern/yarle1",
}

# Sources processed in daily mode; --backfill uses all SOURCES
DEFAULT_SOURCES = ["obsidian"]

# ChromaDB
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "knowledge_base"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ollama models
OLLAMA_LLM = "qwen3:8b"

# Image processing
MIN_IMAGE_SIZE_BYTES = 20 * 1024        # skip icons/thumbnails (was 5KB)
MAX_IMAGE_AGE_DAYS = 30                 # default: only process last 30 days
DEFAULT_VISION_MODELS = [
    "gemini/gemini-2.5-flash",
    "bedrock/eu.anthropic.claude-sonnet-4-6",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
```

</details>

## Indexing Pipeline (index.py)

```bash
/usr/bin/python3 index.py --help
```

```output
usage: index.py [-h] [--skip-images] [--only-images]
                [--only-source {obsidian,yarle}]
                [--vision-models MODEL [MODEL ...]] [--backfill]
                [--min-image-kb MIN_IMAGE_KB] [--reset]
                [--max-images MAX_IMAGES] [--replace-models MODEL [MODEL ...]]
                [--verify] [--fix]

Index knowledge base into ChromaDB

options:
  -h, --help            show this help message and exit
  --skip-images         Skip image processing (much faster)
  --only-images         Only process images (resume image indexing)
  --only-source {obsidian,yarle}
                        Index only one source
  --vision-models MODEL [MODEL ...]
                        Vision model(s) in order. First runs on all new
                        images; subsequent models only run on images the first
                        model succeeded on. Default:
                        ['gemini/gemini-2.5-flash',
                        'bedrock/eu.anthropic.claude-sonnet-4-6']
  --backfill            Process all images regardless of age (for bulk initial
                        indexing). Resumable.
  --min-image-kb MIN_IMAGE_KB
                        Minimum image size in KB. Default: 20 (from config).
  --reset               Delete existing index before building
  --max-images MAX_IMAGES
                        Limit number of images processed (useful for testing)
  --replace-models MODEL [MODEL ...]
                        Delete chunks from these models when the primary model
                        succeeds on the same image (e.g. --replace-models glm-
                        ocr to remove old local-model chunks during backfill).
  --verify              Check which files on disk are indexed; no indexing is
                        performed
  --fix                 Use with --verify to index any missing files found
```

The indexer walks source directories, parses files into text chunks, embeds them with `all-MiniLM-L6-v2`, and upserts into ChromaDB. Indexing is resumable: chunk IDs are derived from the MD5 of the file path so already-indexed files are skipped on re-runs.

### Chunk ID design

Each chunk ID has the form `{md5_12chars}_{chunk_index}`. For images the MD5 includes a vision-model suffix, e.g. `abc123def456:gemini/gemini-2.5-flash`. This means:

- Re-runs skip already-processed files without duplicating data
- Multiple vision models produce independent, coexisting chunks for the same image
- Failure sentinels (`source_type: image_failed`) also use model-scoped IDs, preventing retry storms

### File type handling

| Type | Processing |
|------|-----------|
| `.md` | Read → strip Excalidraw blocks, HTML tags, base64 blobs → chunk |
| `.pdf` | Extract text layer → chunk; if no text layer, render pages at 150 DPI and OCR via primary vision model |
| Images | Two-pass cloud vision (see below); skipped if < 20 KB or older than 30 days in daily mode |

### Two-pass cloud vision

Images go through two sequential model passes. Pass 2 only runs on files Pass 1 succeeded on, so Bedrock costs are not incurred for images Gemini could not process. Rate limiting uses adaptive retry: on a 429 the `Retry-After` header is respected; without one, exponential backoff (5s, 10s, 20s…) kicks in.

```text
Pass 1 — Gemini 2.5 Flash (primary)
  ├─ All new images matching size/age filters
  ├─ Rate-limited: adaptive retry on 429 (Retry-After header or exponential backoff)
  ├─ On quota exhaustion: flush buffer, break — secondary pass still runs on already-processed images
  └─ On other failure: store image_failed sentinel (skipped on next run)

Pass 2 — Claude Sonnet via Bedrock (secondary)
  ├─ Only images where Pass 1 produced at least one successful chunk
  ├─ No rate limiting (Bedrock pay-per-use)
  └─ Produces separate, coexisting chunks under its own model-scoped ID
```

The `_call_vision_model()` helper routes by model name: presence of `/` indicates a cloud model dispatched via LiteLLM; otherwise the call goes to local Ollama.

### Replacing old model chunks

`--replace-models MODEL …` deletes chunks from the specified model(s) whenever the primary model succeeds on the same image. Used during backfill to swap out lower-quality local-model chunks:

```bash
python3 -u index.py --backfill --replace-models glm-ocr
```

### Verify index status

`--verify` scans all files on disk and reports how many are indexed against the primary vision model. It uses no age filter so the full picture is visible regardless of daily-mode cutoffs.

```bash
/usr/bin/python3 index.py --verify 2>&1 | grep -v 'Loading weights\|Materializing\|LOAD REPORT\|UNEXPECTED\|Notes:\|can be ignored\|Key \|----\|BertModel\|Warning:'
```

```output
Existing documents in collection: 201594
Checking already-indexed chunks...
Already indexed: 201594 chunks

============================================================
Verifying: obsidian (/home/wook/Documents/obsidiangit)
============================================================
  md    : 589/589 indexed  [OK]
  pdf   : 20/20 indexed  [OK]
  image : 1369/1373 indexed  [FAILED 4]

```

The index has grown from 176k to 201k chunks since the previous walkthrough — the backfill and cloud vision pipeline have processed most of the Obsidian image library. The 4 FAILED entries are images the primary vision model could not process; they are stored as `image_failed` sentinels so they are not retried on every run.

## Note Date Metadata

Every chunk now carries a `note_date` field (`YYYY-MM-DD`) extracted by `note_date_utils.py`. This enables date-range filtering in both the CLI and the MCP server.

### Extraction priority

1. **Obsidian** — date prefix in filename (e.g. `2026-02-27 Meeting notes.md`)
2. **Yarle** — `Created at:` line in the first 15 lines of the file
3. **Fallback** — file modification time

Each extracted date string is validated with `datetime.strptime` before use — regex-shaped but calendar-invalid strings (e.g. month 13) fall through to the mtime fallback.

<details>
<summary>note_date_utils.py</summary>

```python
"""Lightweight date-extraction helper shared by index.py and migrate_add_dates.py.

Kept as a separate module so migrate_add_dates.py does not pull in the heavy
indexing dependencies (chromadb, sentence-transformers, fitz, etc.) that
index.py imports at the top level.
"""

import os
import re
from datetime import datetime


def extract_note_date(path: str, collection_name: str, text: str = "") -> str:
    """Extract note date as 'YYYY-MM-DD'.

    Priority:
    1. Obsidian: date prefix in filename (e.g. "2026-02-27 Epic.md")
    2. Yarle: 'Created at:' line in the first 15 lines of text
    3. Fallback: file modification time
    """
    if collection_name == "obsidian":
        m = re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(path))
        if m:
            try:
                datetime.strptime(m.group(0), "%Y-%m-%d")
                return m.group(0)
            except ValueError:
                pass
    elif collection_name == "yarle" and text:
        for line in text.splitlines()[:15]:
            if line.startswith("Created at:"):
                m = re.search(r'\d{4}-\d{2}-\d{2}', line)
                if m:
                    try:
                        datetime.strptime(m.group(0), "%Y-%m-%d")
                        return m.group(0)
                    except ValueError:
                        pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except Exception:
        return "1970-01-01"
```

</details>

### Backfilling existing chunks

`migrate_add_dates.py` patches `note_date` onto chunks that predate the field — metadata-only writes, no re-embedding. Safe to re-run: chunks that already have `note_date` are skipped.

<details>
<summary>migrate_add_dates.py</summary>

```python
#!/usr/bin/env python3
"""One-off migration: patch note_date metadata onto all existing ChromaDB chunks.

Reads source_file from each chunk's existing metadata and derives the date using
the same logic as extract_note_date() in index.py. No model calls — just metadata
writes. Safe to re-run: chunks that already have note_date are skipped.
"""

import os

import chromadb

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from note_date_utils import extract_note_date


def main():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"Total chunks in collection: {total}")

    FETCH_BATCH = 5000
    to_update_ids = []
    to_update_metadatas = []
    already_have_date = 0

    for offset in range(0, total, FETCH_BATCH):
        results = collection.get(
            offset=offset,
            limit=FETCH_BATCH,
            include=["metadatas"],
        )
        for chunk_id, meta in zip(results["ids"], results["metadatas"], strict=True):
            if not isinstance(meta, dict):
                print(f"  WARN: skipping chunk {chunk_id} — unexpected metadata type: {type(meta)}")
                continue
            if "note_date" in meta:
                already_have_date += 1
                continue

            path = meta.get("source_file", "")
            coll = meta.get("collection", "")

            # For yarle md files read the first 15 lines to find "Created at:"
            text = ""
            if coll == "yarle" and path.endswith(".md") and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        lines = []
                        for _ in range(15):
                            line = f.readline()
                            if not line:
                                break
                            lines.append(line)
                        text = "".join(lines)
                except Exception as e:
                    print(f"  WARN: could not read {path} for date extraction: {e}")

            note_date = extract_note_date(path, coll, text)
            new_meta = dict(meta)
            new_meta["note_date"] = note_date
            to_update_ids.append(chunk_id)
            to_update_metadatas.append(new_meta)

        fetched = min(offset + FETCH_BATCH, total)
        print(f"  Scanned {fetched}/{total} chunks...")

    print(f"\nAlready have note_date: {already_have_date}")
    print(f"Chunks to update: {len(to_update_ids)}")

    if not to_update_ids:
        print("Nothing to do.")
        return

    UPDATE_BATCH = 5000
    for i in range(0, len(to_update_ids), UPDATE_BATCH):
        collection.update(
            ids=to_update_ids[i:i + UPDATE_BATCH],
            metadatas=to_update_metadatas[i:i + UPDATE_BATCH],
        )
        updated = min(i + UPDATE_BATCH, len(to_update_ids))
        print(f"  Updated {updated}/{len(to_update_ids)} chunks")

    print("Done!")


if __name__ == "__main__":
    main()
```

</details>

## MCP Server (mcp_server.py)

`mcp_server.py` exposes the ChromaDB index as a `search_notes` tool that Claude can call directly during any conversation — no browser or CLI needed. It is registered with Claude Code via:

```bash
claude mcp add --transport stdio personal-notes -- python3 /home/wook/Documents/ever/mcp_server.py
```

The server loads the embedding model and ChromaDB collection once at startup and reuses them across all tool calls. The `search_notes` tool supports five parameters:

```bash
/usr/bin/python3 -c "
import ast, sys
src = open('mcp_server.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'search_notes':
        print(ast.get_source_segment(src, node)[:1200])
        break
"
```

```output
def search_notes(
    query: str,
    source: str | None = None,
    content_type: str | None = None,
    top_k: int = 5,
    date_after: str | None = None,
) -> str:
    """Search personal notes (Obsidian + Yarle/Evernote) using semantic similarity.

    Returns raw note chunks with source metadata so you can synthesise an answer
    and cite specific notes.

    Args:
        query: Natural-language search query.
        source: Limit to one collection — "obsidian" or "yarle". Omit for both.
        content_type: Limit to file type — "md", "pdf", or "image". Omit for all.
        top_k: Number of chunks to return (default 5, capped at 100).
        date_after: Only return notes on or after this ISO date, e.g. "2025-03-01".
            Note: implemented as a post-filter over top_k*20 candidates — if most
            chunks predate the cutoff you may receive fewer than top_k results.
    """
    if source and source not in {"obsidian", "yarle"}:
        return f"Invalid source '{source}'. Must be 'obsidian' or 'yarle'."
    if content_type and content_type not in {"md", "pdf", "image"}:
        return f"Invalid content_type '{content_type}'. Must be 'md', 'pdf', or 'image'."
    if
```

| Parameter | Type | Description |
|---|---|---|
| `query` | str | Natural-language search query |
| `source` | str \| None | Limit to `"obsidian"` or `"yarle"` |
| `content_type` | str \| None | Limit to `"md"`, `"pdf"`, or `"image"` |
| `top_k` | int | Chunks to return (default 5, capped at 100) |
| `date_after` | str \| None | ISO date filter e.g. `"2025-03-01"`; validated with `date.fromisoformat()` |

Input validation returns descriptive error strings (rather than raising) so Claude receives actionable feedback. `date_after` is implemented as a Python post-filter over `top_k * 20` candidates — ChromaDB's `$gte` operator does not support strings, but ISO dates sort correctly lexicographically.

<details>
<summary>mcp_server.py — full source</summary>

```python
#!/usr/bin/env python3
"""MCP server exposing the personal notes RAG as a tool for Claude.

Loaded once at startup; Claude calls search_notes() as a tool during conversations.
Returns raw chunks with metadata so Claude can synthesise the answer.
"""

import os
import sys
from datetime import date

import chromadb
from sentence_transformers import SentenceTransformer

# fastmcp must be installed: pip install fastmcp
try:
    from fastmcp import FastMCP
except ImportError:
    print("fastmcp not installed. Run: pip install fastmcp", file=sys.stderr)
    sys.exit(1)

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

mcp = FastMCP("personal-notes")

# Load once at module level — these are reused across all tool calls
print("Loading embedding model...", file=sys.stderr, flush=True)
_model = SentenceTransformer(EMBEDDING_MODEL)

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
try:
    _collection = _client.get_collection(COLLECTION_NAME)
    print(f"Index loaded: {_collection.count()} chunks", file=sys.stderr, flush=True)
except Exception as e:
    print(f"ERROR: Could not load ChromaDB collection: {e}", file=sys.stderr)
    print("Run index.py first to build the index.", file=sys.stderr)
    sys.exit(1)


@mcp.tool()
def search_notes(
    query: str,
    source: str | None = None,
    content_type: str | None = None,
    top_k: int = 5,
    date_after: str | None = None,
) -> str:
    """Search personal notes (Obsidian + Yarle/Evernote) using semantic similarity.

    Returns raw note chunks with source metadata so you can synthesise an answer
    and cite specific notes.

    Args:
        query: Natural-language search query.
        source: Limit to one collection — "obsidian" or "yarle". Omit for both.
        content_type: Limit to file type — "md", "pdf", or "image". Omit for all.
        top_k: Number of chunks to return (default 5, capped at 100).
        date_after: Only return notes on or after this ISO date, e.g. "2025-03-01".
            Note: implemented as a post-filter over top_k*20 candidates — if most
            chunks predate the cutoff you may receive fewer than top_k results.
    """
    if source and source not in {"obsidian", "yarle"}:
        return f"Invalid source '{source}'. Must be 'obsidian' or 'yarle'."
    if content_type and content_type not in {"md", "pdf", "image"}:
        return f"Invalid content_type '{content_type}'. Must be 'md', 'pdf', or 'image'."
    if top_k < 1:
        return "top_k must be at least 1."
    top_k = min(top_k, 100)
    if date_after:
        try:
            date.fromisoformat(date_after)
        except ValueError:
            return f"Invalid date_after '{date_after}'. Must be a valid ISO date YYYY-MM-DD."

    conditions = [{"source_type": {"$ne": "image_failed"}}]
    if source:
        conditions.append({"collection": source})
    if content_type:
        conditions.append({"source_type": content_type})
    # note_date is stored as "YYYY-MM-DD" string; ChromaDB $gte only supports numbers,
    # so we post-filter in Python (ISO date strings compare correctly lexicographically).
    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    embedding = _model.encode([query]).tolist()
    fetch_k = top_k * 20 if date_after else top_k
    results = _collection.query(
        query_embeddings=embedding,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if date_after:
        filtered = [
            (d, m, dist)
            for d, m, dist in zip(docs, metas, dists, strict=True)
            if m.get("note_date", "") >= date_after
        ]
        docs, metas, dists = (
            [x[0] for x in filtered[:top_k]],
            [x[1] for x in filtered[:top_k]],
            [x[2] for x in filtered[:top_k]],
        )

    if not docs:
        return "No matching notes found."

    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists, strict=True), 1):
        collection = meta.get("collection", "unknown")
        stype = meta.get("source_type", "unknown")
        note_date = meta.get("note_date", "unknown")
        source_file = meta.get("source_file", "unknown")
        similarity = 1 - dist
        filename = os.path.basename(source_file)

        parts.append(
            f"[{i}] {collection} | {stype} | {note_date} | similarity: {similarity:.3f}\n"
            f"File: {filename}\n\n"
            f"{doc}"
        )

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()
```

</details>

## Search Pipeline (search.py)

```bash
/usr/bin/python3 search.py --help
```

```output
usage: search.py [-h] [--source {yarle,obsidian}] [--type {md,pdf,image}]
                 [--top-k TOP_K] [--no-llm]
                 [query ...]

Search your knowledge base

positional arguments:
  query                 One-shot query (omit for interactive mode)

options:
  -h, --help            show this help message and exit
  --source {yarle,obsidian}
                        Filter by source collection
  --type {md,pdf,image}
                        Filter by content type
  --top-k TOP_K         Number of chunks to retrieve (default: 5)
  --no-llm              Show raw chunks without LLM synthesis
```

The retrieval pipeline is shared between CLI and web:

1. Encode query with `all-MiniLM-L6-v2` (384-dim cosine embedding)
2. Query ChromaDB with optional `where` filters on `collection` and/or `source_type`
3. Format top-k chunks as context with source file citations
4. Send context + question to `qwen3:8b` via Ollama with a grounding system prompt
5. Stream the answer back to the user

`--no-llm` skips steps 4–5 and returns raw chunks — useful for debugging retrieval without Ollama running.

### Demo — raw chunk retrieval

```bash
/usr/bin/python3 search.py --no-llm --top-k 3 'FHIR standard' 2>&1 | grep -v 'Loading weights\|Materializing\|LOAD REPORT\|UNEXPECTED\|Notes:\|can be ignored\|Key \|----\|BertModel\|Warning:'
```

```output
Loading embedding model...

Index loaded: 201594 chunks

============================================================
[Source: /home/wook/Documents/evern/yarle1/_resources/FHIR_Specification_Home_Page_-_FHIR_v0.0.81.resources/unknown_filename.1.png (image, similarity=0.783)]
FHIR®

---

[Source: /home/wook/Documents/evern/yarle1/FHIR Specification Home Page - FHIR v0.0.81.md (md, similarity=0.666)]
atest.fhir.me/).

# 0 Welcome to FHIR® [![[./_resources/FHIR_Specification_Home_Page_-_FHIR_v0.0.81.resources/unknown_filename.8.png|16x16]]](http://www.hl7.org/implement/standards/fhir/index.html#root)

First time here? Read the [high level summary](http://www.hl7.org/implement/standards/fhir/summary.html) and then the [FHIR overview / roadmap](http://www.hl7.org/implement/standards/fhir/overview.html). See also the [open license](http://www.hl7.org/implement/standards/fhir/license.html).

**DS

---

[Source: /home/wook/Documents/evern/yarle1/FHIR Specification Home Page - FHIR v0.0.81.md (md, similarity=0.666)]
atest.fhir.me/).

# 0 Welcome to FHIR® [![[./_resources/FHIR_Specification_Home_Page_-_FHIR_v0.0.81.resources/unknown_filename.8.png|16x16]]](http://www.hl7.org/implement/standards/fhir/index.html#root)

First time here? Read the [high level summary](http://www.hl7.org/implement/standards/fhir/summary.html) and then the [FHIR overview / roadmap](http://www.hl7.org/implement/standards/fhir/overview.html). See also the [open license](http://www.hl7.org/implement/standards/fhir/license.html).

**DS
```

The top result is an image chunk (similarity 0.783) — its text description was embedded by a vision model and competes on equal footing with markdown chunks. The image was indexed by the legacy `glm-ocr` local model which predates the `note_date` field, so it lacks that metadata.

## ChromaDB Index

```bash
/usr/bin/python3 -c "
import chromadb, sys
sys.path.insert(0, \"/home/wook/Documents/ever\")
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(COLLECTION_NAME)
print(f\"Total chunks: {col.count()}\")

# Markdown chunk (note_date populated by migration)
md = col.get(where={\"source_type\": \"md\"}, limit=1, include=[\"metadatas\"])
print(\"Markdown chunk:\", md[\"metadatas\"][0])

# Gemini image chunk (note_date set at index time)
gem = col.get(where={\"vision_model\": \"gemini/gemini-2.5-flash\"}, limit=1, include=[\"metadatas\"])
print(\"Gemini image chunk:\", gem[\"metadatas\"][0])

# Sonnet chunk
son = col.get(where={\"vision_model\": \"bedrock/eu.anthropic.claude-sonnet-4-6\"}, limit=1, include=[\"metadatas\"])
print(\"Sonnet chunk:\", son[\"metadatas\"][0])
"
```

```output
Total chunks: 201594
Markdown chunk: {'source_type': 'md', 'source_file': '/home/wook/Documents/evern/yarle1/Mono- and biallelic variant effects on disease at biobank scale - Nature.md', 'note_date': '2023-01-19', 'collection': 'yarle', 'chunk_index': 0}
Gemini image chunk: {'note_date': '2026-02-27', 'source_file': '/home/wook/Documents/obsidiangit/2026/attachments/Pasted image 20260227145020.png', 'source_type': 'image_failed', 'vision_model': 'gemini/gemini-2.5-flash', 'collection': 'obsidian', 'chunk_index': 0}
Sonnet chunk: {'collection': 'obsidian', 'note_date': '2026-02-27', 'vision_model': 'bedrock/eu.anthropic.claude-sonnet-4-6', 'chunk_index': 0, 'source_file': '/home/wook/Documents/obsidiangit/2026/attachments/Pasted image 20260227144303.png', 'source_type': 'image'}
```

All chunk types now carry `note_date`. The complete metadata schema:

| Field | Values | Purpose |
|---|---|---|
| `source_type` | `md`, `pdf`, `image`, `image_failed` | Type filter in search; sentinel detection |
| `collection` | `yarle`, `obsidian` | Source filter in search |
| `source_file` | absolute path | Citation in answers |
| `chunk_index` | 0, 1, 2… | Ordering; first-chunk resume check |
| `note_date` | `YYYY-MM-DD` | Date-range filtering in search and MCP |
| `vision_model` | model name (image chunks only) | Multi-model coexistence; `--replace-models` targeting |

Legacy chunks indexed by the old `glm-ocr` local model have `note_date` backfilled by `migrate_add_dates.py` but lack `vision_model` (pre-cloud pipeline).

## Web UI (web.py)

A minimal Flask app exposing the same retrieval pipeline through a browser interface. Models are loaded once at startup and reused across requests. The UI provides filter dropdowns for source collection, content type, and top-k, plus a "Chunks only" toggle that bypasses the LLM.

> **Flask caveat**: `render_template_string()` has a built-in parameter named `source`. Passing a template variable with that name silently breaks rendering. The workaround is to rename it `src_filter` before passing to the template.

## Utilities

### test_vision.py

Quick script to run a sample image through multiple cloud vision models side-by-side before committing to a full index run. Useful for comparing output quality across Gemini, Claude Haiku/Sonnet/Opus via Bedrock.

<details>
<summary>test_vision.py</summary>

```python
#!/usr/bin/env python3
"""Quick script to compare cloud vision models on sample images before committing to a full index run.

Usage:
    python3 test_vision.py /path/to/image.jpg [/path/to/image2.png ...]

Set API keys as environment variables before running:
    export GEMINI_API_KEY=...        # Gemini
    export OPENAI_API_KEY=...        # GPT-4o-mini
    export OPENROUTER_API_KEY=...    # Qwen2-VL via OpenRouter
    export GROQ_API_KEY=...          # Llama Vision via Groq
    export ANTHROPIC_API_KEY=...     # Claude Haiku
"""

import base64
import os
import sys

import litellm

MODELS = [
    ("gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
    # ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
    # ("openrouter/qwen/qwen2-vl-7b-instruct", "OPENROUTER_API_KEY"),
    # ("groq/llama-3.2-11b-vision-preview", "GROQ_API_KEY"),
    ("bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0", "AWS credentials"),
    ("bedrock/eu.anthropic.claude-sonnet-4-6", "AWS credentials"),
    ("bedrock/eu.anthropic.claude-opus-4-6-v1", "AWS credentials"),
]

PROMPT = "Describe the content of this image in detail, including any text visible."


def run_image_check(img_path: str) -> None:
    if not os.path.isfile(img_path):
        print(f"ERROR: File not found: {img_path}")
        return

    print(f"\n{'#'*70}")
    print(f"Image: {img_path}")
    print(f"{'#'*70}")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Detect MIME type from extension
    ext = img_path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp"}.get(ext, "image/png")

    for model, key_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}  (needs {key_name})")
        print(f"{'='*60}")
        try:
            r = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ]}],
            )
            print(r.choices[0].message.content)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_vision.py <image> [image2 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        run_image_check(path)
```

</details>

## Dependencies

```bash
/usr/bin/python3 -c "
import sentence_transformers, chromadb, importlib.metadata
print(\"sentence-transformers:\", sentence_transformers.__version__)
print(\"chromadb:\", chromadb.__version__)
print(\"flask:\", importlib.metadata.version(\"flask\"))
print(\"litellm:\", importlib.metadata.version(\"litellm\"))
print(\"pymupdf:\", importlib.metadata.version(\"pymupdf\"))
print(\"ollama:\", importlib.metadata.version(\"ollama\"))
print(\"fastmcp:\", importlib.metadata.version(\"fastmcp\"))
"
```

```output
sentence-transformers: 5.2.2
chromadb: 1.5.0
flask: 3.1.2
litellm: 1.81.16
pymupdf: 1.27.1
ollama: 0.6.1
fastmcp: 3.0.2
```

All dependencies install via pip:

```bash
pip install sentence-transformers chromadb pymupdf ollama flask litellm fastmcp
```

`litellm` is imported lazily inside `_call_vision_model()` and only loaded when a cloud model (identified by `/` in the model name) is actually called, so Ollama-only usage incurs no import cost. `fastmcp` is only needed for the MCP server.
