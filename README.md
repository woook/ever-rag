# Personal Knowledge Base RAG

Local RAG tool that indexes and searches across Evernote (Yarle) and Obsidian note collections. Supports markdown, PDFs, and images.

## Setup

```bash
pip install sentence-transformers chromadb pymupdf ollama flask litellm
```

### Local models (Ollama)

Requires [Ollama](https://ollama.com) running locally with:
- `qwen3:8b` — answer generation (search and web UI)

### Cloud vision models (image indexing)

```bash
export GEMINI_API_KEY=...           # Gemini 2.5 Flash — primary OCR/description model
export AWS_ACCESS_KEY_ID=...        # Claude Sonnet via Bedrock — secondary interpretation pass
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=eu-west-2
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Paths, model names, chunking parameters |
| `index.py` | Indexing pipeline (markdown, PDFs, images) |
| `search.py` | CLI search interface |
| `web.py` | Web search interface (localhost:5000) |
| `chroma_db/` | Persisted vector database (auto-created) |

## Indexing

### Daily update

```bash
# Text + images (obsidian only, last 30 days, ≥20KB images)
python3 -u index.py

# Images only
python3 -u index.py --only-images

# Text only, skip images
python3 -u index.py --skip-images
```

### Backfill (all sources, all ages)

```bash
# Process all images across both sources with rate limiting
python3 -u index.py --backfill

# Backfill and replace old glm-ocr chunks as Gemini succeeds
python3 -u index.py --backfill --replace-models glm-ocr

# Backfill images only
python3 -u index.py --backfill --only-images
```

### Testing and maintenance

```bash
# Test end-to-end with one image
python3 -u index.py --only-images --max-images 1

# Index one source only
python3 -u index.py --only-source yarle

# Reset and rebuild from scratch (text only)
python3 -u index.py --reset --skip-images

# Verify what's indexed (obsidian only by default; checks against primary vision model)
python3 -u index.py --verify
python3 -u index.py --verify --only-source yarle
python3 -u index.py --verify --backfill   # check all sources

# Verify and re-index any missing text files
python3 -u index.py --verify --fix --skip-images
```

Indexing is **resumable** — stop anytime with Ctrl+C and re-run to continue. If Gemini hits its daily quota, the secondary (Sonnet) pass still runs on images already processed, and re-running the next day continues from where it stopped.

### Flags

| Flag | Description |
|------|-------------|
| `--skip-images` | Skip image processing |
| `--only-images` | Only process images, skip markdown/PDFs |
| `--only-source yarle\|obsidian` | Index one collection only |
| `--backfill` | Process all images regardless of age (for bulk initial indexing). Retries on 429s using `Retry-After` header with exponential backoff. Default processes last 30 days only |
| `--vision-models MODEL …` | Vision models in order. First runs on all new images; subsequent models run only on images the first succeeded on. Default: `gemini/gemini-2.5-flash` then `bedrock/eu.anthropic.claude-sonnet-4-6` |
| `--replace-models MODEL …` | Delete chunks from these models when the primary model succeeds (e.g. `--replace-models glm-ocr`) |
| `--min-image-kb N` | Minimum image size in KB (default: 20) |
| `--max-images N` | Limit images processed per pass per source |
| `--reset` | Delete existing index before rebuilding |
| `--verify` | Audit indexed files against disk; no indexing performed |
| `--fix` | Use with `--verify` to re-index any missing files |

## MCP Server (Claude Code integration)

`mcp_server.py` exposes the RAG index as a tool that Claude can call directly during conversations, without needing the web UI or CLI.

### Setup

```bash
pip install fastmcp
```

Register with Claude Code (user scope — available in all projects):

```bash
claude mcp add --transport stdio personal-notes -- python3 /home/wook/Documents/ever/mcp_server.py
```

Or project-scoped (saves to `.mcp.json` in the repo):

```bash
claude mcp add --scope project --transport stdio personal-notes -- python3 /home/wook/Documents/ever/mcp_server.py
```

Verify it's registered:

```bash
claude mcp list
```

### Tool: `search_notes`

Once registered, Claude can call `search_notes` during any conversation. Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Natural-language search query |
| `source` | string (optional) | Limit to `"obsidian"` or `"yarle"` |
| `content_type` | string (optional) | Limit to `"md"`, `"pdf"`, or `"image"` |
| `top_k` | int (optional) | Number of chunks to return (default 5) |
| `date_after` | string (optional) | ISO date filter, e.g. `"2025-03-01"` |

Returns raw chunks with collection, file type, date, similarity score, and filename so Claude can synthesise an answer and cite specific notes.

### Notes

- The server loads the embedding model and ChromaDB index once at startup — first call may be slow.
- Requires the index to be built first (`index.py`). If the collection doesn't exist, the server exits with an error.

## Search

### CLI

```bash
# One-shot query with LLM answer
python3 search.py "What is the FHIR standard?"

# Interactive mode
python3 search.py

# Filter by source or content type
python3 search.py --source obsidian "WGS validation"
python3 search.py --type pdf "sequencing quality"

# Raw chunks without LLM synthesis
python3 search.py --no-llm "pipeline architecture"

# More context chunks
python3 search.py --top-k 10 "variant calling"
```

### Web UI

```bash
python3 web.py
# Open http://localhost:5000
```

Features: search box, source/type/top-k filters, "chunks only" mode, similarity scores, source file paths.

## Architecture

```
[Markdown/PDF/Image files] → [Indexer] → [ChromaDB vectors]
                                              ↓
                            [User query] → [Retriever] → [Top chunks] → [Qwen3 8B] → [Answer]
```

- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, ~80MB)
- **Vector store**: ChromaDB with cosine similarity
- **Chunking**: 500 chars with 50 char overlap
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp`; processed newest-first by modification time; skips files < 20KB; default age filter ≤ 30 days (overridden by `--backfill`)
- **Two-pass vision**: Pass 1 (Gemini 2.5 Flash) runs on all new images; Pass 2 (Claude Sonnet) runs only on images Pass 1 succeeded on. Each model produces separate chunks that coexist in the index.
- **PDF OCR fallback**: if a PDF has no text layer, each page is rendered at 150 DPI and passed to the primary vision model; disabled by `--skip-images`
- **Chunk IDs**: include the vision model name so multiple models can index the same image independently, and so re-runs skip already-indexed files
- **Date metadata**: every chunk stores a `note_date` field (`YYYY-MM-DD`) extracted by priority:
  1. Obsidian — date prefix in filename (e.g. `2026-02-27 Epic.md`)
  2. Yarle — `Created at:` line in the first 15 lines of the file
  3. Fallback — file modification time
  This field enables date-range filtering in both the CLI (`search.py`) and the MCP `search_notes` tool (`date_after` parameter).

## Sources

| Collection | Path | Content |
|------------|------|---------|
| yarle | `/home/wook/Documents/evern/yarle1` | ~1,715 Markdown files, ~263 PDFs, ~2,203 images |
| obsidian | `/home/wook/Documents/obsidiangit` | ~600 Markdown files, ~20 PDFs, ~1,373 images |

## Further Details

See [technical_walkthrough.md](technical_walkthrough.md) for a deeper dive into the implementation, including:

- Live-executed examples of the indexing pipeline, chunk ID design, and two-pass cloud vision
- ChromaDB metadata schema with real chunk samples
- MCP server internals and `search_notes` parameter reference
- Note date extraction logic and migration script walkthrough
- Dependency versions (verified at document build time via showboat)
