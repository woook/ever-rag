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

## Sources

| Collection | Path | Content |
|------------|------|---------|
| yarle | `/home/wook/Documents/evern/yarle1` | ~1,715 Markdown files, ~263 PDFs, ~2,203 images |
| obsidian | `/home/wook/Documents/obsidiangit` | ~600 Markdown files, ~20 PDFs, ~1,373 images |
