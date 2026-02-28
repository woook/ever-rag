# Personal Knowledge Base RAG

Local RAG tool that indexes and searches across Evernote (Yarle) and Obsidian note collections. Supports markdown, PDFs, and images.

## Setup

```bash
pip install sentence-transformers chromadb pymupdf ollama flask
```

Requires [Ollama](https://ollama.com) running locally with:
- `qwen3:8b` — answer generation
- `glm-ocr` — image OCR (0.9B, fast)
- `qwen3-vl:4b` — image descriptions (optional, richer but slower)

## Files

| File | Purpose |
|------|---------|
| `config.py` | Paths, model names, chunking parameters |
| `index.py` | Indexing pipeline (markdown, PDFs, images) |
| `search.py` | CLI search interface |
| `web.py` | Web search interface (localhost:5000) |
| `chroma_db/` | Persisted vector database (auto-created) |

## Indexing

```bash
# Index all text (markdown + PDFs), skip images (~3 min)
python3 -u index.py --skip-images

# Index images from one source with a specific vision model
python3 -u index.py --only-images --only-source obsidian --vision-model glm-ocr
python3 -u index.py --only-images --only-source yarle --vision-model glm-ocr

# Add a second pass with a different model (coexists, doesn't overwrite)
python3 -u index.py --only-images --only-source obsidian --vision-model qwen3-vl:4b

# Reset and rebuild from scratch
python3 -u index.py --reset --skip-images

# Test image capture with a small sample
python3 -u index.py --only-images --max-images 5 --vision-model glm-ocr

# Test PDF text extraction and OCR fallback with a small sample
python3 -u index.py --only-source yarle --max-images 3 --vision-model glm-ocr

# Verify what's been indexed without re-indexing
python3 -u index.py --verify
python3 -u index.py --verify --only-source yarle
python3 -u index.py --verify --vision-model glm-ocr

# Verify and re-index any missing files in one step
python3 -u index.py --verify --fix --skip-images
python3 -u index.py --verify --fix --only-images --vision-model glm-ocr
```

Indexing is **resumable** — stop anytime with Ctrl+C and re-run to continue where you left off. Each vision model produces separate chunks so multiple models can be used on the same images.

### Flags

| Flag | Description |
|------|-------------|
| `--skip-images` | Skip image processing |
| `--only-images` | Only process images, skip markdown/PDFs |
| `--only-source yarle\|obsidian` | Index one collection only |
| `--vision-model MODEL` | Ollama vision model for images (default: `qwen3-vl`) |
| `--max-images N` | Process at most N new images, and at most N scanned PDFs (per source) via OCR fallback |
| `--reset` | Delete existing index before rebuilding |
| `--verify` | Audit indexed files against disk; no indexing performed |
| `--fix` | Use with `--verify` to index any missing files found |

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
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp` only; processed newest-first by modification time; skips files < 5KB
- **PDF OCR fallback**: if a PDF has no text layer, each page is rendered at 150 DPI and passed to the vision model for OCR; disabled by `--skip-images`

## Sources

| Collection | Path | Content |
|------------|------|---------|
| yarle | `/home/wook/Documents/evern/yarle1` | ~1,715 markdown, ~263 PDFs, ~2,942 images |
| obsidian | `/home/wook/Documents/obsidiangit` | ~570 markdown, ~17 PDFs, ~1,360 images |
