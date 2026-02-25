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
```

Indexing is **resumable** — stop anytime with Ctrl+C and re-run to continue where you left off. Each vision model produces separate chunks so multiple models can be used on the same images.

### Flags

| Flag | Description |
|------|-------------|
| `--skip-images` | Skip image processing |
| `--only-images` | Only process images, skip markdown/PDFs |
| `--only-source yarle\|obsidian` | Index one collection only |
| `--vision-model MODEL` | Ollama vision model for images (default: `qwen3-vl`) |
| `--max-images N` | Process at most N new images (useful for testing) |
| `--reset` | Delete existing index before rebuilding |

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
- **Images**: processed newest-first, skips files < 5KB

## Sources

| Collection | Path | Content |
|------------|------|---------|
| yarle | `/home/wook/Documents/evern/yarle1` | ~1,715 markdown, ~263 PDFs, ~2,942 images |
| obsidian | `/home/wook/Documents/obsidiangit` | ~570 markdown, ~17 PDFs, ~1,360 images |
