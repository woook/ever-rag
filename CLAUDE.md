# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Local RAG (Retrieval-Augmented Generation) tool that indexes and searches across two personal note collections (Evernote/Yarle and Obsidian). Supports markdown, PDFs, and images.

## Commands

```bash
# Install dependencies
pip install sentence-transformers chromadb pymupdf ollama flask

# Index text only (~3 min)
python3 -u index.py --skip-images

# Index images for one source
python3 -u index.py --only-images --only-source obsidian --vision-model glm-ocr

# CLI search
python3 search.py "your query"

# Web UI on localhost:5000
python3 web.py
```

Always use `python3 -u` for indexing to get unbuffered output (progress visibility).

## Architecture

```
config.py          → Central configuration (paths, models, chunking params)
index.py           → Indexing pipeline: scan → parse → chunk → embed → store in ChromaDB
search.py          → CLI search: query → embed → retrieve top-k → send to Ollama → answer
web.py             → Flask web frontend wrapping the same search logic
chroma_db/         → Persisted ChromaDB vector database (gitignored)
```

All four files import constants from `config.py`. Both `search.py` and `web.py` use the same retrieval pattern: encode query with sentence-transformers, query ChromaDB with cosine similarity, format context, send to Ollama LLM.

## Key Design Decisions

**Chunk IDs**: Based on MD5 of file path + optional vision model suffix. Format: `{hash12}_{chunk_index}`. This enables:
- Resumable indexing (skip files whose first chunk ID already exists)
- Multiple vision models on the same images (different suffix = different IDs, chunks coexist)

**Metadata per chunk**: `source_file`, `source_type` (md/pdf/image), `collection` (yarle/obsidian), `chunk_index`, and optionally `vision_model`. Filtering in search uses ChromaDB `where` clauses on these fields.

**Image processing**: Images sorted newest-first by creation time. Flushed to ChromaDB every 10 images (vs 64 for text files) since vision model calls are slow. Files < 5KB are skipped.

**Error handling**: File processing functions return `[]` on error and print a warning. Indexing never crashes on a single bad file.

## Ollama Models Required

- `qwen3:8b` — answer generation (used by search.py and web.py)
- `glm-ocr` — fast image OCR (0.9B params, fits in 8GB VRAM)
- `qwen3-vl:4b` — richer image descriptions (optional second pass)

## Flask Caveat

`render_template_string()` has a built-in `source` parameter. Never pass a variable named `source` to it — use `src_filter` or similar instead.
