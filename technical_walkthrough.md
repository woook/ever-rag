# Personal Knowledge Base RAG Tool — Technical Walkthrough

*2026-02-22T22:33:01Z by Showboat 0.6.0*
<!-- showboat-id: cadbeca8-8a84-48eb-9812-af2f3d8c853a -->

This tool indexes two personal note collections (Evernote/Yarle and Obsidian) into a local ChromaDB vector database and answers natural-language queries using a locally-running Ollama LLM. Everything runs offline — no cloud services are required after the initial model download.

## Project Structure

```bash
ls -1 /home/wook/Documents/ever/*.py /home/wook/Documents/ever/CLAUDE.md
```

```output
/home/wook/Documents/ever/CLAUDE.md
/home/wook/Documents/ever/config.py
/home/wook/Documents/ever/index.py
/home/wook/Documents/ever/search.py
/home/wook/Documents/ever/web.py
```

The four Python modules have clearly separated responsibilities: config.py holds all constants, index.py builds the vector database, search.py provides a CLI interface, and web.py serves a browser-based UI. Both search interfaces share the same retrieval logic.

## Configuration (config.py)

All tunable parameters live in a single file. This makes it easy to swap models, change chunk sizes, or point to different note directories without touching the pipeline code.

```bash
cat /home/wook/Documents/ever/config.py
```

```output
"""Configuration for the RAG knowledge base tool."""

import os

# Source directories
SOURCES = {
    "yarle": "/home/wook/Documents/evern/yarle1",
    "obsidian": "/home/wook/Documents/obsidiangit",
}

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
OLLAMA_VISION = "qwen3-vl"

# Image processing
MIN_IMAGE_SIZE_BYTES = 5 * 1024  # skip images < 5KB
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
```

## Indexing Pipeline (index.py)

The indexer walks both source directories, parses files into text chunks, embeds them with a local sentence-transformer model, and upserts them into ChromaDB. Indexing is resumable: chunk IDs are derived from the MD5 of the file path, so already-indexed files are skipped on re-runs.

```bash
PYTHONPATH=/home/wook/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages python3 /home/wook/Documents/ever/index.py --help
```

```output
usage: index.py [-h] [--skip-images] [--only-images]
                [--only-source {yarle,obsidian}] [--vision-model VISION_MODEL]
                [--reset]

Index knowledge base into ChromaDB

options:
  -h, --help            show this help message and exit
  --skip-images         Skip image processing (much faster)
  --only-images         Only process images (resume image indexing)
  --only-source {yarle,obsidian}
                        Index only one source
  --vision-model VISION_MODEL
                        Ollama vision model for images (default: qwen3-vl)
  --reset               Delete existing index before building
```

### Typical indexing commands

For a fast first pass, skip image processing (images require a vision LLM call per file which is slow). Images can be indexed separately in a second pass.

The `--only-source` flag lets you re-index a single collection without touching the other.

```bash
echo '# Index text only (fast, ~3 min)
python3 -u index.py --skip-images

# Index images for one source using a fast vision model
python3 -u index.py --only-images --only-source obsidian --vision-model glm-ocr

# Rebuild from scratch
python3 -u index.py --reset --skip-images'
```

```output
# Index text only (fast, ~3 min)
python3 -u index.py --skip-images

# Index images for one source using a fast vision model
python3 -u index.py --only-images --only-source obsidian --vision-model glm-ocr

# Rebuild from scratch
python3 -u index.py --reset --skip-images
```

### Chunk ID design

Each chunk is assigned an ID of the form `{md5_12chars}_{chunk_index}`. The MD5 is computed from the file path (plus an optional vision-model suffix for images). On re-runs, the indexer checks whether the first chunk of a file already exists in ChromaDB and skips the file if so. This makes indexing resumable after crashes or interruptions without duplicating data.

Multiple vision models can coexist for the same image file because they produce different IDs (different suffix).

## CLI Search (search.py)

The CLI tool encodes a query with the same sentence-transformer model used at index time, retrieves the top-k most similar chunks from ChromaDB, and feeds them as context to an Ollama LLM to produce a grounded answer.

```bash
PYTHONPATH=/home/wook/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages python3 /home/wook/Documents/ever/search.py --help
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

### Retrieval pipeline

The same pipeline powers both CLI and web:

1. Encode the query with sentence-transformers (all-MiniLM-L6-v2, 384-dim embeddings)
2. Query ChromaDB with cosine similarity, optionally filtered by collection and/or content type
3. Format retrieved chunks as context with source file citations
4. Send context + question to Ollama (qwen3:8b) with a grounding system prompt
5. Return the synthesised answer

Passing --no-llm skips step 4-5 and returns raw chunks, which is useful for debugging retrieval quality.

## Web UI (web.py)

A minimal Flask app that exposes the same retrieval pipeline through a browser interface. Models are loaded once at startup and reused across requests. The UI adds filter dropdowns for source collection, content type, and top-k, as well as a "Chunks only" toggle that bypasses the LLM.

```bash
echo '# Start the web UI
python3 web.py

# Enable Flask debug mode (auto-reload on file changes)
FLASK_DEBUG=1 python3 web.py'
```

```output
# Start the web UI
python3 web.py

# Enable Flask debug mode (auto-reload on file changes)
FLASK_DEBUG=1 python3 web.py
```

### Flask caveat

render_template_string() has a built-in parameter named `source`. Passing a template variable with that name silently breaks rendering. The workaround used here is to rename it `src_filter` before passing to the template.

## ChromaDB Index

The vector database is persisted to the `chroma_db/` directory alongside the scripts. Chunk metadata enables filtered queries:

```bash
PYTHONPATH=/home/wook/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages python3 -c "
import chromadb, sys
sys.path.insert(0, '/home/wook/Documents/ever')
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(COLLECTION_NAME)
print(f'Total chunks: {col.count()}')
sample = col.get(limit=1, include=['metadatas'])
if sample['metadatas']:
    print('Metadata fields:', list(sample['metadatas'][0].keys()))
"
```

```output
Total chunks: 101504
Metadata fields: ['collection', 'source_file', 'source_type', 'chunk_index']
```

## Ollama Models

Three local models are used depending on the task:

| Model | Role | Size |
|-------|------|------|
| qwen3:8b | Answer generation (search + web) | ~5 GB |
| glm-ocr | Fast image OCR | ~0.9B params, fits in 8 GB VRAM |
| qwen3-vl:4b | Richer image descriptions (optional) | ~4B params |

All models are served by a locally-running Ollama instance. No data leaves the machine.

```bash
ollama list
```

```output
NAME                        ID              SIZE      MODIFIED         
qwen3:8b                    500a1f067a9f    5.2 GB    8 days ago            
qwen3-vl:latest             901cae732162    6.1 GB    2 weeks ago      
```

## Dependencies

All dependencies install via pip. There are no unusual system requirements beyond a working Ollama installation.

```bash
echo 'pip install sentence-transformers chromadb pymupdf ollama flask'
```

```output
pip install sentence-transformers chromadb pymupdf ollama flask
```

```bash
PYTHONPATH=/home/wook/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages python3 -c "
import sentence_transformers, chromadb, flask, ollama
print('sentence-transformers:', sentence_transformers.__version__)
print('chromadb:', chromadb.__version__)
print('flask:', flask.__version__)
"
```

```output
<string>:5: DeprecationWarning: The '__version__' attribute is deprecated and will be removed in Flask 3.2. Use feature detection or 'importlib.metadata.version("flask")' instead.
sentence-transformers: 5.2.2
chromadb: 1.5.0
flask: 3.1.2
```
