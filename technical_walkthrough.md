# Personal Knowledge Base RAG — Technical Walkthrough

*2026-02-28T13:40:41Z by Showboat 0.6.1*
<!-- showboat-id: d073f667-7502-422e-a1fe-b718ea7ec57d -->

Local RAG tool that indexes two personal note collections (Evernote/Yarle and Obsidian) into a ChromaDB vector database and answers natural-language queries using a locally-running Ollama LLM. Images are processed through a two-pass cloud vision pipeline: Gemini 2.5 Flash for OCR/description, then Claude Sonnet via Bedrock for structured interpretation of complex images.

## Project Structure

```bash
ls -1 /home/wook/Documents/ever/*.py /home/wook/Documents/ever/CLAUDE.md
```

```output
/home/wook/Documents/ever/CLAUDE.md
/home/wook/Documents/ever/config.py
/home/wook/Documents/ever/index.py
/home/wook/Documents/ever/search.py
/home/wook/Documents/ever/test_vision.py
/home/wook/Documents/ever/web.py
```

The four production modules have clearly separated responsibilities: `config.py` holds all constants, `index.py` builds the vector database, `search.py` provides a CLI interface, and `web.py` serves a browser-based UI. Both search interfaces share the same retrieval logic.

## Configuration (config.py)

All tunable parameters live in a single file. Key design choices:

- `DEFAULT_SOURCES = ["obsidian"]` — daily mode processes only the active notes source; `--backfill` expands to all sources
- `MIN_IMAGE_SIZE_BYTES = 20KB` — filters out icons and thumbnails
- `MAX_IMAGE_AGE_DAYS = 30` — daily mode only touches recent images; `--backfill` disables this
- `GEMINI_DEFAULT_DELAY = 2.0s` — throttles to ~12 RPM, safely under the 15 RPM free-tier limit
- `GEMINI_FREE_TIER_DELAY = 4.5s` — more conservative rate during bulk backfill

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
GEMINI_FREE_TIER_DELAY = 4.5            # seconds between calls in --backfill (bulk mode)
GEMINI_DEFAULT_DELAY = 2.0              # seconds between calls in daily mode (~12 RPM, under 15 RPM free tier)
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
                        indexing). Applies free-tier rate limiting between
                        Gemini calls. Resumable.
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
| `.pdf` | Extract text layer → chunk; if no text layer, render pages at 150 DPI and run OCR via primary vision model |
| Images | Two-pass cloud vision (see below); skipped if < 20 KB or older than 30 days in daily mode |

### Two-pass cloud vision

Images go through two sequential model passes. Pass 2 only runs on files Pass 1 succeeded on, so Bedrock costs are not incurred for images Gemini could not process.

```text
Pass 1 — Gemini 2.5 Flash (primary)
  ├─ All new images matching size/age filters
  ├─ Rate-limited: 2.0s delay (daily) or 4.5s delay (--backfill)
  ├─ On quota: flush buffer, break — secondary pass still runs on already-processed images
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
Existing documents in collection: 176345
Checking already-indexed chunks...
Already indexed: 176345 chunks

============================================================
Verifying: obsidian (/home/wook/Documents/obsidiangit)
============================================================
  md    : 589/589 indexed  [OK]
  pdf   : 20/20 indexed  [OK]
  image : 62/1373 indexed  [MISSING 1310, FAILED 1]

```

The 1,310 "MISSING" images are not unindexed — they exist in ChromaDB under the old `glm-ocr` model. `--verify` checks against the current primary model (`gemini/gemini-2.5-flash`) only. Running `--backfill --replace-models glm-ocr` will process them with Gemini and delete the old chunks as it goes.

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

`--no-llm` skips steps 4–5 and returns raw chunks, useful for debugging retrieval quality without needing Ollama running.

### Demo — raw chunk retrieval

```bash
/usr/bin/python3 search.py --no-llm --top-k 3 'FHIR standard' 2>&1 | grep -v 'Loading weights\|Materializing\|LOAD REPORT\|UNEXPECTED\|Notes:\|can be ignored\|Key \|----\|BertModel\|Warning:'
```

```output
Loading embedding model...

Index loaded: 176339 chunks

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

The top result is an image chunk (similarity 0.783) — the image was indexed by a vision model and its text description was embedded just like any other chunk. Image and text chunks compete on equal footing in retrieval.

## ChromaDB Index

```bash
/usr/bin/python3 -c "
import chromadb, sys
sys.path.insert(0, '/home/wook/Documents/ever')
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(COLLECTION_NAME)
print(f'Total chunks: {col.count()}')
sample = col.get(limit=3, include=['metadatas'])
for m in sample['metadatas']:
    print(m)
"
```

```output
Total chunks: 176333
{'collection': 'yarle', 'source_file': '/home/wook/Documents/evern/yarle1/Mono- and biallelic variant effects on disease at biobank scale - Nature.md', 'chunk_index': 0, 'source_type': 'md'}
{'chunk_index': 1, 'collection': 'yarle', 'source_file': '/home/wook/Documents/evern/yarle1/Mono- and biallelic variant effects on disease at biobank scale - Nature.md', 'source_type': 'md'}
{'collection': 'yarle', 'source_file': '/home/wook/Documents/evern/yarle1/Mono- and biallelic variant effects on disease at biobank scale - Nature.md', 'chunk_index': 2, 'source_type': 'md'}
```

```bash
/usr/bin/python3 -c "
import chromadb, sys
sys.path.insert(0, '/home/wook/Documents/ever')
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(COLLECTION_NAME)
img = col.get(where={'source_type': 'image'}, limit=1, include=['metadatas'])
print('Image chunk metadata fields:', list(img['metadatas'][0].keys()))
print('Example:', img['metadatas'][0])
"
```

```output
Image chunk metadata fields: ['source_type', 'collection', 'chunk_index', 'source_file']
Example: {'source_type': 'image', 'collection': 'obsidian', 'chunk_index': 0, 'source_file': '/home/wook/Documents/obsidiangit/2025/attachments/Pasted image 20260127163605.png'}
```

This chunk was indexed by the legacy `glm-ocr` local model, which predates the cloud vision pipeline. `vision_model` is absent for all legacy chunks. Cloud-model chunks (Gemini, Sonnet) always carry `vision_model`; see the Gemini and Sonnet examples below.

```bash
/usr/bin/python3 -c "
import chromadb, sys
sys.path.insert(0, '/home/wook/Documents/ever')
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(COLLECTION_NAME)
img = col.get(where={'vision_model': 'gemini/gemini-2.5-flash'}, limit=1, include=['metadatas'])
print('Gemini chunk metadata:', img['metadatas'][0])
"
```

```output
Gemini chunk metadata: {'collection': 'obsidian', 'source_type': 'image_failed', 'source_file': '/home/wook/Documents/obsidiangit/2026/attachments/Pasted image 20260227145020.png', 'vision_model': 'gemini/gemini-2.5-flash', 'chunk_index': 0}
```

```bash
/usr/bin/python3 -c "
import chromadb, sys
sys.path.insert(0, '/home/wook/Documents/ever')
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(COLLECTION_NAME)
img = col.get(where={'vision_model': 'bedrock/eu.anthropic.claude-sonnet-4-6'}, limit=1, include=['metadatas'])
print('Sonnet chunk metadata:', img['metadatas'][0])
"
```

```output
Sonnet chunk metadata: {'source_file': '/home/wook/Documents/obsidiangit/2026/attachments/Pasted image 20260227144303.png', 'chunk_index': 0, 'vision_model': 'bedrock/eu.anthropic.claude-sonnet-4-6', 'collection': 'obsidian', 'source_type': 'image'}
```

Each chunk carries metadata for filtered retrieval. Image chunks additionally carry `vision_model`, enabling the coexistence of multiple model passes for the same file. Failure sentinels (`source_type: image_failed`) also use this field so re-runs can skip files that have already failed rather than retrying endlessly.

| Metadata field | Values | Used for |
|---|---|---|
| `source_type` | `md`, `pdf`, `image`, `image_failed` | Type filter in search; sentinel detection |
| `collection` | `yarle`, `obsidian` | Source filter in search |
| `source_file` | absolute path | Citation in answers |
| `chunk_index` | 0, 1, 2… | Ordering; first-chunk resume check |
| `vision_model` | model name (image chunks only) | Multi-model coexistence; `--replace-models` targeting |

## Web UI (web.py)

A minimal Flask app exposing the same retrieval pipeline through a browser interface. Models are loaded once at startup and reused across requests. The UI provides filter dropdowns for source collection, content type, and top-k, plus a "Chunks only" toggle that bypasses the LLM.

> **Flask caveat**: `render_template_string()` has a built-in parameter named `source`. Passing a template variable with that name silently breaks rendering. The workaround is to rename it `src_filter` before passing to the template.

## Dependencies

```bash
/usr/bin/python3 -c "
import sentence_transformers, chromadb, importlib.metadata
print('sentence-transformers:', sentence_transformers.__version__)
print('chromadb:', chromadb.__version__)
print('flask:', importlib.metadata.version('flask'))
print('litellm:', importlib.metadata.version('litellm'))
print('pymupdf:', importlib.metadata.version('pymupdf'))
print('ollama:', importlib.metadata.version('ollama'))
"
```

```output
sentence-transformers: 5.2.2
chromadb: 1.5.0
flask: 3.1.2
litellm: 1.81.16
pymupdf: 1.27.1
ollama: 0.6.1
```

All dependencies install via pip:

```bash
pip install sentence-transformers chromadb pymupdf ollama flask litellm
```

`litellm` is the only new addition over the original local-only setup. It is imported lazily inside `_call_vision_model()` and only loaded when a cloud model (identified by `/` in the model name) is actually called, so Ollama-only usage incurs no import cost.
