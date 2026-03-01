#!/usr/bin/env python3
"""MCP server exposing the personal notes RAG as a tool for Claude.

Loaded once at startup; Claude calls search_notes() as a tool during conversations.
Returns raw chunks with metadata so Claude can synthesise the answer.
"""

import os
import sys

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
        top_k: Number of chunks to return (default 5, max ~20).
        date_after: Only return notes on or after this ISO date, e.g. "2025-03-01".
            Note: implemented as a post-filter over top_k*20 candidates — if most
            chunks predate the cutoff you may receive fewer than top_k results.
    """
    if source and source not in {"obsidian", "yarle"}:
        return f"Invalid source '{source}'. Must be 'obsidian' or 'yarle'."
    if content_type and content_type not in {"md", "pdf", "image"}:
        return f"Invalid content_type '{content_type}'. Must be 'md', 'pdf', or 'image'."

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
