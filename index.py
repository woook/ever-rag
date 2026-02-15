#!/usr/bin/env python3
"""Indexing script for the personal knowledge base RAG tool. Supports resume."""

import argparse
import base64
import hashlib
import os
import re
import time

import chromadb
import fitz  # PyMuPDF
import ollama
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    IMAGE_EXTENSIONS,
    MIN_IMAGE_SIZE_BYTES,
    OLLAMA_VISION,
    SOURCES,
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def clean_markdown(text: str) -> str:
    """Clean markdown text by removing Excalidraw blocks, excessive HTML, etc."""
    text = re.sub(r'%%\s*\[\[drawing.*?\]\]\s*%%', '', text, flags=re.DOTALL)
    text = re.sub(r'```excalidraw.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'data:[^;]+;base64,[A-Za-z0-9+/=]{100,}', '[embedded data]', text)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def file_id(path: str, suffix: str = "") -> str:
    """Stable ID prefix for a file based on its path and optional suffix."""
    key = path + suffix
    return hashlib.md5(key.encode()).hexdigest()[:12]


def scan_files(source_dir: str) -> dict[str, list[str]]:
    """Walk a directory and collect files by type."""
    files = {"md": [], "pdf": [], "image": []}
    for root, _, filenames in os.walk(source_dir):
        for fname in filenames:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".md":
                files["md"].append(path)
            elif ext == ".pdf":
                files["pdf"].append(path)
            elif ext in IMAGE_EXTENSIONS:
                if os.path.getsize(path) >= MIN_IMAGE_SIZE_BYTES:
                    files["image"].append(path)
    # Newest images first
    files["image"].sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def get_indexed_ids(collection) -> set[str]:
    """Query ChromaDB to get all existing chunk IDs."""
    indexed = set()
    total = collection.count()
    if total == 0:
        return indexed
    BATCH = 5000
    for offset in range(0, total, BATCH):
        results = collection.get(
            offset=offset,
            limit=BATCH,
            include=[],
        )
        indexed.update(results["ids"])
    return indexed


def process_markdown(path: str, collection_name: str) -> list[dict]:
    """Read and chunk a markdown file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print(f"  WARN: could not read {path}: {e}")
        return []

    text = clean_markdown(text)
    chunks = chunk_text(text)
    fid = file_id(path)
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "id": f"{fid}_{i}",
            "text": chunk,
            "metadata": {
                "source_file": path,
                "source_type": "md",
                "collection": collection_name,
                "chunk_index": i,
            },
        })
    return results


def process_pdf(path: str, collection_name: str) -> list[dict]:
    """Extract text from a PDF and chunk it."""
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  WARN: could not open PDF {path}: {e}")
        return []

    full_text = ""
    try:
        for page in doc:
            full_text += page.get_text() + "\n"
    except Exception as e:
        print(f"  WARN: error reading pages from {path}: {e}")
    finally:
        doc.close()

    chunks = chunk_text(full_text)
    fid = file_id(path)
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "id": f"{fid}_{i}",
            "text": chunk,
            "metadata": {
                "source_file": path,
                "source_type": "pdf",
                "collection": collection_name,
                "chunk_index": i,
            },
        })
    return results


def process_image(path: str, collection_name: str, vision_model: str) -> list[dict]:
    """Use Ollama vision model to describe an image, return as chunks."""
    try:
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = ollama.chat(
            model=vision_model,
            messages=[{
                "role": "user",
                "content": "Describe the content of this image in detail, including any text visible.",
                "images": [image_data],
            }],
        )
        description = response["message"]["content"].strip()
        if not description:
            return []

        chunks = chunk_text(description)
        # Include model name in ID so different models produce separate chunks
        fid = file_id(path, suffix=f":{vision_model}")
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "id": f"{fid}_{i}",
                "text": chunk,
                "metadata": {
                    "source_file": path,
                    "source_type": "image",
                    "collection": collection_name,
                    "chunk_index": i,
                    "vision_model": vision_model,
                },
            })
        return results
    except Exception as e:
        print(f"  WARN: image processing failed for {path}: {e}")
        return []


def store_chunks(chunks: list[dict], collection, model: SentenceTransformer):
    """Embed and store a batch of chunks into ChromaDB."""
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def main():
    parser = argparse.ArgumentParser(description="Index knowledge base into ChromaDB")
    parser.add_argument("--skip-images", action="store_true", help="Skip image processing (much faster)")
    parser.add_argument("--only-images", action="store_true", help="Only process images (resume image indexing)")
    parser.add_argument("--only-source", choices=list(SOURCES.keys()), help="Index only one source")
    parser.add_argument("--vision-model", default=OLLAMA_VISION, help=f"Ollama vision model for images (default: {OLLAMA_VISION})")
    parser.add_argument("--reset", action="store_true", help="Delete existing index before building")
    args = parser.parse_args()

    vision_model = args.vision_model

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Existing documents in collection: {collection.count()}")

    # Build set of already-indexed IDs for resume support
    print("Checking already-indexed chunks...")
    indexed_ids = get_indexed_ids(collection)
    print(f"Already indexed: {len(indexed_ids)} chunks")

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    if not args.skip_images:
        print(f"Vision model: {vision_model}")

    sources_to_process = {k: v for k, v in SOURCES.items() if not args.only_source or k == args.only_source}

    FLUSH_EVERY = 64  # store after this many files to avoid losing progress
    t0 = time.time()
    total_new = 0

    for source_name, source_dir in sources_to_process.items():
        print(f"\n{'='*60}")
        print(f"Scanning: {source_name} ({source_dir})")
        print(f"{'='*60}")

        if not os.path.isdir(source_dir):
            print("  Directory not found, skipping.")
            continue

        files = scan_files(source_dir)
        print(f"  Found: {len(files['md'])} markdown, {len(files['pdf'])} PDFs, {len(files['image'])} images")

        # --- Markdown ---
        if not args.only_images:
            # Skip files where first chunk ID already exists
            new_md = [p for p in files["md"] if f"{file_id(p)}_0" not in indexed_ids]
            print(f"\n  Markdown: {len(new_md)} new (skipping {len(files['md']) - len(new_md)} already indexed)")
            buffer = []
            for i, path in enumerate(new_md, 1):
                buffer.extend(process_markdown(path, source_name))
                if i % FLUSH_EVERY == 0 or i == len(new_md):
                    store_chunks(buffer, collection, model)
                    total_new += len(buffer)
                    buffer = []
                    print(f"    [{i}/{len(new_md)}] markdown — {total_new} chunks stored")

        # --- PDFs ---
        if not args.only_images:
            new_pdf = [p for p in files["pdf"] if f"{file_id(p)}_0" not in indexed_ids]
            print(f"\n  PDFs: {len(new_pdf)} new (skipping {len(files['pdf']) - len(new_pdf)} already indexed)")
            buffer = []
            for i, path in enumerate(new_pdf, 1):
                buffer.extend(process_pdf(path, source_name))
                if i % FLUSH_EVERY == 0 or i == len(new_pdf):
                    store_chunks(buffer, collection, model)
                    total_new += len(buffer)
                    buffer = []
                    print(f"    [{i}/{len(new_pdf)}] PDFs — {total_new} chunks stored")

        # --- Images ---
        if args.skip_images:
            print("\n  Skipping images (--skip-images)")
        else:
            # Check using model-specific IDs so each model gets its own pass
            new_img = [p for p in files["image"]
                       if f"{file_id(p, suffix=f':{vision_model}')}_0" not in indexed_ids]
            print(f"\n  Images ({vision_model}): {len(new_img)} new (skipping {len(files['image']) - len(new_img)} already indexed)")
            buffer = []
            for i, path in enumerate(new_img, 1):
                buffer.extend(process_image(path, source_name, vision_model))
                # Flush more frequently for images since each one is slow
                if i % 10 == 0 or i == len(new_img):
                    store_chunks(buffer, collection, model)
                    total_new += len(buffer)
                    buffer = []
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (len(new_img) - i) / rate if rate > 0 else 0
                    print(f"    [{i}/{len(new_img)}] images — {total_new} chunks stored — ~{remaining/60:.0f}min remaining")

    elapsed = time.time() - t0
    print(f"\nDone! Added {total_new} new chunks in {elapsed:.1f}s")
    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
