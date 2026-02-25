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
    return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]


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


def get_failed_source_files(collection, vision_model: str) -> set[str]:
    """Return source_file paths for images that have a failure sentinel for this vision model."""
    try:
        results = collection.get(
            where={"$and": [{"source_type": "image_failed"}, {"vision_model": vision_model}]},
            include=["metadatas"],
        )
        return {m["source_file"] for m in results["metadatas"]}
    except Exception:
        return set()


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
    if not chunks:
        print(f"  WARN: no text extracted from {path} (empty or fully stripped)")
        return []
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


def process_pdf(path: str, collection_name: str, vision_model: str | None = None) -> tuple[list[dict], bool]:
    """Extract text from a PDF. Falls back to vision model OCR if no text layer found.

    Returns (chunks, used_ocr) so callers can count OCR attempts for --max-images.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  WARN: could not open PDF {path}: {e}")
        return [], False

    full_text = ""
    page_images = []
    try:
        for page in doc:
            full_text += page.get_text() + "\n"
        # Render pages to images now (before closing) if OCR fallback may be needed
        if not chunk_text(full_text) and vision_model:
            for page in doc:
                try:
                    png = page.get_pixmap(dpi=150).tobytes("png")
                    page_images.append(base64.b64encode(png).decode("utf-8"))
                except Exception as e:
                    print(f"  WARN: could not render page {page.number} of {path}: {e}")
    except Exception as e:
        print(f"  WARN: error reading pages from {path}: {e}")
    finally:
        doc.close()

    chunks = chunk_text(full_text)
    used_ocr = False

    if not chunks:
        if not page_images:
            print(f"  WARN: no text extracted from {path} (scanned/image-only PDF?)")
            return [], False
        print(f"  INFO: no text layer in {path}, running OCR ({vision_model})")
        ocr_parts = []
        for i, image_data in enumerate(page_images):
            try:
                response = ollama.chat(
                    model=vision_model,
                    messages=[{
                        "role": "user",
                        "content": "Extract all text from this document page. Output only the text content.",
                        "images": [image_data],
                    }],
                )
                text = response.message.content.strip()
                if text:
                    ocr_parts.append(text)
            except Exception as e:
                print(f"  WARN: OCR failed for page {i} of {path}: {e}")
        chunks = chunk_text("\n\n".join(ocr_parts))
        if not chunks:
            print(f"  WARN: OCR produced no text for {path}")
            return [], True
        used_ocr = True

    fid = file_id(path)
    results = []
    for i, chunk in enumerate(chunks):
        meta = {
            "source_file": path,
            "source_type": "pdf",
            "collection": collection_name,
            "chunk_index": i,
        }
        if used_ocr:
            meta["vision_model"] = vision_model
        results.append({"id": f"{fid}_{i}", "text": chunk, "metadata": meta})
    return results, used_ocr


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
        description = response.message.content.strip()
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
        fid = file_id(path, suffix=f":{vision_model}")
        return [{
            "id": f"{fid}_0",
            "text": "[image processing failed]",
            "metadata": {
                "source_file": path,
                "source_type": "image_failed",
                "collection": collection_name,
                "chunk_index": 0,
                "vision_model": vision_model,
            },
        }]


CHROMA_BATCH_SIZE = 5000  # ChromaDB max batch size is ~5461; stay safely under


def store_chunks(chunks: list[dict], collection, model: SentenceTransformer):
    """Embed and store a batch of chunks into ChromaDB."""
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )


def main():
    parser = argparse.ArgumentParser(description="Index knowledge base into ChromaDB")
    parser.add_argument("--skip-images", action="store_true", help="Skip image processing (much faster)")
    parser.add_argument("--only-images", action="store_true", help="Only process images (resume image indexing)")
    parser.add_argument("--only-source", choices=list(SOURCES.keys()), help="Index only one source")
    parser.add_argument("--vision-model", default=OLLAMA_VISION, help=f"Ollama vision model for images (default: {OLLAMA_VISION})")
    parser.add_argument("--reset", action="store_true", help="Delete existing index before building")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images processed (useful for testing)")
    parser.add_argument("--verify", action="store_true", help="Check which files on disk are indexed; no indexing is performed")
    parser.add_argument("--fix", action="store_true", help="Use with --verify to index any missing files found")
    args = parser.parse_args()

    # Suppress MuPDF's internal diagnostic output — it crashes on PDFs with
    # surrogate characters in error messages. Our own WARN handling is sufficient.
    fitz.TOOLS.mupdf_display_errors(False)

    vision_model = args.vision_model

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection.")
        except (ValueError, Exception) as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                pass  # collection doesn't exist yet
            else:
                raise

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Existing documents in collection: {collection.count()}")

    # Build set of already-indexed IDs for resume support
    print("Checking already-indexed chunks...")
    indexed_ids = get_indexed_ids(collection)
    print(f"Already indexed: {len(indexed_ids)} chunks")

    if args.verify:
        sources_to_verify = {k: v for k, v in SOURCES.items() if not args.only_source or k == args.only_source}
        missing_files = {}  # source_name -> {"md": [...], "pdf": [...], "image": [...]}
        print()
        for source_name, source_dir in sources_to_verify.items():
            print(f"{'='*60}")
            print(f"Verifying: {source_name} ({source_dir})")
            print(f"{'='*60}")
            if not os.path.isdir(source_dir):
                print("  Directory not found, skipping.")
                continue
            files = scan_files(source_dir)
            missing_files[source_name] = {"md": [], "pdf": [], "image": []}
            failed_image_files = get_failed_source_files(collection, vision_model)
            for file_type, paths in [("md", files["md"]), ("pdf", files["pdf"]), ("image", files["image"])]:
                if file_type == "image":
                    missing = [p for p in paths if f"{file_id(p, suffix=f':{vision_model}')}_0" not in indexed_ids]
                    failed = [p for p in paths if p in failed_image_files]
                    n_ok = len(paths) - len(missing) - len(failed)
                    status_parts = []
                    if missing:
                        status_parts.append(f"MISSING {len(missing)}")
                    if failed:
                        status_parts.append(f"FAILED {len(failed)}")
                    status = "OK" if not status_parts else ", ".join(status_parts)
                    print(f"  {file_type:6s}: {n_ok}/{len(paths)} indexed  [{status}]")
                else:
                    missing = [p for p in paths if f"{file_id(p)}_0" not in indexed_ids]
                    missing_files[source_name][file_type] = missing
                    indexed = len(paths) - len(missing)
                    status = "OK" if not missing else f"MISSING {len(missing)}"
                    print(f"  {file_type:6s}: {indexed}/{len(paths)} indexed  [{status}]")
                    if missing:
                        for p in missing:
                            print(f"    - {p}")
                missing_files[source_name][file_type] = missing
        print()

        if not args.fix:
            return

        total_missing = sum(len(v) for src in missing_files.values() for v in src.values())
        if total_missing == 0:
            print("Nothing to fix.")
            return

        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        if not args.skip_images:
            print(f"Vision model: {vision_model}")

        FLUSH_EVERY = 64
        t0 = time.time()
        total_new = 0

        for source_name, missing in missing_files.items():
            if not any(missing.values()):
                continue
            print(f"\n{'='*60}")
            print(f"Fixing: {source_name}")
            print(f"{'='*60}")

            if not args.only_images:
                print(f"\n  Markdown: {len(missing['md'])} missing")
                buffer = []
                for i, path in enumerate(missing["md"], 1):
                    buffer.extend(process_markdown(path, source_name))
                    if i % FLUSH_EVERY == 0 or i == len(missing["md"]):
                        store_chunks(buffer, collection, model)
                        total_new += len(buffer)
                        buffer = []
                        print(f"    [{i}/{len(missing['md'])}] markdown — {total_new} chunks stored")

                print(f"\n  PDFs: {len(missing['pdf'])} missing")
                buffer = []
                ocr_count = 0
                for i, path in enumerate(missing["pdf"], 1):
                    ocr_limit_reached = args.max_images is not None and ocr_count >= args.max_images
                    vm = None if (args.skip_images or ocr_limit_reached) else vision_model
                    chunks, used_ocr = process_pdf(path, source_name, vm)
                    if used_ocr:
                        ocr_count += 1
                    buffer.extend(chunks)
                    if i % FLUSH_EVERY == 0 or i == len(missing["pdf"]):
                        store_chunks(buffer, collection, model)
                        total_new += len(buffer)
                        buffer = []
                        print(f"    [{i}/{len(missing['pdf'])}] PDFs — {total_new} chunks stored")

            if not args.skip_images:
                new_img = missing["image"]
                if args.max_images is not None:
                    new_img = new_img[:args.max_images]
                print(f"\n  Images ({vision_model}): {len(new_img)} missing")
                buffer = []
                images_t0 = time.time()
                for i, path in enumerate(new_img, 1):
                    buffer.extend(process_image(path, source_name, vision_model))
                    if i % 10 == 0 or i == len(new_img):
                        store_chunks(buffer, collection, model)
                        total_new += len(buffer)
                        buffer = []
                        elapsed = time.time() - images_t0
                        rate = i / elapsed if elapsed > 0 else 0
                        remaining = (len(new_img) - i) / rate if rate > 0 else 0
                        print(f"    [{i}/{len(new_img)}] images — {total_new} chunks stored — ~{remaining/60:.0f}min remaining")

        elapsed = time.time() - t0
        print(f"\nDone! Added {total_new} new chunks in {elapsed:.1f}s")
        print(f"Total documents in collection: {collection.count()}")
        return

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
            ocr_count = 0
            for i, path in enumerate(new_pdf, 1):
                ocr_limit_reached = args.max_images is not None and ocr_count >= args.max_images
                vm = None if (args.skip_images or ocr_limit_reached) else vision_model
                chunks, used_ocr = process_pdf(path, source_name, vm)
                if used_ocr:
                    ocr_count += 1
                buffer.extend(chunks)
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
            if args.max_images is not None:
                new_img = new_img[:args.max_images]
            print(f"\n  Images ({vision_model}): {len(new_img)} new (skipping {len(files['image']) - len(new_img)} already indexed)")
            buffer = []
            images_t0 = time.time()
            for i, path in enumerate(new_img, 1):
                buffer.extend(process_image(path, source_name, vision_model))
                # Flush more frequently for images since each one is slow
                if i % 10 == 0 or i == len(new_img):
                    store_chunks(buffer, collection, model)
                    total_new += len(buffer)
                    buffer = []
                    elapsed = time.time() - images_t0
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (len(new_img) - i) / rate if rate > 0 else 0
                    print(f"    [{i}/{len(new_img)}] images — {total_new} chunks stored — ~{remaining/60:.0f}min remaining")

    elapsed = time.time() - t0
    print(f"\nDone! Added {total_new} new chunks in {elapsed:.1f}s")
    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
