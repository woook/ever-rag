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
    DEFAULT_SOURCES,
    DEFAULT_VISION_MODELS,
    EMBEDDING_MODEL,
    IMAGE_EXTENSIONS,
    MAX_IMAGE_AGE_DAYS,
    MIN_IMAGE_SIZE_BYTES,
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


def scan_files(source_dir: str, min_size_bytes: int = MIN_IMAGE_SIZE_BYTES,
               max_age_days: int | None = MAX_IMAGE_AGE_DAYS) -> dict[str, list[str]]:
    """Walk a directory and collect files by type."""
    files = {"md": [], "pdf": [], "image": []}
    cutoff = time.time() - max_age_days * 86400 if max_age_days is not None else None
    for root, _, filenames in os.walk(source_dir):
        for fname in filenames:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".md":
                files["md"].append(path)
            elif ext == ".pdf":
                files["pdf"].append(path)
            elif ext in IMAGE_EXTENSIONS:
                if os.path.getsize(path) >= min_size_bytes:
                    if cutoff is None or os.path.getmtime(path) >= cutoff:
                        files["image"].append(path)
    # Newest images first (by modification time)
    files["image"].sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _call_vision_model(vision_model: str, image_b64: str, prompt: str,
                       mime_type: str = "image/png", max_retries: int = 8) -> str:
    """Call a vision model (cloud via LiteLLM or local via Ollama) and return response text.

    For cloud models, retries on rate limits using the Retry-After header when available,
    falling back to exponential backoff (5s, 10s, 20s, ...).
    """
    if "/" in vision_model:
        import litellm
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=vision_model,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                if "RateLimitError" not in type(e).__name__ and "rate_limit" not in str(e).lower():
                    raise
                if attempt == max_retries - 1:
                    raise
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                wait = int(retry_after) if retry_after else (5 * 2 ** attempt)
                print(f"  Rate limited by {vision_model}. Waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
    else:
        response = ollama.chat(
            model=vision_model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        )
        return (response.message.content or "").strip()


def get_failed_source_files(collection, vision_model: str) -> set[str]:
    """Return source_file paths for images that have a failure sentinel for this vision model."""
    try:
        results = collection.get(
            where={"$and": [{"source_type": "image_failed"}, {"vision_model": vision_model}]},
            include=["metadatas"],
        )
        return {m["source_file"] for m in results["metadatas"]}
    except Exception as e:
        print(f"  WARN: could not read failed image sentinels: {e}")
        return set()


def get_successfully_indexed_images(collection, vision_model: str) -> set[str]:
    """Return source_file paths that have at least one successful chunk for this vision model."""
    try:
        results = collection.get(
            where={"$and": [{"source_type": "image"}, {"vision_model": vision_model}]},
            include=["metadatas"],
        )
        return {m["source_file"] for m in results["metadatas"]}
    except Exception as e:
        print(f"  WARN: could not query successfully indexed images: {e}")
        return set()


def delete_image_chunks(collection, path: str, vision_model: str) -> int:
    """Delete all chunks (including failure sentinels) for an image/model pair. Returns count."""
    try:
        results = collection.get(
            where={"$and": [{"source_file": path}, {"vision_model": vision_model}]},
            include=[],
        )
        ids = results["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception as e:
        print(f"  WARN: could not delete {vision_model} chunks for {path}: {e}")
        return 0


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
    needs_ocr = False
    try:
        for page in doc:
            full_text += page.get_text() + "\n"
        needs_ocr = not chunk_text(full_text) and bool(vision_model)
    except Exception as e:
        print(f"  WARN: error reading pages from {path}: {e}")
    finally:
        doc.close()

    chunks = chunk_text(full_text)
    used_ocr = False

    if not chunks:
        if not needs_ocr:
            print(f"  WARN: no text extracted from {path} (scanned/image-only PDF?)")
            return [], False
        print(f"  INFO: no text layer in {path}, running OCR ({vision_model})")
        ocr_parts = []
        ocr_doc = fitz.open(path)
        try:
            for i, page in enumerate(ocr_doc):
                try:
                    image_data = base64.b64encode(page.get_pixmap(dpi=150).tobytes("png")).decode("utf-8")
                    text = _call_vision_model(
                        vision_model,
                        image_data,
                        "Extract all text from this document page. Output only the text content.",
                    )
                    if text:
                        ocr_parts.append(text)
                except Exception as e:
                    print(f"  WARN: OCR failed for page {i} of {path}: {e}")
        finally:
            ocr_doc.close()
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
    """Use a vision model to describe an image, return as chunks."""
    try:
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(path)[1].lower()
        mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                     ".webp": "image/webp"}.get(ext, "image/png")
        description = _call_vision_model(
            vision_model,
            image_data,
            "Describe the content of this image in detail, including any text visible.",
            mime_type=mime_type,
        )
        if not description:
            raise RuntimeError("vision model returned empty content")

        chunks = chunk_text(description)
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
        if "RateLimitError" in type(e).__name__ or "rate_limit" in str(e).lower():
            print(f"  WARN: rate limit hit for {path}: {e}")
            raise  # propagate — outer loop will catch and stop the pass
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
    parser.add_argument("--vision-models", nargs="+", default=DEFAULT_VISION_MODELS, metavar="MODEL",
        help=f"Vision model(s) in order. First runs on all new images; subsequent models only run on "
             f"images the first model succeeded on. Default: {DEFAULT_VISION_MODELS}")
    parser.add_argument("--backfill", action="store_true",
        help="Process all images regardless of age (for bulk initial indexing). Resumable.")
    parser.add_argument("--min-image-kb", type=int, default=None,
        help="Minimum image size in KB. Default: 20 (from config).")
    parser.add_argument("--reset", action="store_true", help="Delete existing index before building")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images processed (useful for testing)")
    parser.add_argument("--replace-models", nargs="+", default=None, metavar="MODEL",
        help="Delete chunks from these models when the primary model succeeds on the same image "
             "(e.g. --replace-models glm-ocr to remove old local-model chunks during backfill).")
    parser.add_argument("--verify", action="store_true", help="Check which files on disk are indexed; no indexing is performed")
    parser.add_argument("--fix", action="store_true", help="Use with --verify to index any missing files found")
    args = parser.parse_args()
    if args.fix and not args.verify:
        parser.error("--fix requires --verify")

    # Suppress MuPDF's internal diagnostic output — it crashes on PDFs with
    # surrogate characters in error messages. Our own WARN handling is sufficient.
    fitz.TOOLS.mupdf_display_errors(False)

    vision_models = args.vision_models
    primary_model = vision_models[0]
    secondary_models = vision_models[1:]
    min_size = (args.min_image_kb * 1024) if args.min_image_kb else MIN_IMAGE_SIZE_BYTES

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

    # --- Source selection ---
    if args.only_source:
        sources_to_process = {args.only_source: SOURCES[args.only_source]}
    elif args.backfill:
        sources_to_process = dict(SOURCES)
    else:
        sources_to_process = {s: SOURCES[s] for s in DEFAULT_SOURCES}

    if args.verify:
        if args.only_source:
            sources_to_verify = {args.only_source: SOURCES[args.only_source]}
        elif args.backfill:
            sources_to_verify = dict(SOURCES)
        else:
            sources_to_verify = {s: SOURCES[s] for s in DEFAULT_SOURCES}

        missing_files = {}  # source_name -> {"md": [...], "pdf": [...], "image": [...]}
        print()
        for source_name, source_dir in sources_to_verify.items():
            print(f"{'='*60}")
            print(f"Verifying: {source_name} ({source_dir})")
            print(f"{'='*60}")
            if not os.path.isdir(source_dir):
                print("  Directory not found, skipping.")
                continue
            # Scan all images for verify (no age filter — show full picture)
            files = scan_files(source_dir, min_size_bytes=min_size, max_age_days=None)
            missing_files[source_name] = {"md": [], "pdf": [], "image": []}
            failed_image_files = get_failed_source_files(collection, primary_model)
            for file_type, paths in [("md", files["md"]), ("pdf", files["pdf"]), ("image", files["image"])]:
                if file_type == "image":
                    missing = [p for p in paths if f"{file_id(p, suffix=f':{primary_model}')}_0" not in indexed_ids]
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
            print(f"Vision model: {primary_model}")

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
                    vm = None if (args.skip_images or ocr_limit_reached) else primary_model
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
                print(f"\n  Images ({primary_model}): {len(new_img)} missing")
                buffer = []
                images_t0 = time.time()
                for i, path in enumerate(new_img, 1):
                    result = process_image(path, source_name, primary_model)
                    if result and result[0]["metadata"].get("source_type") == "image_failed":
                        store_chunks(result, collection, model)
                        total_new += len(result)
                    else:
                        buffer.extend(result)
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
        print(f"Vision models: {vision_models}")

    FLUSH_EVERY = 64  # store after this many files to avoid losing progress
    t0 = time.time()
    total_new = 0
    total_deleted = 0

    for source_name, source_dir in sources_to_process.items():
        print(f"\n{'='*60}")
        print(f"Scanning: {source_name} ({source_dir})")
        print(f"{'='*60}")

        if not os.path.isdir(source_dir):
            print("  Directory not found, skipping.")
            continue

        max_age = None if args.backfill else MAX_IMAGE_AGE_DAYS
        files = scan_files(source_dir, min_size_bytes=min_size, max_age_days=max_age)
        age_note = "all ages" if max_age is None else f"≤{max_age}d old"
        print(f"  Found: {len(files['md'])} markdown, {len(files['pdf'])} PDFs, {len(files['image'])} images ({age_note})")

        # --- Markdown ---
        if not args.only_images:
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
                vm = None if (args.skip_images or ocr_limit_reached) else primary_model
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
            # Pass 1: primary model (Gemini)
            new_img = [p for p in files["image"]
                       if f"{file_id(p, suffix=f':{primary_model}')}_0" not in indexed_ids]
            if args.max_images is not None:
                new_img = new_img[:args.max_images]
            print(f"\n  Images pass 1 ({primary_model}): {len(new_img)} new "
                  f"(skipping {len(files['image']) - len(new_img)} already indexed)")
            buffer = []
            images_t0 = time.time()
            for i, path in enumerate(new_img, 1):
                try:
                    result = process_image(path, source_name, primary_model)
                except Exception as e:
                    if "RateLimitError" in type(e).__name__ or "rate_limit" in str(e).lower():
                        print(f"\n  WARN: {primary_model} quota reached after {i-1} images. "
                              "Re-run later to continue. Secondary pass will proceed on already-processed images.")
                        if buffer:
                            store_chunks(buffer, collection, model)
                            total_new += len(buffer)
                            buffer = []
                        break
                    raise
                if result and result[0]["metadata"].get("source_type") == "image_failed":
                    store_chunks(result, collection, model)
                    total_new += len(result)
                else:
                    for old_model in (args.replace_models or []):
                        total_deleted += delete_image_chunks(collection, path, old_model)
                    buffer.extend(result)
                    if i % 10 == 0 or i == len(new_img):
                        store_chunks(buffer, collection, model)
                        total_new += len(buffer)
                        buffer = []
                elapsed = time.time() - images_t0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(new_img) - i) / rate if rate > 0 else 0
                print(f"    [{i}/{len(new_img)}] pass1 — {total_new + len(buffer)} chunks stored — ~{remaining/60:.0f}min remaining")

            # Pass 2: secondary models (Sonnet), only where primary succeeded
            if secondary_models:
                gemini_success = get_successfully_indexed_images(collection, primary_model)
                for sec_model in secondary_models:
                    new_sec = [p for p in files["image"]
                               if f"{file_id(p, suffix=f':{sec_model}')}_0" not in indexed_ids
                               and p in gemini_success]
                    if args.max_images is not None:
                        new_sec = new_sec[:args.max_images]
                    print(f"\n  Images pass 2 ({sec_model}): {len(new_sec)} new "
                          f"(filtered to Gemini successes)")
                    buffer = []
                    images_t0 = time.time()
                    for i, path in enumerate(new_sec, 1):
                        try:
                            result = process_image(path, source_name, sec_model)
                        except Exception as e:
                            if "RateLimitError" in type(e).__name__ or "rate_limit" in str(e).lower():
                                print(f"\n  WARN: {sec_model} quota reached after {i-1} images. Re-run later to continue.")
                                if buffer:
                                    store_chunks(buffer, collection, model)
                                    total_new += len(buffer)
                                    buffer = []
                                break
                            raise
                        if result and result[0]["metadata"].get("source_type") == "image_failed":
                            store_chunks(result, collection, model)
                            total_new += len(result)
                        else:
                            buffer.extend(result)
                            if i % 10 == 0 or i == len(new_sec):
                                store_chunks(buffer, collection, model)
                                total_new += len(buffer)
                                buffer = []
                        elapsed = time.time() - images_t0
                        rate = i / elapsed if elapsed > 0 else 0
                        remaining = (len(new_sec) - i) / rate if rate > 0 else 0
                        print(f"    [{i}/{len(new_sec)}] pass2 — {total_new + len(buffer)} chunks stored — ~{remaining/60:.0f}min remaining")

    elapsed = time.time() - t0
    summary = f"\nDone! Added {total_new} new chunks in {elapsed:.1f}s"
    if total_deleted:
        summary += f" (deleted {total_deleted} replaced chunks)"
    print(summary)
    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
