#!/usr/bin/env python3
"""One-off migration: patch note_date metadata onto all existing ChromaDB chunks.

Reads source_file from each chunk's existing metadata and derives the date using
the same logic as extract_note_date() in index.py. No model calls — just metadata
writes. Safe to re-run: chunks that already have note_date are skipped.
"""

import os

import chromadb

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from note_date_utils import extract_note_date


def main():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"Total chunks in collection: {total}")

    FETCH_BATCH = 5000
    to_update_ids = []
    to_update_metadatas = []
    already_have_date = 0

    for offset in range(0, total, FETCH_BATCH):
        results = collection.get(
            offset=offset,
            limit=FETCH_BATCH,
            include=["metadatas"],
        )
        for chunk_id, meta in zip(results["ids"], results["metadatas"], strict=True):
            if not isinstance(meta, dict):
                print(f"  WARN: skipping chunk {chunk_id} — unexpected metadata type: {type(meta)}")
                continue
            if "note_date" in meta:
                already_have_date += 1
                continue

            path = meta.get("source_file", "")
            coll = meta.get("collection", "")

            # For yarle md files read the first 15 lines to find "Created at:"
            text = ""
            if coll == "yarle" and path.endswith(".md") and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        lines = []
                        for _ in range(15):
                            line = f.readline()
                            if not line:
                                break
                            lines.append(line)
                        text = "".join(lines)
                except Exception as e:
                    print(f"  WARN: could not read {path} for date extraction: {e}")

            note_date = extract_note_date(path, coll, text)
            new_meta = dict(meta)
            new_meta["note_date"] = note_date
            to_update_ids.append(chunk_id)
            to_update_metadatas.append(new_meta)

        fetched = min(offset + FETCH_BATCH, total)
        print(f"  Scanned {fetched}/{total} chunks...")

    print(f"\nAlready have note_date: {already_have_date}")
    print(f"Chunks to update: {len(to_update_ids)}")

    if not to_update_ids:
        print("Nothing to do.")
        return

    UPDATE_BATCH = 5000
    for i in range(0, len(to_update_ids), UPDATE_BATCH):
        collection.update(
            ids=to_update_ids[i:i + UPDATE_BATCH],
            metadatas=to_update_metadatas[i:i + UPDATE_BATCH],
        )
        updated = min(i + UPDATE_BATCH, len(to_update_ids))
        print(f"  Updated {updated}/{len(to_update_ids)} chunks")

    print("Done!")


if __name__ == "__main__":
    main()
