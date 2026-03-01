"""Lightweight date-extraction helper shared by index.py and migrate_add_dates.py.

Kept as a separate module so migrate_add_dates.py does not pull in the heavy
indexing dependencies (chromadb, sentence-transformers, fitz, etc.) that
index.py imports at the top level.
"""

import os
import re
import sys
from datetime import datetime


def extract_note_date(path: str, collection_name: str, text: str = "") -> str:
    """Extract note date as 'YYYY-MM-DD'.

    Priority:
    1. Obsidian: date prefix in filename (e.g. "2026-02-27 Epic.md")
    2. Yarle: 'Created at:' line in the first 15 lines of text
    3. Fallback: file modification time
    """
    if collection_name == "obsidian":
        m = re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(path))
        if m:
            try:
                datetime.strptime(m.group(0), "%Y-%m-%d")
                return m.group(0)
            except ValueError:
                pass
    elif collection_name == "yarle" and text:
        for line in text.splitlines()[:15]:
            if line.startswith("Created at:"):
                m = re.search(r'\d{4}-\d{2}-\d{2}', line)
                if m:
                    try:
                        datetime.strptime(m.group(0), "%Y-%m-%d")
                        return m.group(0)
                    except ValueError:
                        pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except Exception as e:
        print(f"WARN: could not read mtime for {path}: {e} — using 1970-01-01", file=sys.stderr)
        return "1970-01-01"
