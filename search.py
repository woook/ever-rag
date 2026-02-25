#!/usr/bin/env python3
"""Interactive search CLI for the personal knowledge base."""

import argparse
import sys

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_LLM,
)

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Cite the source file for each piece of information you use. "
    "If the context doesn't contain enough information, say so."
)


def build_context(results) -> str:
    """Format ChromaDB results into a context string for the LLM."""
    parts = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        source = meta.get("source_file", "unknown")
        stype = meta.get("source_type", "unknown")
        similarity = 1 - dist  # cosine distance → similarity
        parts.append(f"[Source: {source} ({stype}, similarity={similarity:.3f})]\n{doc}")
    return "\n\n---\n\n".join(parts)


def search(collection, model, query: str, k: int = 5, where_filter: dict | None = None) -> dict:
    """Embed query and retrieve top-k chunks."""
    embedding = model.encode([query]).tolist()
    kwargs = {
        "query_embeddings": embedding,
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter
    return collection.query(**kwargs)


def ask_llm(context: str, question: str) -> str:
    """Send context + question to Ollama LLM and return the answer."""
    response = ollama.chat(
        model=OLLAMA_LLM,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.message.content


def build_where_filter(source: str | None, content_type: str | None) -> dict | None:
    """Build a ChromaDB where filter from CLI flags."""
    conditions = []
    if source:
        conditions.append({"collection": source})
    if content_type:
        conditions.append({"source_type": content_type})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def main():
    parser = argparse.ArgumentParser(description="Search your knowledge base")
    parser.add_argument("query", nargs="*", help="One-shot query (omit for interactive mode)")
    parser.add_argument("--source", choices=["yarle", "obsidian"], help="Filter by source collection")
    parser.add_argument("--type", choices=["md", "pdf", "image"], dest="content_type", help="Filter by content type")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--no-llm", action="store_true", help="Show raw chunks without LLM synthesis")
    args = parser.parse_args()

    # Load resources
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except ValueError:
        print("Error: No index found. Run index.py first.")
        sys.exit(1)

    print(f"Index loaded: {collection.count()} chunks")

    where_filter = build_where_filter(args.source, args.content_type)

    # One-shot mode
    if args.query:
        query = " ".join(args.query)
        results = search(collection, model, query, k=args.top_k, where_filter=where_filter)
        context = build_context(results)

        if args.no_llm:
            print(f"\n{'='*60}")
            print(context)
        else:
            print(f"\nRetrieved {len(results['documents'][0])} chunks, generating answer...\n")
            answer = ask_llm(context, query)
            print(answer)
        return

    # Interactive mode
    print("\nEnter your questions (Ctrl+C or 'quit' to exit):\n")
    while True:
        try:
            query = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        results = search(collection, model, query, k=args.top_k, where_filter=where_filter)
        context = build_context(results)

        if args.no_llm:
            print(f"\n{'='*60}")
            print(context)
            print(f"{'='*60}\n")
        else:
            print(f"\nRetrieved {len(results['documents'][0])} chunks, generating answer...\n")
            answer = ask_llm(context, query)
            print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
