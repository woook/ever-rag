#!/usr/bin/env python3
"""Web frontend for the personal knowledge base RAG tool."""

import os
import time

import chromadb
import ollama
from flask import Flask, render_template_string, request
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_LLM,
)

app = Flask(__name__)

# Load once at startup
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Index loaded: {collection.count()} chunks")
except Exception:
    print("Warning: No index found. Run index.py first. Searches will fail.")
    collection = None

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Cite the source file for each piece of information you use. "
    "If the context doesn't contain enough information, say so."
)

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Base Search</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #f5f5f5; color: #222; }
  .container { max-width: 860px; margin: 0 auto; padding: 1.5rem; }
  h1 { font-size: 1.4rem; margin-bottom: 1rem; color: #333; }
  form { background: #fff; padding: 1.2rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
  .search-row { display: flex; gap: 0.5rem; margin-bottom: 0.8rem; }
  input[type=text] { flex: 1; padding: 0.6rem 0.8rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }
  input[type=text]:focus { outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }
  button { padding: 0.6rem 1.2rem; font-size: 1rem; background: #4a90d9; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
  button:hover { background: #3a7bc8; }
  .filters { display: flex; gap: 1rem; align-items: center; font-size: 0.85rem; color: #666; }
  .filters select { padding: 0.3rem 0.5rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85rem; }
  .filters label { display: flex; align-items: center; gap: 0.3rem; }
  .answer-box { background: #fff; padding: 1.2rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem; }
  .answer-box h2 { font-size: 1rem; color: #666; margin-bottom: 0.6rem; }
  .answer-content { line-height: 1.6; white-space: pre-wrap; }
  .meta { font-size: 0.8rem; color: #999; margin-top: 0.5rem; }
  .chunk { background: #fff; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 0.6rem; }
  .chunk-source { font-size: 0.75rem; color: #4a90d9; word-break: break-all; margin-bottom: 0.4rem; }
  .chunk-text { font-size: 0.85rem; line-height: 1.5; color: #444; }
  .chunk-sim { font-size: 0.7rem; color: #999; float: right; }
  .spinner { display: none; }
  form.loading .spinner { display: inline-block; }
  form.loading button span { display: none; }
</style>
</head>
<body>
<div class="container">
  <h1>Knowledge Base Search</h1>
  <form method="POST" onsubmit="this.classList.add('loading')">
    <div class="search-row">
      <input type="text" name="query" placeholder="Ask a question..." value="{{ query or '' }}" autofocus required>
      <button type="submit"><span>Search</span><span class="spinner">...</span></button>
    </div>
    <div class="filters">
      <label>Source:
        <select name="src_filter">
          <option value="">All</option>
          <option value="yarle" {{ 'selected' if src_filter == 'yarle' }}>Yarle / Evernote</option>
          <option value="obsidian" {{ 'selected' if src_filter == 'obsidian' }}>Obsidian</option>
        </select>
      </label>
      <label>Type:
        <select name="content_type">
          <option value="">All</option>
          <option value="md" {{ 'selected' if content_type == 'md' }}>Markdown</option>
          <option value="pdf" {{ 'selected' if content_type == 'pdf' }}>PDF</option>
          <option value="image" {{ 'selected' if content_type == 'image' }}>Image</option>
        </select>
      </label>
      <label>Top-k:
        <select name="top_k">
          {% for k in [3, 5, 10, 15] %}
          <option value="{{ k }}" {{ 'selected' if top_k == k }}>{{ k }}</option>
          {% endfor %}
        </select>
      </label>
      <label>
        <input type="checkbox" name="no_llm" {{ 'checked' if no_llm }}> Chunks only
      </label>
    </div>
  </form>

  {% if answer %}
  <div class="answer-box">
    <h2>Answer</h2>
    <div class="answer-content">{{ answer }}</div>
    <div class="meta">Generated in {{ elapsed }}s using {{ chunk_count }} chunks</div>
  </div>
  {% endif %}

  {% if chunks %}
  <h2 style="font-size:0.9rem; color:#666; margin-bottom:0.6rem;">Retrieved Chunks</h2>
  {% for c in chunks %}
  <div class="chunk">
    <span class="chunk-sim">{{ "%.3f"|format(c.similarity) }}</span>
    <div class="chunk-source">{{ c.file }} ({{ c.stype }}{% if c.vision_model %}, {{ c.vision_model }}{% endif %})</div>
    <div class="chunk-text">{{ c.text }}</div>
  </div>
  {% endfor %}
  {% endif %}
</div>
</body>
</html>
"""


def do_search(query, top_k=5, where_filter=None):
    embedding = embed_model.encode([query]).tolist()
    kwargs = {
        "query_embeddings": embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter
    return collection.query(**kwargs)


def ask_llm(context, question):
    response = ollama.chat(
        model=OLLAMA_LLM,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response["message"]["content"]


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    src_filter = ""
    content_type = ""
    top_k = 5
    no_llm = False
    answer = None
    chunks = []
    elapsed = None
    chunk_count = 0

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        src_filter = request.form.get("src_filter", "")
        content_type = request.form.get("content_type", "")
        try:
            top_k = int(request.form.get("top_k", 5))
        except (ValueError, TypeError):
            top_k = 5
        no_llm = bool(request.form.get("no_llm"))

        if query and collection is None:
            answer = "No index found. Run index.py first."
        elif query:
            t0 = time.time()

            # Build filter
            conditions = []
            if src_filter:
                conditions.append({"collection": src_filter})
            if content_type:
                conditions.append({"source_type": content_type})
            where_filter = None
            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}

            results = do_search(query, top_k=top_k, where_filter=where_filter)

            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({
                    "text": doc,
                    "file": meta.get("source_file", "?"),
                    "stype": meta.get("source_type", "?"),
                    "vision_model": meta.get("vision_model", ""),
                    "similarity": 1 - dist,
                })

            chunk_count = len(chunks)

            if not no_llm and chunks:
                context = "\n\n---\n\n".join(
                    f"[Source: {c['file']}]\n{c['text']}" for c in chunks
                )
                answer = ask_llm(context, query)

            elapsed = f"{time.time() - t0:.1f}"

    return render_template_string(
        HTML,
        query=query,
        src_filter=src_filter,
        content_type=content_type,
        top_k=top_k,
        no_llm=no_llm,
        answer=answer,
        chunks=chunks,
        elapsed=elapsed,
        chunk_count=chunk_count,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
