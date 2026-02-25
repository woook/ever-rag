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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
