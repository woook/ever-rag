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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
