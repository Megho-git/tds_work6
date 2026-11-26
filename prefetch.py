"""Run at build time so the model is downloaded once, not on every cold start."""
import os
from fastembed import TextEmbedding

TextEmbedding(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
              cache_dir=os.getenv("MODEL_CACHE", "model_cache"))
print("model cached")
