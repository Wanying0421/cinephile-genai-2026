"""Retrieval layer package."""
import os

# This project uses PyTorch only. If TensorFlow/Keras happen to be installed
# in the environment, stop transformers from importing them (Keras 3 is
# incompatible with transformers and crashes the import).
os.environ.setdefault("USE_TF", "0")

def retrieve(query, k=5):
    """Shortcut so other modules can write: from retrieval import retrieve"""
    from .retriever import retrieve as _retrieve
    return _retrieve(query, k)