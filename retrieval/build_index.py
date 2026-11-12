"""Build chunk files, embeddings and FAISS indexes for every chunk size.

Usage (from the project root): python -m retrieval.build_index
Outputs in index/: chunks_{size}.json, embeddings_{size}.npy, faiss_{size}.index
"""
import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import load_documents, chunk_corpus
from .config import CHUNK_SIZES, EMBEDDING_MODEL, INDEX_DIR, PASSAGE_PREFIX


def passage_text(chunk):
    """Text that gets embedded for a chunk.

    The title is prepended because chunks from the middle of a plot often
    never mention the film's name (e.g. a chunk about Saito alone).
    """
    return f"{PASSAGE_PREFIX}{chunk['title']}. {chunk['text']}"


def build(size, docs, model):
    chunks = chunk_corpus(docs, size)
    texts = [passage_text(c) for c in chunks]

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,   # unit vectors: inner product = cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    elapsed = time.time() - start

    index = faiss.IndexFlatIP(embeddings.shape[1])  # exact inner-product search
    index.add(embeddings)

    with open(INDEX_DIR / f"chunks_{size}.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    np.save(INDEX_DIR / f"embeddings_{size}.npy", embeddings)
    faiss.write_index(index, str(INDEX_DIR / f"faiss_{size}.index"))

    print(f"size={size}: {len(chunks)} chunks, dim={embeddings.shape[1]}, "
          f"encoded in {elapsed:.1f}s")


def main():
    INDEX_DIR.mkdir(exist_ok=True)
    docs = load_documents()
    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    for size in CHUNK_SIZES:
        build(size, docs, model)
    print(f"\nDone. Index files written to {INDEX_DIR}")


if __name__ == "__main__":
    main()