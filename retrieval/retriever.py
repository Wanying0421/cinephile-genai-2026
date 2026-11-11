"""Public retrieval API.

Usage:
    from retrieval import retrieve
    results = retrieve("Who is Saito in Inception?", k=5)

Each result: {chunk_id, doc_id, title, text, score, dense_score, type, year, source_url}
- score:       ranking score of the chosen method (cosine for dense, RRF for hybrid)
- dense_score: cosine similarity between query and chunk, always computed.
               Use THIS one for abstention thresholds (RRF scores only reflect rank).
"""
import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .bm25 import BM25Index
from .config import (CANDIDATES, DEFAULT_CHUNK_SIZE, DEFAULT_METHOD, EMBEDDING_MODEL,
                     INDEX_DIR, QUERY_PREFIX, RRF_K, TOP_K)
from .fusion import rrf


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL)


class Retriever:
    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, method=DEFAULT_METHOD):
        if method not in ("dense", "hybrid"):
            raise ValueError(f"method must be 'dense' or 'hybrid', got {method!r}")
        self.chunk_size = chunk_size
        self.method = method

        with open(INDEX_DIR / f"chunks_{chunk_size}.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embeddings = np.load(INDEX_DIR / f"embeddings_{chunk_size}.npy")
        self.index = faiss.read_index(str(INDEX_DIR / f"faiss_{chunk_size}.index"))
        self.bm25 = BM25Index(self.chunks) if method == "hybrid" else None
        self.model = get_model()

    def encode_query(self, query):
        return self.model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

    def retrieve(self, query, k=TOP_K):
        query_vec = self.encode_query(query)
        n = min(CANDIDATES, len(self.chunks))

        # Dense search: FAISS returns (scores, indices), best first
        dense_scores, dense_ids = self.index.search(query_vec, n)
        dense_ranking = [int(i) for i in dense_ids[0]]

        if self.method == "dense":
            ranked = list(zip(dense_ranking, dense_scores[0].tolist()))[:k]
        else:
            bm25_ranking = [i for i, _ in self.bm25.search(query, n)]
            ranked = rrf([dense_ranking, bm25_ranking], k=RRF_K)[:k]

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "score": round(float(score), 4),
                # Cosine similarity, computed for every result (including
                # chunks found only by BM25)
                "dense_score": round(float(self.embeddings[idx] @ query_vec[0]), 4),
                "type": chunk["type"],
                "year": chunk["year"],
                "source_url": chunk["source_url"],
            })
        return results


_default_retriever = None


def retrieve(query, k=TOP_K):
    """Retrieve with the default configuration (set in config.py)."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever.retrieve(query, k)


if __name__ == "__main__":
    queries = [
        "Who is Saito in Inception?",
        # No title in the question: tests semantic matching
        "In which film does a daughter stay on Earth while her father travels through a wormhole?",
        # Completely unrelated: dense_score should be lower
        "What is the capital of France?",
    ]
    for method in ("dense", "hybrid"):
        retriever = Retriever(chunk_size=200, method=method)
        print(f"\n=========== method = {method} ===========")
        for q in queries:
            print(f"\nQuery: {q}")
            for r in retriever.retrieve(q, k=3):
                print(f"  score={r['score']:<8} dense_score={r['dense_score']:<8} {r['chunk_id']}")