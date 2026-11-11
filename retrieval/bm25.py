"""Keyword search with BM25 (Okapi)."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\w+")


def tokenize(text):
    """Lowercase and split into word tokens (letters/digits, accents kept).
    The SAME function must be used for chunks and for queries."""
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks):
        # Include the title, mirroring what the dense index embeds
        tokenized = [tokenize(f"{c['title']} {c['text']}") for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, n):
        """Return the top-n results as [(chunk_index, bm25_score)], best first."""
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top]


if __name__ == "__main__":
    # Quick sanity check on the 200-word chunks
    import json
    from .config import INDEX_DIR

    with open(INDEX_DIR / "chunks_200.json", encoding="utf-8") as f:
        chunks = json.load(f)
    index = BM25Index(chunks)

    for query in ["Who is Saito in Inception?", "In which town does the child disappear in Dark?"]:
        print(f"\nQuery: {query}")
        for i, score in index.search(query, 3):
            print(f"  {score:6.2f}  {chunks[i]['chunk_id']}")