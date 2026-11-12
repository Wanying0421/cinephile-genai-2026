"""Reciprocal Rank Fusion (RRF), written by hand without any framework."""


def rrf(ranked_lists, k=60):
    """Fuse several ranked lists of chunk indices into one ranking.

    ranked_lists: list of lists, each ordered best-first, e.g. [dense, bm25]
    k: smoothing constant; 60 is the value from the original RRF paper

    For each chunk: score = sum over lists of 1 / (k + rank), rank starting at 1.
    A chunk ranked high in several lists gets the highest fused score.
    Returns [(chunk_index, rrf_score)] sorted best-first.
    """
    scores = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    # Hand-checkable example
    dense = [3, 1, 7]
    bm25 = [3, 9, 1]
    for idx, score in rrf([dense, bm25]):
        print(idx, round(score, 4))