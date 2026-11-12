"""Compare chunk sizes x retrieval methods on two test sets.

- original:    rag_document cases from tests/tests.jsonl (mostly name the film)
- paraphrased: data/retrieval_tests.jsonl (no titles, paraphrased, hard negatives)

Usage (from the project root): python -m retrieval.eval_retrieval
Outputs:
  results/retrieval_comparison.md   summary table (for the report)
  results/retrieval_details.jsonl   per-question results (for failure analysis)
"""
import json
import time

from .config import CHUNK_SIZES, RESULTS_DIR, ROOT, TESTS_PATH, TOP_K
from .retriever import Retriever

METHODS = ["dense", "hybrid"]
PARAPHRASED_PATH = ROOT / "data" / "retrieval_tests.jsonl"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_test_sets():
    """Return {set_name: [{"id", "question", "gold"}]}."""
    original = [
        {"id": f"T{c['id']}", "question": c["input_prompt"], "gold": c["relevant_doc_ids"]}
        for c in read_jsonl(TESTS_PATH)
        if c.get("category") == "rag_document" and c.get("relevant_doc_ids")
    ]
    paraphrased = [
        {"id": c["id"], "question": c["question"], "gold": c["relevant_doc_ids"]}
        for c in read_jsonl(PARAPHRASED_PATH)
    ]
    return {"original": original, "paraphrased": paraphrased}


def first_relevant_rank(results, gold_doc_ids):
    """Rank (1-based) of the first retrieved chunk from a gold document, else None.
    Evaluation is at document level: any chunk of the right film counts."""
    for rank, r in enumerate(results, start=1):
        if r["doc_id"] in gold_doc_ids:
            return rank
    return None


def evaluate(retriever, cases, set_name, k=TOP_K):
    hits, reciprocal_ranks, context_words, latency = 0, 0.0, 0, 0.0
    details = []
    for case in cases:
        gold = set(case["gold"])
        start = time.perf_counter()
        results = retriever.retrieve(case["question"], k=k)
        latency += time.perf_counter() - start

        rank = first_relevant_rank(results, gold)
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / rank
        context_words += sum(len(r["text"].split()) for r in results)

        details.append({
            "test_set": set_name,
            "chunk_size": retriever.chunk_size,
            "method": retriever.method,
            "test_id": case["id"],
            "question": case["question"],
            "gold": sorted(gold),
            "retrieved": [r["chunk_id"] for r in results],
            "first_relevant_rank": rank,
            "top1_dense_score": results[0]["dense_score"],
        })

    n = len(cases)
    metrics = {
        "recall@k": hits / n,
        "mrr@k": reciprocal_ranks / n,
        "context_words": context_words / n,   # words sent to the LLM per query
        "latency_ms": 1000 * latency / n,
    }
    return metrics, details


def main():
    test_sets = load_test_sets()
    for name, cases in test_sets.items():
        print(f"{name}: {len(cases)} cases")
    print(f"k={TOP_K}\n")
    RESULTS_DIR.mkdir(exist_ok=True)

    rows, all_details = [], []
    for size in CHUNK_SIZES:
        for method in METHODS:
            retriever = Retriever(chunk_size=size, method=method)
            retriever.retrieve("warm-up query", k=1)  # exclude loading cost from latency
            for set_name, cases in test_sets.items():
                metrics, details = evaluate(retriever, cases, set_name)
                rows.append((set_name, size, method, metrics))
                all_details.extend(details)
                print(f"{set_name:<12} size={size:<4} {method:<7} "
                      f"Recall@{TOP_K}={metrics['recall@k']:.3f}  "
                      f"MRR@{TOP_K}={metrics['mrr@k']:.3f}  "
                      f"words={metrics['context_words']:.0f}  "
                      f"latency={metrics['latency_ms']:.0f}ms")

    sizes = ", ".join(f"{name}={len(cases)}" for name, cases in test_sets.items())
    lines = [
        f"# Retrieval comparison (k={TOP_K}; cases: {sizes})",
        "",
        f"| Test set | Chunk size (words) | Method | Recall@{TOP_K} | MRR@{TOP_K} "
        "| Context words / query | Latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for set_name in test_sets:
        for name, size, method, m in rows:
            if name == set_name:
                lines.append(f"| {name} | {size} | {method} | {m['recall@k']:.3f} "
                             f"| {m['mrr@k']:.3f} | {m['context_words']:.0f} "
                             f"| {m['latency_ms']:.0f} |")
    lines += ["", "Context tokens are roughly 1.3 x context words."]
    (RESULTS_DIR / "retrieval_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(RESULTS_DIR / "retrieval_details.jsonl", "w", encoding="utf-8") as f:
        for d in all_details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nSaved {RESULTS_DIR / 'retrieval_comparison.md'}")
    print(f"Saved {RESULTS_DIR / 'retrieval_details.jsonl'}")


if __name__ == "__main__":
    main()