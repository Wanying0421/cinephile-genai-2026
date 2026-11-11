import argparse
import json
import math
from pathlib import Path


def load_cases(path):
    with path.open(encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def collect_scores(path, retrieve, retrieval_cases=None):
    cases = load_cases(path)
    if retrieval_cases is not None:
        cases.extend(
            {"id": case["id"], "category": "rag_document", "input_prompt": case["question"]}
            for case in load_cases(retrieval_cases)
        )

    scores = {"rag_document": [], "rag_absence": []}
    seen = set()
    for case in cases:
        category = case["category"]
        if category not in scores:
            continue
        query = case["input_prompt"]
        if (category, query) in seen:
            continue
        seen.add((category, query))

        passages = retrieve(query, k=5)
        top_dense_score = float(passages[0]["dense_score"]) if passages else None
        if top_dense_score is not None and not math.isfinite(top_dense_score):
            raise ValueError(f"Test {case['id']} returned a non-finite dense_score.")
        scores[category].append(top_dense_score)

    if not all(scores.values()):
        raise ValueError("Calibration requires both answerable and out-of-corpus cases.")
    return scores["rag_document"], scores["rag_absence"]


def compare_thresholds(answerable, absent):
    def passes(score, cutoff):
        return score is not None and (cutoff is None or score >= cutoff)

    observed_scores = {score for score in answerable + absent if score is not None}
    rows = []
    for cutoff in [None, *sorted(observed_scores)]:
        answerable_passed = sum(passes(score, cutoff) for score in answerable)
        absent_blocked = sum(not passes(score, cutoff) for score in absent)
        rows.append((cutoff, answerable_passed, absent_blocked))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare Level 1 retrieval score cutoffs.")
    parser.add_argument("cases", nargs="?", type=Path, default=Path("tests/tests.jsonl"))
    parser.add_argument("--retrieval-cases", type=Path, help="Additional answerable retrieval questions")
    args = parser.parse_args()

    from retrieval import retrieve

    answerable, absent = collect_scores(args.cases, retrieve, args.retrieval_cases)
    rows = compare_thresholds(answerable, absent)
    _, baseline_passed, baseline_blocked = rows[0]

    print("Score used: dense_score of the first returned passage.\n")
    print("| Minimum dense score | Answerable passed | Out-of-corpus blocked |")
    print("|---|---:|---:|")
    for cutoff, passed, blocked in rows:
        label = "Off" if cutoff is None else str(cutoff)
        print(f"| {label} | {passed}/{len(answerable)} | {blocked}/{len(absent)} |")

    eligible = [row for row in rows if row[1] == baseline_passed]
    cutoff, _, blocked = max(eligible, key=lambda row: row[2])
    if baseline_passed and cutoff is not None and blocked > baseline_blocked:
        print(f"\nSuggested setting: RAG_MIN_SCORE={cutoff}")
    else:
        print("\nSuggested setting: RAG_MIN_SCORE= (leave the cutoff unset)")


if __name__ == "__main__":
    main()
