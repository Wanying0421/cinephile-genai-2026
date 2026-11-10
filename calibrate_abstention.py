import argparse
import json
import math
from pathlib import Path


def collect_scores(path, retrieve):
    scores = {"rag_document": [], "rag_absence": []}
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if not line.strip():
                continue
            case = json.loads(line)
            category = case["category"]
            if category not in scores:
                continue

            passages = retrieve(case["input_prompt"], k=5)
            best_score = max((float(passage["score"]) for passage in passages), default=None)
            if best_score is not None and not math.isfinite(best_score):
                raise ValueError(f"Test {case['id']} returned a non-finite retrieval score.")
            scores[category].append(best_score)

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
    args = parser.parse_args()

    from retrieval import retrieve

    answerable, absent = collect_scores(args.cases, retrieve)
    rows = compare_thresholds(answerable, absent)
    _, baseline_passed, baseline_blocked = rows[0]

    print("| Minimum score | Answerable passed | Out-of-corpus blocked |")
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
