import json
import sys
import argparse
from level0 import run as run_level0, complete
from level1 import rag_answer as run_level1

JUDGE_PROMPT = """You are a strict grader. Compare the ANSWER to the REFERENCE.
Reply with exactly one word: CORRECT if the answer states the same key facts as the
reference, otherwise INCORRECT. An answer that refuses or says it does not know is INCORRECT.

QUESTION: {q}
REFERENCE: {ref}
ANSWER: {ans}"""

def judge(q, ref, ans):
    r = complete([
        {"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, ans=ans)},
    ])
    return (r.choices[0].message.content or "").strip().upper().startswith("CORRECT")

def score(test, out):
    cat = test["category"]
    if cat == "rag_absence":
        return out["abstained"]
    if cat == "ambiguous":
        return out["abstained"] or "?" in out["answer"]
    if cat == "rag_document":
        if out["abstained"]:
            return False
        return judge(test["input_prompt"], test["expected_output"], out["answer"])
    return None  # multi_step

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cinephile GenAI RAG pipelines.")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1], help="Pipeline level to evaluate (0 or 1)")
    parser.add_argument("--path", type=str, default="tests/tests.jsonl", help="Path to test dataset")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        tests = [json.loads(l) for l in f if l.strip()]

    output_file = f"results/level{args.level}_results.jsonl"
    rows = []

    with open(output_file, "w", encoding="utf-8") as log:
        for t in tests:
            if args.level == 0:
                out = run_level0(t["input_prompt"])
            else:
                res = run_level1(t["input_prompt"])
                # Normalize Level 1 output keys to match evaluation expectations
                out = {
                    "level": 1,
                    "question": t["input_prompt"],
                    "answer": res["answer"],
                    "abstained": res["abstained"],
                    "prompt_tokens": res["usage"].get("prompt_tokens", 0),
                    "completion_tokens": res["usage"].get("completion_tokens", 0),
                    "latency_s": res["latency"],
                    "citations": res["citations"]
                }

            out["test_id"], out["category"] = t["id"], t["category"]
            out["correct"] = score(t, out)
            log.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows.append(out)
            print(f"Test {t['id']} [{t['category']}]: Correct={out['correct']}, Abstained={out['abstained']}")

    scored = [r for r in rows if r["correct"] is not None]
    n = len(rows)
    print(f"\n=== Level {args.level} Evaluation ===")
    print(f"Tests: {n} (scored: {len(scored)})")
    if scored:
        print(f"Accuracy: {sum(r['correct'] for r in scored)/len(scored):.0%}")
    print(f"Avg latency: {sum(r['latency_s'] for r in rows)/n:.2f}s")

if __name__ == "__main__":
    main()