import json
import sys
from level0 import run, client, MODEL

JUDGE_PROMPT = """You are a strict grader. Compare the ANSWER to the REFERENCE.
Reply with exactly one word: CORRECT if the answer states the same key facts as the
reference, otherwise INCORRECT. An answer that refuses or says it does not know is INCORRECT.

QUESTION: {q}
REFERENCE: {ref}
ANSWER: {ans}"""

def judge(q, ref, ans):
    r = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, ans=ans)}],
    )
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
    return None  # multi_step: not scored at level 0

def main(path):
    with open(path, encoding="utf-8") as f:
        tests = [json.loads(l) for l in f if l.strip()]
    rows = []
    with open("results/level0_results.jsonl", "w", encoding="utf-8") as log:
        for t in tests:
            out = run(t["input_prompt"])
            out["test_id"], out["category"] = t["id"], t["category"]
            out["correct"] = score(t, out)
            log.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows.append(out)
            print(t["id"], t["category"], out["correct"])

    scored = [r for r in rows if r["correct"] is not None]
    absent = [r for r in rows if r["category"] == "rag_absence"]
    n = len(rows)
    print("\n=== Level 0 ===")
    print(f"Tests: {n} (scored: {len(scored)})")
    print(f"Accuracy: {sum(r['correct'] for r in scored)/len(scored):.0%}")
    if absent:
        print(f"Abstention: {sum(r['abstained'] for r in absent)/len(absent):.0%}")
    print(f"Tokens/request: {sum(r['prompt_tokens']+r['completion_tokens'] for r in rows)/n:.0f}")
    print(f"Avg latency: {sum(r['latency_s'] for r in rows)/n:.2f}s")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/tests.jsonl")