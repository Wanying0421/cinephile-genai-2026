import json
from agent import agent_answer

def run_handoff_tests():
    print("=== 1. SUCCESS EXAMPLE (Standard RAG Query) ===")
    res1 = agent_answer("What is the plot of Inception and who directed it?")
    print(json.dumps({
        "answer": res1["answer"],
        "stop_reason": res1["stop_reason"],
        "latency": f"{res1['latency']:.2f}s"
    }, indent=2))

    print("\n=== 2. PENDING CONFIRMATION EXAMPLE (State-Mutating Action) ===")
    res2 = agent_answer("Add movie ID 27205 to my watchlist")
    print(json.dumps({
        "needs_confirmation": res2["needs_confirmation"],
        "pending_action": res2["pending_action"],
        "stop_reason": res2["stop_reason"]
    }, indent=2))

    print("\n=== 3. RESUMPTION / CONFIRMATION EXECUTION ===")
    # Pass user_decision="confirm" with the state containing the pending action
    res3 = agent_answer("", state=res2["state"], user_decision="confirm")
    print(json.dumps({
        "answer": res3["answer"],
        "state": res3["state"],
        "stop_reason": res3["stop_reason"]
    }, indent=2))

    print("\n=== 4. CANCELLATION EXAMPLE ===")
    res_prop = agent_answer("Add movie ID 27205 to my watchlist")
    res4 = agent_answer("", state=res_prop["state"], user_decision="cancel")
    print(json.dumps({
        "answer": res4["answer"],
        "state": res4["state"],
        "stop_reason": res4["stop_reason"]
    }, indent=2))

if __name__ == "__main__":
    run_handoff_tests()