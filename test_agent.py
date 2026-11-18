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
    res2 = agent_answer("Add Inception to my watchlist")
    print(json.dumps({
        "needs_confirmation": res2["needs_confirmation"],
        "pending_action": res2["pending_action"],
        "stop_reason": res2["stop_reason"]
    }, indent=2))

    print("\n=== 3. RESUMPTION / SUCCESS EXAMPLE (Action Approved & Re-submitted) ===")
    # Simulate user approving the action by adding its signature to the state
    approved_signature = res2["pending_action"]["signature"]
    resumed_state = {
        "approved_actions": [approved_signature]
    }
    
    # Re-running the same request with the approved state
    res3 = agent_answer("Add Inception to my watchlist", state=resumed_state)
    print(json.dumps({
        "answer": res3["answer"],
        "needs_confirmation": res3["needs_confirmation"],
        "state": res3["state"],
        "stop_reason": res3["stop_reason"]
    }, indent=2))

    print("\n=== 4. CANCELLATION EXAMPLE (User Rejects Action) ===")
    res4 = agent_answer("Add Inception to my watchlist", user_decision="cancel")
    print(json.dumps({
        "answer": res4["answer"],
        "needs_confirmation": res4["needs_confirmation"],
        "state": res4["state"],
        "stop_reason": res4["stop_reason"]
    }, indent=2))

if __name__ == "__main__":
    run_handoff_tests()