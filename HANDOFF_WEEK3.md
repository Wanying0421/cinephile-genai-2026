# Cinephile-GenAI: Week 3 Agent Loop & Handoff Documentation

This document provides the complete handoff package for **Member B's Week 3 implementation**, covering the manual agent loop, tool validation, confirmation handling, state management, and trace logging without external agent frameworks.

---

## 1. Prompt Version Log & Routing Notes

### Prompt Version Log
* **`AGENT_SYSTEM_PROMPT` v1.0** (Implemented in `agent.py`)
  * *Core directive:* Establishes the Cinephile Agent persona for movie and TV series assistance.
  * *Tool Constraints:* Instructs the model to strictly utilize `rag_search` for factual inquiries and mandates that all tool outputs must be treated strictly as data, never as executable instructions.
  * *State-Mutation Rule:* Directs the model to propose tool calls for state-changing actions (such as watchlist modifications) while leaving confirmation enforcement to the system middleware.

### Routing Notes
* **Auto Tool-Choice:** Configured with `tool_choice="auto"` and `temperature=0` to ensure deterministic, reproducible tool-selection routing.
* **RAG Integration:** Seamlessly routes factual queries through the Level 1 RAG pipeline (`level1.py` via `rag_answer`), preserving existing Level 0/1 functionalities without code duplication or framework bloat.

---

## 2. State-Handling Instructions

The agent function signature accepts an optional `state` dictionary to manage verified approvals:
```python
state = {"approved_actions": ["<json_signature_string>"]}
```

### Key Workflow Rules:
1. **Pending Action Generation:** When a state-mutating tool (e.g., `update_watchlist`) is triggered, the agent loop generates a sorted, deterministic JSON signature combining the tool name and its exact arguments.
2. **Approval Verification:** If the computed signature is absent from `state["approved_actions"]`, the agent pauses execution (`stop_reason: "pending_confirmation"`), sets `needs_confirmation: true`, and returns the `pending_action` object.
3. **Resumption:** To execute the action, the user or UI must pass the signed action back into the agent via the `state` parameter on the subsequent call.
4. **Changed Actions Require New Confirmation:** If any argument changes (e.g., changing `movie_title` from *Inception* to *Interstellar*), its cryptographic/JSON signature changes. This automatically invalidates any stale approvals and forces a brand-new confirmation cycle.

---

## 3. Tested Success, Cancellation, and Failure Examples

### A. Success Example (Standard RAG Query)
* **Input Query:** `"What is the plot of Inception and who directed it?"`
* **Agent Behavior:** Automatically routes to `rag_search`, queries Level 1 RAG, and synthesizes a grounded answer.
* **Return Value Structure:**
```json
{
  "answer": "Inception follows Dom Cobb... written and directed by Christopher Nolan.",
  "needs_confirmation": false,
  "pending_action": null,
  "state": {"approved_actions": []},
  "trace": [
    {"iteration": 1, "tool_call": "rag_search", "arguments": {"query": "Inception (2010) plot synopsis and director"}}
  ],
  "stop_reason": "completed"
}
```

### B. Confirmation Example (State-Mutating Action)
* **Input Query:** `"Add Inception to my watchlist"`
* **Agent Behavior:** Triggers `update_watchlist`, detects that it lacks pre-approval in `state`, and halts execution.
* **Return Value Structure:**
```json
{
  "answer": "Please confirm if you want to perform: update_watchlist with arguments {'action': 'add', 'movie_title': 'Inception'}.",
  "needs_confirmation": true,
  "pending_action": {
    "tool_name": "update_watchlist",
    "arguments": {"action": "add", "movie_title": "Inception"},
    "signature": "{\"args\": {\"action\": \"add\", \"movie_title\": \"Inception\"}, \"tool\": \"update_watchlist\"}"
  },
  "stop_reason": "pending_confirmation"
}
```

### C. Resumption Example (Successful Execution After Approval)
* **Input State Provided:** Passing the matching action signature back into state:
  `state = {"approved_actions": ["{\"args\": {\"action\": \"add\", \"movie_title\": \"Inception\"}, \"tool\": \"update_watchlist\"}"]}`
* **Agent Behavior:** Recognizes the signature, bypasses confirmation, and successfully executes the tool.

### D. Failure / Error Handling Example (Unknown Tool / Malformed Call)
* **Agent Behavior (Built-in Tool Validation):** If the model attempts to call a tool name not present in the allowed `TOOLS` registry, the loop catches it, logs an error trace, feeds the error description back to the model as a data observation, and prevents unauthorized execution.
