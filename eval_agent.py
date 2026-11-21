"""Level 2 evaluation for the Cinephile agent.

Runs each agent task several times, each run in a fresh temporary SQLite
watchlist, simulates the user's confirm/cancel decision, and checks the
*actual* outcome by code: final watchlist contents, tool calls, tool results,
confirmation behaviour. Reports task success, pass^k, tokens and latency, and
saves one JSONL trace line per run.

Usage:
    python eval_agent.py                      # all tasks, 3 runs each
    python eval_agent.py --runs 1 --only L2-01 L2-02
    python eval.py --level 2                  # same, through eval.py
"""
import argparse
import copy
import json
import os
import shutil
import tempfile
from functools import partial

import agent
import tools.tmdb as tmdb
from tools import watchlist as wl

REQUIRED_MODEL = "gpt-6-luna"


def load_tasks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def simulated_tmdb_error(*args, **kwargs):
    return {"ok": False, "data": None,
            "error": {"code": "tmdb_unavailable",
                      "message": "Simulated TMDB timeout (evaluation)"}}


class Sandbox:
    """Fresh temporary watchlist for one run, plus optional simulated API failure.

    The agent's watchlist functions are redirected to a temporary database, so the
    real data/watchlist.sqlite3 is never touched. Everything is restored on exit.
    """

    def __init__(self, initial_watchlist=(), simulate=None):
        self.initial = list(initial_watchlist)
        self.simulate = simulate
        self.saved = []

    def _patch(self, module, name, value):
        if hasattr(module, name):
            self.saved.append((module, name, getattr(module, name)))
            setattr(module, name, value)

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="cinephile_eval_")
        self.db = os.path.join(self.dir, "watchlist.sqlite3")
        # Seed the initial watchlist through A's real prepare + confirm path.
        for movie_id in self.initial:
            p = wl.update_watchlist("add", movie_id, session_id=agent.SESSION_ID, db_path=self.db)
            if not p["ok"] or not p["data"].get("request_id"):
                raise RuntimeError(f"Could not seed movie {movie_id}: {p}")
            wl.confirm_watchlist_action(p["data"]["request_id"], approved=True,
                                        session_id=agent.SESSION_ID, db_path=self.db)
        self._patch(agent, "update_watchlist", partial(wl.update_watchlist, db_path=self.db))
        self._patch(agent, "confirm_watchlist_action",
                    partial(wl.confirm_watchlist_action, db_path=self.db))
        if self.simulate == "tmdb_error":
            for module in (agent, tmdb, wl):
                for name in ("search_movies", "get_movie_details"):
                    self._patch(module, name, simulated_tmdb_error)
        return self

    def watchlist_ids(self):
        r = wl.list_watchlist(db_path=self.db)
        return sorted(m["movie_id"] for m in r["data"]["movies"])

    def __exit__(self, *exc):
        for module, name, value in reversed(self.saved):
            setattr(module, name, value)
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def tool_calls(trace):
    return [s["tool_call"] for s in trace if "tool_call" in s]


def tool_errors(trace):
    errors = []
    for s in trace:
        if "error" in s:  # rejected by the agent's validation
            errors.append(s["error"])
        res = s.get("tool_result")
        if isinstance(res, dict) and res.get("ok") is False:
            errors.append(res.get("error"))
    return errors


def run_once(task):
    """Run one task in a fresh sandbox; return a trace record with checks."""
    rec = {"task_id": task["id"], "category": task["category"],
           "source_test_id": task.get("source_test_id"), "calls": [],
           "error": None}
    tools_used, errors, pendings = [], [], []
    tokens, latency, user_requests = 0, 0.0, 0
    final_answer = ""

    with Sandbox(task.get("initial_watchlist", []), task.get("simulate")) as sb:
        rec["initial_watchlist"] = sb.watchlist_ids()
        state = None
        try:
            for turn in task["turns"]:
                r = agent.agent_answer(turn["user"], state=state)
                user_requests += 1
                calls = [("user", turn["user"], copy.deepcopy(r))]
                state = r["state"]
                if r["needs_confirmation"]:
                    pendings.append(r["pending_action"])
                    decision = turn.get("decision")
                    if decision in ("confirm", "cancel"):
                        r2 = agent.agent_answer("", state=state, user_decision=decision)
                        calls.append((decision, "", copy.deepcopy(r2)))
                        state = r2["state"]
                for kind, text, res in calls:
                    tools_used += tool_calls(res.get("trace", []))
                    errors += tool_errors(res.get("trace", []))
                    tokens += res.get("usage", {}).get("total_tokens", 0)
                    latency += res.get("latency", 0.0)
                    final_answer = res.get("answer") or ""
                    rec["calls"].append({"kind": kind, "input": text, "result": res})
        except Exception as e:  # an agent crash is a failed run, not a crashed eval
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["final_watchlist"] = sb.watchlist_ids()

    final = rec["final_watchlist"]
    checks = {"no_crash": rec["error"] is None}
    if "expect_confirmation" in task:
        checks["confirmation"] = bool(pendings) == task["expect_confirmation"]
    if "expected_pending" in task:
        exp = task["expected_pending"]
        args = (pendings[-1] or {}).get("arguments", {}) if pendings else {}
        checks["pending_action"] = all(args.get(k) == v for k, v in exp.items())
    if "expected_watchlist" in task:
        checks["final_watchlist"] = final == sorted(task["expected_watchlist"])
    if "expected_watchlist_size" in task:
        checks["final_watchlist_size"] = len(final) == task["expected_watchlist_size"]
    if "expected_tools" in task:
        checks["expected_tools"] = all(t in tools_used for t in task["expected_tools"])
    if "forbidden_tools" in task:
        checks["forbidden_tools"] = not any(t in tools_used for t in task["forbidden_tools"])
    if task.get("expect_tool_error"):
        checks["tool_error_seen"] = bool(errors)
    if "answer_contains" in task:
        low = final_answer.lower()
        checks["answer_contains"] = any(s.lower() in low for s in task["answer_contains"])
    if task.get("expect_question"):
        checks["asks_question"] = "?" in final_answer

    rec.update({
        "success": all(checks.values()),
        "checks": checks,
        "tools_used": tools_used,
        "tool_errors": errors,
        "confirmation_requests": len(pendings),
        "final_answer": final_answer,
        "user_requests": user_requests,
        "total_tokens": tokens,
        "latency_s": round(latency, 3),
    })
    return rec


def summarize(rows, runs):
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    n_runs = len(rows)
    n_ok = sum(r["success"] for r in rows)
    passk = {t: all(x["success"] for x in rs) and len(rs) == runs for t, rs in by_task.items()}
    requests = sum(r["user_requests"] for r in rows) or 1
    summary = {
        "model": agent.MODEL,
        "tasks": len(by_task),
        "runs_per_task": runs,
        "total_runs": n_runs,
        "task_success_rate": n_ok / n_runs if n_runs else 0.0,
        f"pass^{runs}": sum(passk.values()) / len(by_task) if by_task else 0.0,
        "tokens_per_request": sum(r["total_tokens"] for r in rows) / requests,
        "latency_per_request_s": sum(r["latency_s"] for r in rows) / requests,
        "per_task": {t: {"category": rs[0]["category"],
                         "successes": sum(x["success"] for x in rs),
                         "runs": len(rs),
                         f"pass^{runs}": passk[t]} for t, rs in by_task.items()},
    }
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Level 2 (agent) evaluation with pass^k.")
    p.add_argument("--path", default="tests/agent_tests.jsonl")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--only", nargs="*", help="task ids to run, e.g. L2-01 L2-02")
    p.add_argument("--out", default="results/level2_traces.jsonl")
    args = p.parse_args(argv)

    if agent.MODEL != REQUIRED_MODEL:
        raise SystemExit(f"LLM_MODEL must be {REQUIRED_MODEL}, got {agent.MODEL}")

    tasks = load_tasks(args.path)
    if args.only:
        tasks = [t for t in tasks if t["id"] in args.only]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    with open(args.out, "w", encoding="utf-8") as log:
        for task in tasks:
            for k in range(1, args.runs + 1):
                rec = run_once(task)
                rec["run"] = k
                log.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                log.flush()
                rows.append(rec)
                failed = [c for c, ok in rec["checks"].items() if not ok]
                status = "PASS" if rec["success"] else f"FAIL {failed}"
                print(f"{task['id']} [{task['category']}] run {k}/{args.runs}: {status}"
                      f"  tools={rec['tools_used']} watchlist={rec['final_watchlist']}"
                      + (f"  error={rec['error']}" if rec["error"] else ""))

    s = summarize(rows, args.runs)
    summary_path = os.path.join(os.path.dirname(args.out) or ".", "level2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)

    print(f"\n=== Level 2 Evaluation ({s['model']}) ===")
    print(f"Tasks: {s['tasks']} x {args.runs} runs = {s['total_runs']} runs")
    print(f"Task success (all runs): {s['task_success_rate']:.0%}")
    print(f"pass^{args.runs}: {s[f'pass^{args.runs}']:.0%}")
    print(f"Tokens / request: {s['tokens_per_request']:.0f}")
    print(f"Latency / request: {s['latency_per_request_s']:.2f}s")
    print(f"Traces: {args.out}\nSummary: {summary_path}")
    return s


if __name__ == "__main__":
    main()
