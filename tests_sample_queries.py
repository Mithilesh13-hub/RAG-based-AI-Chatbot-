"""Benchmark / calibration script (NOT collected by pytest; needs live credentials + ingested index).

  python tests_sample_queries.py              # run the 6 benchmark questions through the graph
  python tests_sample_queries.py --api        # ... through the running FastAPI backend
  python tests_sample_queries.py --calibrate  # retrieval-only score stats + suggested threshold
"""
import argparse
import sys

from src.config import REFUSAL_MESSAGE, get_settings

QUESTIONS = [
    ("What is Agentic AI according to the eBook?", True),
    ("How do AI agents differ from traditional automation systems?", True),
    ("What are the core components of an Agentic Architecture?", True),
    ("What role does memory play in Agentic AI workflows?", True),
    ("How do AI agents use tools to complete tasks?", True),
    ("Who won the 2022 FIFA World Cup?", False),  # out of scope: refusal expected
]


def ask(question: str, use_api: bool) -> dict:
    if use_api:
        import requests
        r = requests.post(f"{get_settings().api_base_url}/chat", json={"query": question}, timeout=120)
        r.raise_for_status()
        return r.json()
    from src.graph import run_graph
    return run_graph(question)


def run_benchmark(use_api: bool) -> int:
    failures = 0
    for q, in_scope in QUESTIONS:
        out = ask(q, use_api)
        refused = out["final_answer"].strip() == REFUSAL_MESSAGE
        scores = [c["relevance_score"] for c in out["retrieved_context"]]
        if not in_scope:
            verdict = "PASS" if refused else "FAIL (should refuse)"
        else:
            # Whether the eBook covers Q1-5 must be checked against the PDF itself.
            verdict = "REVIEW (refused - check the PDF / threshold)" if refused else "ANSWERED (verify vs PDF)"
        failures += verdict.startswith("FAIL")
        print(f"\nQ: {q}\n  {verdict}\n  overall={out['confidence_score']:.3f}  chunk scores={scores}")
        print(f"  citations={out['citations']}\n  answer={out['final_answer'][:300]}")
    return failures


def calibrate() -> None:
    from src.graph import create_vectorstore, normalize_score
    settings = get_settings()
    store = create_vectorstore(settings)
    top = {}
    for q, in_scope in QUESTIONS:
        results = store.similarity_search_with_score(q, k=settings.top_k)
        top[q] = (in_scope, max((normalize_score(s) for _, s in results), default=0.0))
        print(f"{'IN ' if in_scope else 'OUT'} top-score={top[q][1]:.3f}  {q}")
    ins = [s for ok, s in top.values() if ok]
    outs = [s for ok, s in top.values() if not ok]
    lo, hi = min(ins), max(outs)
    if lo > hi:
        print(f"\nSuggested RELEVANCE_THRESHOLD ~ {(lo + hi) / 2:.2f} (gap between {hi:.3f} and {lo:.3f}).")
    else:
        print("\nNo clean gap: in-scope and out-of-scope scores overlap. Add more benchmark queries "
              "and rely on the term-coverage check (MIN_TERM_COVERAGE).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    else:
        sys.exit(1 if run_benchmark(a.api) else 0)
