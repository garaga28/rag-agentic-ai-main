"""Run the benchmark queries and save the JSON responses.

Direct (no server):   python tests_sample_queries.py
Against the API:      python tests_sample_queries.py --api http://127.0.0.1:8000
"""

import argparse
import json
from pathlib import Path

import requests

SAMPLE_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What role does memory play in Agentic AI workflows?",
    # Out-of-scope checks: the bot must refuse these.
    "What is the capital of France?",
    "Who won the 2022 FIFA World Cup?",
]

OUTPUT_FILE = Path(__file__).parent / "sample_outputs.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="base URL of a running FastAPI server")
    args = parser.parse_args()

    if args.api:
        def run(q: str) -> dict:
            resp = requests.post(f"{args.api.rstrip('/')}/chat", json={"query": q}, timeout=120)
            resp.raise_for_status()
            return resp.json()
    else:
        from src.graph import ask, build_rag_graph
        graph = build_rag_graph()

        def run(q: str) -> dict:
            return ask(graph, q)

    results = []
    for i, query in enumerate(SAMPLE_QUERIES, start=1):
        payload = run(query)
        results.append(payload)
        print(f"\n=== Query {i}: {query}")
        print(f"confidence_score: {payload['confidence_score']}")
        print(f"final_answer: {payload['final_answer']}")
        print(f"retrieved_context_chunks: {len(payload['retrieved_context_chunks'])}")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved full responses to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
