import json
from rag_pipeline import run_pipeline, RAGAnswer

TEST_QUERIES = [
    "What are the transparency requirements for high-risk AI systems?",
    "What record-keeping obligations apply to high-risk AI systems?",
    "How does the AI Act define an AI system?",
]


def compare(query: str) -> dict:
    without = run_pipeline(query, use_reranking=False)
    with_rerank = run_pipeline(query, use_reranking=True)
    return {"query": query, "without_reranking": without, "with_reranking": with_rerank}


def run_eval():
    results = []
    for q in TEST_QUERIES:
        print(f"\n{'='*80}\nQUERY: {q}\n{'='*80}")
        result = compare(q)
        results.append(result)

        print("\n--- WITHOUT RERANKING ---")
        print(result["without_reranking"].answer)
        print("Sources:", [f"{e.source}/{e.section}" for e in result["without_reranking"].evidence])

        print("\n--- WITH RERANKING ---")
        print(result["with_reranking"].answer)
        print("Sources:", [f"{e.source}/{e.section}" for e in result["with_reranking"].evidence])

    # Save raw output for lab_proof.md
    with open("eval_results.json", "w") as f:
        json.dump(
            [
                {
                    "query": r["query"],
                    "without_reranking": r["without_reranking"].model_dump(),
                    "with_reranking": r["with_reranking"].model_dump(),
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print("\nSaved eval_results.json")


if __name__ == "__main__":
    run_eval()