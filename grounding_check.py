import re
from rag_pipeline import run_pipeline

TEST_QUERIES = [
    "What are the transparency requirements for high-risk AI systems?",
    "What record-keeping obligations apply to high-risk AI systems?",
    "How does the AI Act define an AI system?",
]


def check_grounding(query: str):
    result = run_pipeline(query, use_reranking=True)
    cited_sections = set(re.findall(r'page_\d+_chunk_\d+', result.answer))
    retrieved_sections = {e.section for e in result.evidence}

    unused = retrieved_sections - cited_sections
    print(f"Query: {query}")
    print(f"Retrieved: {len(retrieved_sections)} chunks")
    print(f"Actually cited in answer: {len(cited_sections)} chunks")
    print(f"Retrieved but NOT cited: {unused}")
    print()
    return {"query": query, "retrieved": retrieved_sections, "cited": cited_sections, "unused": unused}


if __name__ == "__main__":
    results = [check_grounding(q) for q in TEST_QUERIES]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        pct_unused = len(r["unused"]) / len(r["retrieved"]) * 100 if r["retrieved"] else 0
        print(f"{r['query'][:50]}... — {len(r['unused'])}/{len(r['retrieved'])} unused ({pct_unused:.0f}%)")
