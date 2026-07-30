import re
from rag_pipeline import run_pipeline

def check_grounding(query: str):
    result = run_pipeline(query, use_reranking=True)
    cited_sections = set(re.findall(r'page_\d+_chunk_\d+', result.answer))
    retrieved_sections = {e.section for e in result.evidence}

    unused = retrieved_sections - cited_sections
    print(f"Query: {query}")
    print(f"Retrieved: {len(retrieved_sections)} chunks")
    print(f"Actually cited in answer: {len(cited_sections)} chunks")
    print(f"Retrieved but NOT cited: {unused}")
    return {"retrieved": retrieved_sections, "cited": cited_sections, "unused": unused}

if __name__ == "__main__":
    check_grounding("What are the transparency requirements for high-risk AI systems?")