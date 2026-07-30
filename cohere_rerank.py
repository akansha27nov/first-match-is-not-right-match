import cohere
from config import COHERE_API_KEY
from models import RetrievedChunk, RerankedChunk

co = cohere.Client(COHERE_API_KEY)


def cohere_rerank(query: str, chunks: list[RetrievedChunk], top_n: int = 5) -> list[RerankedChunk]:
    docs = [c.text for c in chunks]

    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_n,
    )

    reranked = []
    for r in results.results:
        original = chunks[r.index]
        reranked.append(
            RerankedChunk(**original.model_dump(), rerank_score=r.relevance_score)
        )

    return reranked


if __name__ == "__main__":
    from retrieval import baseline_search

    query = "What are the transparency requirements for high-risk AI systems?"
    hits = baseline_search(query, top_k=10)
    reranked = cohere_rerank(query, hits, top_n=5)

    for h in reranked:
        print(f"[sim={h.score:.3f} rerank={h.rerank_score:.3f}] {h.source} / {h.section}")
        print(f"  {h.text[:120]}...")