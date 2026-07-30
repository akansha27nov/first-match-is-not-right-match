import json
from openai import OpenAI
from config import OPENAI_API_KEY
from models import RetrievedChunk, RerankedChunk, LLMRelevanceScore

client = OpenAI(api_key=OPENAI_API_KEY)

TEST_QUERIES = [
    "What are the transparency requirements for high-risk AI systems?",
    "What record-keeping obligations apply to high-risk AI systems?",
    "How does the AI Act define an AI system?",
]


def score_chunk_relevance(query: str, chunk_text: str) -> float:
    prompt = f"""Rate how relevant this passage is to answering the query, on a 0.0-1.0 scale.
Return ONLY a JSON object: {{"score": <float>}}

Query: {query}
Passage: {chunk_text[:800]}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = LLMRelevanceScore.model_validate_json(raw)
    return parsed.score


def llm_rerank(query: str, chunks: list[RetrievedChunk], alpha: float = 0.5) -> list[RerankedChunk]:
    """alpha weights similarity vs LLM relevance: combined = alpha*sim + (1-alpha)*llm_score"""
    reranked = []
    for c in chunks:
        llm_score = score_chunk_relevance(query, c.text)
        combined = alpha * c.score + (1 - alpha) * llm_score
        reranked.append(
            RerankedChunk(**c.model_dump(), llm_score=llm_score, combined_score=combined)
        )

    return sorted(reranked, key=lambda c: c.combined_score, reverse=True)


if __name__ == "__main__":
    from retrieval import baseline_search

    for query in TEST_QUERIES:
        print(f"\n{'='*80}\nQUERY: {query}\n{'='*80}")
        hits = baseline_search(query, top_k=5)
        reranked = llm_rerank(query, hits)

        for h in reranked:
            print(f"[sim={h.score:.3f} llm={h.llm_score:.2f} combined={h.combined_score:.3f}] article={h.article}")
            print(f"  {h.text[:120]}...")
