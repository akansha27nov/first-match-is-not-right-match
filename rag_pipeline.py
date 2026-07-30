from openai import OpenAI
from config import OPENAI_API_KEY
from models import RetrievedChunk, RerankedChunk
from retrieval import baseline_search
from cohere_rerank import cohere_rerank
from pydantic import BaseModel

client = OpenAI(api_key=OPENAI_API_KEY)


class RAGAnswer(BaseModel):
    query: str
    answer: str
    evidence: list[RerankedChunk]
    used_reranking: bool


def build_context(chunks: list[RetrievedChunk] | list[RerankedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.source} / {c.section}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, context: str) -> str:
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't
contain the answer, say so explicitly. Cite which source/section supports each claim.

Context:
{context}

Question: {query}

Answer:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def run_pipeline(query: str, use_reranking: bool = True, top_k: int = 10, top_n: int = 5) -> RAGAnswer:
    candidates = baseline_search(query, top_k=top_k)

    if use_reranking:
        evidence = cohere_rerank(query, candidates, top_n=top_n)
    else:
        # Wrap plain RetrievedChunk into RerankedChunk so the schema stays consistent
        evidence = [RerankedChunk(**c.model_dump()) for c in candidates[:top_n]]

    context = build_context(evidence)
    answer = generate_answer(query, context)

    return RAGAnswer(query=query, answer=answer, evidence=evidence, used_reranking=use_reranking)


if __name__ == "__main__":
    query = "What are the transparency requirements for high-risk AI systems?"
    result = run_pipeline(query, use_reranking=True)

    print("ANSWER:\n", result.answer)
    print("\nEVIDENCE USED:")
    for e in result.evidence:
        print(f"  - {e.source} / {e.section} (rerank_score={e.rerank_score})")