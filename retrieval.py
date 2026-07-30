from openai import OpenAI
from pinecone import Pinecone

from config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME, EMBED_MODEL
from models import RetrievedChunk

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def embed_query(query: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return resp.data[0].embedding


def baseline_search(query: str, top_k: int = 10, source_filter: str = None) -> list[RetrievedChunk]:
    vector = embed_query(query)
    filter_dict = {"source": source_filter} if source_filter else None

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
    )

    return [
        RetrievedChunk(
            id=m["id"],
            score=m["score"],
            text=m["metadata"]["text"],
            source=m["metadata"]["source"],
            section=m["metadata"]["section"],
        )
        for m in results["matches"]
    ]


if __name__ == "__main__":
    query = "What are the transparency requirements for high-risk AI systems?"
    hits = baseline_search(query, top_k=5)
    for h in hits:
        print(f"[{h.score:.3f}] {h.source} / {h.section}")
        print(f"  {h.text[:120]}...")