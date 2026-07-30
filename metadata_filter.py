from retrieval import embed_query, index
from models import RetrievedChunk


def filtered_search(
    query: str,
    top_k: int = 10,
    source: str = None,
    article: str = None,
    recital: str = None,
    chapter: str = None,
    annex: str = None,
) -> list[RetrievedChunk]:
    filter_dict = {}
    if source:
        filter_dict["source"] = source
    if article:
        filter_dict["article"] = article
    if recital:
        filter_dict["recital"] = recital
    if chapter:
        filter_dict["chapter"] = chapter
    if annex:
        filter_dict["annex"] = annex

    vector = embed_query(query)
    results = index.query(
        vector=vector, top_k=top_k, include_metadata=True,
        filter=filter_dict or None,
    )

    return [
        RetrievedChunk(
            id=m["id"], score=m["score"], text=m["metadata"]["text"],
            source=m["metadata"]["source"], section=m["metadata"]["section"],
            article=m["metadata"].get("article"), recital=m["metadata"].get("recital"),
            chapter=m["metadata"].get("chapter"), annex=m["metadata"].get("annex"),
        )
        for m in results["matches"]
    ]


if __name__ == "__main__":
    # Direct test: can we retrieve Article 13 by filtering on its number,
    # bypassing the similarity-based recall gap entirely?
    query = "What are the transparency requirements for high-risk AI systems?"

    print("--- Unfiltered ---")
    for c in filtered_search(query, top_k=5):
        print(f"  [{c.score:.3f}] article={c.article} recital={c.recital} — {c.section}")

    print("\n--- Filtered: article=13 ---")
    for c in filtered_search(query, top_k=5, article="13"):
        print(f"  [{c.score:.3f}] article={c.article} — {c.section}")
        print(f"    {c.text[:150]}...")