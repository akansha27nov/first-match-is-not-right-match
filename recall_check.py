"""
Checks for a recall gap: does the chunk most likely to be the *correct* legal
answer even show up in the retrieval candidate pool, at any reasonable top_k?

Two-step check:
1. Fetch ALL vectors' metadata from Pinecone (via a wide dummy query) and
   find any chunk whose text contains the target keyword (e.g. "Article 13").
2. Run the real query at top_k=20 and see whether that chunk's ID is present.
   If it's not in top_k=20, similarity search structurally cannot surface it —
   no reranker can fix that, since rerankers only reorder what's retrieved.
"""

from retrieval import baseline_search, embed_query, index

TARGET_KEYWORD = "Article 13" # based on the output observed
QUERY = "What are the transparency requirements for high-risk AI systems?"


def find_chunks_containing(keyword: str, sample_size: int = 354):
    """Pull a large batch of vectors back via a neutral query and grep metadata text."""
    vector = embed_query(keyword)  # bias the search toward the keyword itself
    results = index.query(vector=vector, top_k=sample_size, include_metadata=True)
    matches = [
        m for m in results["matches"]
        if keyword.lower() in m["metadata"]["text"].lower()
    ]
    return matches


def check_recall(query: str, keyword: str, top_k: int = 20):
    target_chunks = find_chunks_containing(keyword)
    if not target_chunks:
        print(f"No chunk found containing '{keyword}' even in a keyword-biased search of the index.")
        print("Either it's phrased differently in the text, or it didn't survive chunking/extraction.")
        return

    target_ids = {m["id"] for m in target_chunks}
    print(f"Found {len(target_ids)} chunk(s) containing '{keyword}':")
    for m in target_chunks:
        print(f"  {m['id']} — {m['metadata']['source']} / {m['metadata']['section']} (sim to '{keyword}' query: {m['score']:.3f})")

    retrieved = baseline_search(query, top_k=top_k)
    retrieved_ids = {c.id for c in retrieved}

    print(f"\nQuery: \"{query}\"")
    print(f"Retrieved top_{top_k} chunk IDs: {sorted(retrieved_ids)}")

    overlap = target_ids & retrieved_ids
    if overlap:
        print(f"\n✅ Target chunk(s) DID appear in top_{top_k}: {overlap}")
    else:
        print(f"\n❌ RECALL GAP: none of the '{keyword}' chunks appear in top_{top_k}.")
        print("This means reranking cannot help — the correct evidence was never in the candidate pool.")
        for wider_k in (30, 50, 100):
            wide = baseline_search(query, top_k=wider_k)
            wide_ids = {c.id for c in wide}
            if target_ids & wide_ids:
                print(f"   (First appears at top_{wider_k})")
                break
        else:
            print("   (Still absent even at top_100 — deep recall failure)")


if __name__ == "__main__":
    check_recall(QUERY, TARGET_KEYWORD, top_k=20)
