# Lab Proof: First Match ≠ Right Match
### Relevance Scoring & Rerankers — EU AI Act + Trustworthy AI Podcast

---

## System Overview

- **Sources:** EU AI Act (144-page PDF, extracted with `pdfplumber`), Trustworthy AI podcast (15.6 min audio, transcribed via OpenAI Whisper API)
- **Chunking:** 500 tokens/chunk (with mid-buffer flush on any article/recital/chapter/annex heading, so no chunk straddles a legal-structure boundary), metadata tagged with `source`, `section`, `article`, `recital`, `chapter`, `annex`
- **Index:** Pinecone, 660 vectors, `text-embedding-3-small` (1536 dims), cosine similarity
- **Rerankers tested:** LLM relevance scoring (GPT-4o-mini, similarity/LLM blend) and Cohere `rerank-english-v3.0`
- **Metadata filtering (Step 5):** exact-match filtering on `article`/`recital`/`chapter`/`annex`, independent of similarity score
- **Pipeline:** `baseline_search` → `cohere_rerank` → context assembly → `gpt-4o-mini` answer generation
- **Validation:** all retrieval/reranking objects typed with pydantic (`RetrievedChunk`, `RerankedChunk`) to catch malformed metadata at the retrieval boundary

---

## Query 1: "What are the transparency requirements for high-risk AI systems?"

### Retrieved documents (baseline, top 5 by similarity)
| Score | Article | Source |
|---|---|---|
| 0.733 | 13 | EU_AI_Act / page_59_chunk_2 |
| 0.708 | — (Chapter IV heading) | EU_AI_Act / page_82_chunk_1 |
| 0.707 | — (Recital 72) | EU_AI_Act / page_21_chunk_1 |
| 0.697 | — (Recital 66) | EU_AI_Act / page_19_chunk_2 |
| 0.694 | — (Chapter III heading) | EU_AI_Act / page_53_chunk_2 |

**Article 13 — the section literally titled "Transparency and provision of information to deployers" — is now the #1 baseline result**, confirming the chunking-boundary fix resolved the earlier recall gap (see Limitation section).

### Reranking process
- **LLM relevance scoring** (α=0.5 blend) kept Article 13 at rank #1 (`llm_score=1.00`, `combined=0.866`) and pushed the weakest candidate (Chapter III heading, `llm_score=0.20`) to last — a sensible, confident reordering.
- **Cohere rerank** scores again clustered near the ceiling (0.998–1.000 across all 5), swapping Article 13 and Recital 72 for the #1/#2 slots but not meaningfully separating the middle of the pack — same score-compression pattern as before, see Limitation.

### Final answer (with Cohere reranking)
> The transparency requirements for high-risk AI systems include the following: High-risk AI systems must be designed and developed to ensure that their operation is sufficiently transparent, enabling deployers to interpret the system's output and use it appropriately (Article 13, Section 1). They must be accompanied by instructions for use that are concise, complete, correct, and clear (Article 13, Section 2), including the identity and contact details of the provider and the system's characteristics, capabilities, and limitations (Article 13, Section 3). Comprehensive information on how the system was developed and performs throughout its lifetime is also required for traceability and compliance verification.

### Evidence supporting the answer (top 5, after Cohere reranking)

| # | Source | Article | Similarity | Rerank score | Cited in answer? |
|---|---|---|---|---|---|
| 1 | EU_AI_Act / page_21_chunk_1 | — | 0.707 | 1.000 | ❌ No |
| 2 | EU_AI_Act / page_59_chunk_2 | 13 | 0.733 | 1.000 | ✅ Yes |
| 3 | EU_AI_Act / page_19_chunk_2 | — | 0.697 | 1.000 | ❌ No |
| 4 | EU_AI_Act / page_20_chunk_4 | — | 0.685 | 0.999 | ✅ Yes |
| 5 | EU_AI_Act / page_55_chunk_2 | 8 | 0.693 | 0.998 | ❌ No |

(full raw output in `eval_results.json`)

**Grounding check result:** 5 chunks retrieved → **3** cited in the answer (Article 13 content plus the Recital-71/page_20 documentation point). 2 of 5 (40%) went unused — an improvement over the pre-fix run (was 4/5, 80% unused), likely because cleaner per-article chunks give the LLM more directly quotable, less redundant material to draw from.

---

## Query 2: "What record-keeping obligations apply to high-risk AI systems?"

### Retrieved documents (baseline, top 5)
| Score | Article | Source |
|---|---|---|
| 0.773 | 12 | EU_AI_Act / page_59_chunk_1 |
| 0.721 | — (Recital 71) | EU_AI_Act / page_20_chunk_4 |
| 0.701 | 16 | EU_AI_Act / page_62_chunk_1 |
| 0.696 | — (Chapter VIII heading) | EU_AI_Act / page_100_chunk_1 |
| 0.685 | — (Chapter III heading) | EU_AI_Act / page_53_chunk_2 |

Article 12 (Record-keeping) is the clean #1 baseline hit, and Article 16 (providers' obligations) already appears in baseline top-5 post-fix — previously it only surfaced after Cohere reranking pulled it in from outside the top-5.

### Reranking process
Cohere reranking swapped in **Article 19** ("Automatically generated logs" — providers must keep logs for a minimum retention period) at rank #4, a chunk that did not appear in the baseline top-5 at all. This is the clearest case in this run of reranking surfacing a genuinely complementary legal provision: baseline gets "systems must be able to log" (Art. 12) and "providers must keep logs" (Art. 16), but only reranking surfaces "...for at least six months" (Art. 19) — a specific, actionable detail absent from the baseline-only answer.

### Final answer (with reranking)
> High-risk AI systems have specific record-keeping obligations: they must technically allow for automatic recording of events over their lifetime (Article 12), with logs enabling identification of risk situations, post-market monitoring, and operational monitoring. For certain systems, minimum logging requirements include the period of each use and reference database checks (Article 12). Providers must keep the automatically generated logs, as long as under their control, for a period appropriate to the system's purpose — at least six months (Article 19). Providers must also keep compliance documentation up to date throughout the system's lifetime (Recital 71).

### Evidence supporting the answer
5 chunks retrieved after reranking (`page_19_chunk_2`, `page_59_chunk_1` [Art. 12], `page_62_chunk_1` [Art. 16], `page_64_chunk_1` [Art. 19], `page_20_chunk_4`). Answer explicitly cites **4 of 5** — Article 12, Article 19, and Recital 71 are all directly referenced; only `page_19_chunk_2` (Recital 66, general risk-management framing) went unused. This is the best grounding ratio of the three test queries (80% cited, only 20% unused per `grounding_check.py`).

---

## Query 3: "How does the AI Act define an AI system?"

### Retrieved documents (baseline, top 5)
| Score | Article | Source |
|---|---|---|
| 0.690 | — (Recital 12) | EU_AI_Act / page_4_chunk_2 |
| 0.681 | 3 | EU_AI_Act / page_46_chunk_2 |
| 0.609 | — (Recital 18-adjacent) | EU_AI_Act / page_47_chunk_10 |
| 0.592 | — (Chapter III heading) | EU_AI_Act / page_53_chunk_2 |
| 0.577 | — (Recital 100) | EU_AI_Act / page_26_chunk_4 |

### Reranking process
Reranking left the answer and citation pattern essentially unchanged — both with and without reranking, the model centers its definition on the same chunk, `page_46_chunk_2` (Article 3, the Act's formal definitions section). Notably, Cohere reranking swapped out the two lowest-relevance baseline chunks (Chapter III heading, Recital 100) for two different low-scoring recitals (`page_14_chunk_4`, `page_14_chunk_3`) without changing which chunk actually drives the answer — a case where reranking churns the tail of the candidate list without affecting the outcome.

### Final answer (with reranking)
> The AI Act defines an "AI system" as a machine-based system that is designed to operate with varying levels of autonomy and may exhibit adaptiveness after deployment. It infers from the input it receives how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments. This definition is supported by the source [EU_AI_Act / page_46_chunk_2].

### Evidence supporting the answer
5 chunks retrieved, only **1** (`page_46_chunk_2`, Article 3) cited — the weakest grounding ratio of the three test queries (80% unused per `grounding_check.py`). Unlike Query 1 and 2, this query has a single, self-contained legal definition as its answer, so the model may be correctly recognizing that the other 4 retrieved chunks (mostly recitals restating or elaborating on the same definition) don't add independently citable content — worth noting as a case where low citation count isn't necessarily a grounding failure, just a query with one clean best-source.

---

## Limitation / Failure Case (revised after root-cause investigation)

**Initial finding (later disproven):** Article 13 — the section titled "Transparency and provision of information to deployers," and the most direct legal answer to Query 1 — did not appear in baseline vector search results until `top_k=100`, even though it was present in the index. This looked like a fundamental recall gap: reranking only reorders retrieved candidates, so a chunk excluded from retrieval can never be recovered by reranking.

**Root cause, once investigated:** the actual problem was a **chunking boundary bug**, not an embedding/retrieval limitation. The original chunking logic only flushed a chunk buffer when it hit the 500-token size threshold, not when a new legal-structure boundary (a new `Article N` heading) appeared in the text. Because Article 12 (Record-keeping) is short, its heading and Article 13's heading both landed inside the same buffer before the token limit was reached — so the chunk was tagged with whichever article heading appeared *last*, while its actual content was still mostly Article 12's text. This didn't just mislabel metadata: it meant Article 13's real text was diluted into a chunk dominated by unrelated content, degrading its own embedding and making it a poor semantic match for a query about Article 13's content.

**Fix:** flush the chunk buffer immediately whenever a new article/recital/chapter/annex heading is detected, in addition to the token-size threshold — so no chunk ever straddles a legal-structure boundary, and each chunk's content and its metadata tag are always describing the same section. Verified fix directly:

```
--- Filtered: article=13 ---
[0.732] article=13 — page_59_chunk_2
  Article 13 Transparency and provision of information to deployers
  1. High-risk AI systems shall be designed and developed...
```

**Corrected result:** after re-chunking, Article 13's chunk (`page_59_chunk_2`) is the **#1 result by plain baseline similarity search** (0.733) for Query 1 — no metadata filtering or reranking required. Chunk count increased from 316 to 660 as a result of flushing at every legal-structure boundary (more, smaller, cleanly-scoped chunks near article-dense pages).

**Actual lesson:** this lab's premise is "first match ≠ right match," typically framed as a retrieval-ranking problem solved by reranking. This project's failure case shows a **different, upstream failure mode**: bad chunking can corrupt both a document's embedding and its metadata simultaneously, in a way that no reranker can fix, because reranking only reorders whatever chunks retrieval already returned — it cannot repair or recover content that was fragmented incorrectly at ingestion time. Chunking-boundary correctness is a precondition for reranking to have anything meaningful to work with, not an independent concern.

**Metadata filtering (Step 5), evaluated separately:** with correct per-article chunking in place, filtering by `article="13"` still adds value beyond similarity search alone — it guarantees retrieval of every chunk belonging to a specific article regardless of similarity score, which matters for exhaustive-evidence use cases (e.g. "show me everything Article 13 says," not just "show me the most similar sentence to my query").

**Secondary limitation (grounding transparency), re-measured against the corrected index:** the pipeline retrieves 5 chunks per query, but citation rate varies a lot by query type. Measured with `grounding_check.py`:

| Query | Cited | Unused | % unused |
|---|---|---|---|
| Transparency requirements | 3/5 | 2/5 | 40% |
| Record-keeping obligations | 4/5 | 1/5 | 20% |
| AI system definition | 1/5 | 4/5 | 80% |

Notably, this improved sharply for Query 1 after the chunking fix (was 80% unused pre-fix, now 40%) — cleaner, non-overlapping chunks appear to give the model more distinct, independently citable content instead of several near-duplicate chunks competing to say the same thing. Query 3 stayed high-unused even post-fix, but for a plausible reason: it has one clean, self-contained legal definition as its correct answer, so most of the other retrieved chunks (recitals restating the same definition) genuinely have nothing new to add — low citation count there may reflect the answer's actual evidentiary needs rather than the LLM ignoring available evidence. This is worth treating as a per-query, content-dependent metric rather than a single fixed system-wide failure rate.

**Secondary limitation (score calibration):** Cohere's rerank scores clustered near 1.0 across all candidates in Query 1 (0.997–1.000) in earlier runs, providing weak discrimination once the baseline candidate set is already strong.

---

## Data Quality Note (process transparency)

Initial PDF extraction with `pypdf` produced broken text spacing (e.g. `"Ar ticle 12"`, `"T o address"`) that would have degraded LLM relevance judgments. Switched to `pdfplumber`, which produced clean extraction. Also caught and fixed a Pinecone index consistency bug where re-running `data_prep.py` after the extractor swap left 107 stale/dirty vectors in the index (461 total vs. 354 newly upserted) because vector IDs were assigned by list position; resolved by clearing the index before each full re-index.
