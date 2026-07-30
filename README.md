# First Match ≠ Right Match — Relevance Scoring & Rerankers

RAG pipeline comparing baseline vector retrieval against LLM-based and Cohere-based reranking, plus article-level metadata filtering, using the EU AI Act and a Trustworthy AI podcast as source material.

## What this project does

1. Extracts and chunks two source documents (PDF + podcast audio), tagging each EU AI Act chunk with its article/recital/chapter/annex number
2. Embeds chunks and indexes them in Pinecone
3. Runs baseline vector similarity search
4. Reranks results two ways — LLM relevance scoring and Cohere's reranker
5. Filters retrieval by exact legal-structure metadata (article/recital/chapter/annex), independent of similarity score
6. Generates a grounded answer via GPT-4o-mini using reranked context
7. Evaluates whether reranking changes/improves the answer, and checks whether the answer's citations actually match what was retrieved

See `lab_proof.md` for the full write-up: queries, retrieved evidence, reranking behavior, a chunking-boundary bug and its fix, and measured grounding/citation gaps.

## Setup

```bash
pip install openai pinecone-client cohere pypdf pdfplumber tiktoken python-dotenv
```

`ffmpeg` is required for audio compression (macOS: `brew install ffmpeg`).

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
COHERE_API_KEY=...
```

## Data

Place source files in `data/`:
- `eu_ai_act.pdf` — EU AI Act text
- `The_Blueprint_For_Trustworthy_AI.m4a` — podcast audio (raw)

Whisper's API has a 25MB file size limit. The raw podcast audio exceeds this, so it's compressed before transcription:

```bash
ffmpeg -i data/The_Blueprint_For_Trustworthy_AI.m4a -ac 1 -ar 16000 -b:a 64k -codec:a libmp3lame data/podcast.mp3
```

## Pipeline — run in order

```bash
# 1. Transcribe podcast audio (cached — skips if transcript already exists)
python transcribe.py

# 2. Chunk both sources (with article/recital/chapter/annex tagging), embed, upsert to Pinecone
#    clear_index.py wipes the index first — always run before re-running data_prep.py after
#    any change to chunking or extraction logic, or the index ends up with a mix of old/new vectors
python clear_index.py
python data_prep.py

# 3. Baseline vector search, across all 3 test queries
python retrieval.py

# 4. Reranking — two approaches, run independently, all 3 test queries
python llm_rerank.py       # GPT-4o-mini relevance scoring blended with similarity
python cohere_rerank.py    # Cohere rerank-english-v3.0

# 5. Metadata filtering — exact match on article/recital/chapter/annex, bypasses similarity ranking entirely
python metadata_filter.py

# 6. Full pipeline: retrieve -> rerank -> generate grounded answer, all 3 test queries
python rag_pipeline.py

# 7. Evaluate: same queries, with vs. without reranking, saved to eval_results.json
python evaluate.py

# 8. Grounding check: do the answer's citations match what was actually retrieved?
python grounding_check.py
```

## Project structure

```
config.py                     # paths, model names, chunk size/overlap
models.py                     # pydantic models: RetrievedChunk, RerankedChunk, LLMRelevanceScore
transcribe.py                 # Whisper API transcription with size guard + caching
data_prep.py                  # PDF chunking with metadata tagging, transcript chunking, embedding, Pinecone upsert
clear_index.py                # wipes the Pinecone index (use before re-running data_prep after changes)
retrieval.py                  # baseline vector similarity search
llm_rerank.py                 # LLM-based relevance scoring reranker
cohere_rerank.py              # Cohere API reranker
metadata_filter.py            # Step 5: exact-match filtering by article/recital/chapter/annex
rag_pipeline.py               # full retrieve -> rerank -> answer pipeline
evaluate.py                   # with/without reranking comparison across test queries
grounding_check.py            # checks whether answer citations match retrieved chunk IDs — cited in lab_proof.md
lab_proof.md                  # submission: queries, evidence, reranking analysis, limitation
eval_results.json             # raw output from evaluate.py
```

### Optional / diagnostic scripts

Used during development to find and confirm a bug, but not required for the pipeline to run:

- `recall_check.py` — checks whether a target chunk is reachable at any top_k before reranking. Used to originally surface an apparent recall gap, later traced to a chunking-boundary bug (see `lab_proof.md`). Not cited by name in the final write-up, but documents part of the debugging trail — safe to keep or drop.

## Known limitations

- **Chunking-boundary bug (found and fixed):** the original chunking logic only flushed a chunk buffer on a token-size threshold, not on legal-structure boundaries (a new `Article N` heading). This let short articles get absorbed into neighboring chunks and mistagged, which initially looked like a fundamental retrieval recall gap (Article 13 didn't surface until top_k=100) but was actually a chunking defect. Fixed by flushing on every article/recital/chapter/annex heading in addition to the token threshold. See `lab_proof.md` for the full before/after trace.
- **Grounding/citation gap:** the pipeline retrieves 5 chunks per query but the LLM cites anywhere from 1-4 of them depending on the query, measured with `grounding_check.py`. Citation rate improved after the chunking fix but still varies — see the per-query table in `lab_proof.md`.
- **Cohere score compression:** rerank scores cluster near 1.0 when the baseline candidate set is already strong, giving weak fine-grained discrimination between top candidates.

## Notes on setup issues encountered

- Initial PDF extraction with `pypdf` produced broken word spacing; switched to `pdfplumber` for clean text.
- Switching extractors or chunking logic changes chunk boundaries/counts — always run `clear_index.py` before re-running `data_prep.py`, or the index ends up with a mix of old and new vectors under overlapping IDs.
- Article/recital tagging initially bled across chunk boundaries (a chunk's metadata tag could reflect a heading that appeared *after* most of the chunk's actual content, if both landed in the same token-size buffer). Fixed by flushing the buffer immediately on any new heading, not just at the token limit.
