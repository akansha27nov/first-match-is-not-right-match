# First Match ≠ Right Match — Relevance Scoring & Rerankers

RAG pipeline comparing baseline vector retrieval against LLM-based and Cohere-based reranking, using the EU AI Act and a Trustworthy AI podcast as source material.

## What this project does

1. Extracts and chunks two source documents (PDF + podcast audio)
2. Embeds chunks and indexes them in Pinecone
3. Runs baseline vector similarity search
4. Reranks results two ways — LLM relevance scoring and Cohere's reranker
5. Generates a grounded answer via `GPT-4o-mini` using reranked context
6. Evaluates whether reranking changes/improves the answer, and checks whether the answer's citations actually match what was retrieved

See `lab_proof.md` for the full write-up: queries, retrieved evidence, reranking behavior, and a documented limitation (citation-vs-retrieval grounding gap).

## Setup

```bash
pip install openai pinecone cohere pypdf pdfplumber tiktoken python-dotenv pydantic
```

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

# 2. Chunk both sources, embed, and upsert to Pinecone
#    (clear_index.py wipes the index first if re-running after a chunking/extraction change)
python clear_index.py
python data_prep.py

# 3. Baseline vector search
python retrieval.py

# 4. Reranking — two approaches, run independently
python llm_rerank.py       # GPT-4o-mini relevance scoring blended with similarity
python cohere_rerank.py    # Cohere rerank-english-v3.0

# 5. Full pipeline: retrieve -> rerank -> generate grounded answer
python rag_pipeline.py

# 6. Evaluate: same queries, with vs. without reranking, saved to eval_results.json
python evaluate.py

# 7. Grounding check: do the answer's citations match what was actually retrieved?
python grounding_check.py
```

## Project structure

```
config.py            # paths, model names, chunk size/overlap
models.py            # pydantic models: RetrievedChunk, RerankedChunk, LLMRelevanceScore
transcribe.py        # Whisper API transcription with size guard + caching
data_prep.py         # PDF (pdfplumber) + transcript chunking, embedding, Pinecone upsert
clear_index.py       # wipes the Pinecone index (use before re-running data_prep after changes)
retrieval.py         # baseline vector similarity search
llm_rerank.py        # LLM-based relevance scoring reranker
cohere_rerank.py     # Cohere API reranker
rag_pipeline.py      # full retrieve -> rerank -> answer pipeline
evaluate.py          # with/without reranking comparison across test queries
grounding_check.py   # checks whether answer citations match retrieved chunk IDs
lab_proof.md         # submission: queries, evidence, reranking analysis, limitation
eval_results.json    # raw output from evaluate.py
```

## Known limitations

- **Grounding gap:** the pipeline retrieves 5 chunks per query but the LLM typically cites only 1–2 in its answer text. The "evidence used" list reflects what was retrieved, not necessarily what grounded the final answer. See `lab_proof.md` for measured numbers and a proposed mitigation (structured claim-to-chunk citation mapping).
- **Cohere score compression:** rerank scores cluster near 1.0 when the baseline candidate set is already strong, giving weak fine-grained discrimination between top candidates.

## Notes on setup issues encountered

- Initial PDF extraction with `pypdf` produced broken word spacing; switched to `pdfplumber` for clean text.
- Switching extractors changes chunk boundaries/counts — always run `clear_index.py` before re-running `data_prep.py` after any change to chunking or extraction logic, or the index will end up with a mix of old and new vectors under overlapping IDs.
