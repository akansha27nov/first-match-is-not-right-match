import re
import tiktoken
import pdfplumber
from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import (
    OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME,
    EMBED_MODEL, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP,
    PODCAST_TRANSCRIPT_PATH, PDF_PATH,
)

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
enc = tiktoken.get_encoding("cl100k_base")

ARTICLE_HEADING_RE = re.compile(r'^Article\s+(\d+)\s*$')
RECITAL_RE = re.compile(r'^\((\d{1,3})\)\s')
CHAPTER_RE = re.compile(r'^CHAPTER\s+([IVXLC]+)\b', re.IGNORECASE)
ANNEX_RE = re.compile(r'^ANNEX\s+([IVXLC]+)\b', re.IGNORECASE)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def load_podcast(path: str = PODCAST_TRANSCRIPT_PATH):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    return [
        {
            "text": c,
            "metadata": {"source": "podcast", "section": f"segment_{i}"},
        }
        for i, c in enumerate(chunks)
    ]


def load_eu_ai_act(path: str):
    records = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            current_article = current_recital = current_chapter = current_annex = None
            buffer, buffer_tokens, chunk_idx = [], 0, 0

            def flush():
                nonlocal buffer, buffer_tokens, chunk_idx
                if not buffer:
                    return
                records.append({
                    "text": " ".join(buffer),
                    "metadata": {
                        "source": "EU_AI_Act",
                        "section": f"page_{page_num + 1}_chunk_{chunk_idx}",
                        "article": current_article or "",
                        "recital": current_recital or "",
                        "chapter": current_chapter or "",
                        "annex": current_annex or "",
                    },
                })
                chunk_idx += 1
                buffer, buffer_tokens = [], 0

            for line in text.split("\n"):
                stripped = line.strip()
                is_new_boundary = (
                    ARTICLE_HEADING_RE.match(stripped)
                    or RECITAL_RE.match(stripped)
                    or CHAPTER_RE.match(stripped)
                    or ANNEX_RE.match(stripped)
                )

                # Flush BEFORE updating the tag, so the outgoing chunk keeps the old label
                if is_new_boundary and buffer:
                    flush()

                if m := ARTICLE_HEADING_RE.match(stripped):
                    current_article = m.group(1)
                if m := RECITAL_RE.match(stripped):
                    current_recital = m.group(1)
                if m := CHAPTER_RE.match(stripped):
                    current_chapter = m.group(1)
                if m := ANNEX_RE.match(stripped):
                    current_annex = m.group(1)

                buffer.append(line)
                buffer_tokens += len(enc.encode(line))
                if buffer_tokens >= CHUNK_SIZE:
                    flush()

            flush()
    return records


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_index():
    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def run_prep():
    records = load_podcast() + load_eu_ai_act(PDF_PATH)
    texts = [r["text"] for r in records]

    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        embeddings.extend(embed_batch(texts[i:i + batch_size]))

    index = build_index()

    vectors = [
        {
            "id": f"chunk-{i}",
            "values": emb,
            "metadata": {**records[i]["metadata"], "text": records[i]["text"][:1000]},
        }
        for i, emb in enumerate(embeddings)
    ]

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])

    return index, len(records)


if __name__ == "__main__":
    index, n_chunks = run_prep()
    print(f"Upserted {n_chunks} chunks into Pinecone index '{INDEX_NAME}'")
    print(index.describe_index_stats())