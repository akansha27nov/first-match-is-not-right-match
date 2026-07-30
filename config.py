import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

INDEX_NAME = "eu-ai-act-reranking"
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
EMBED_DIM = 1536
CHUNK_SIZE = 500        # tokens
CHUNK_OVERLAP = 75

PODCAST_AUDIO_PATH = "data/The_Blueprint_For_Trustworthy_AI.m4a"
PODCAST_TRANSCRIPT_PATH = "data/podcast_transcript.txt"
EU_AI_ACT_PDF_PATH = "data/eu_ai_act.pdf"