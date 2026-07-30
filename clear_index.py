# clear_index.py
from pinecone import Pinecone
from config import PINECONE_API_KEY, INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
index.delete(delete_all=True)
print("Index cleared.")