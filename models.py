from pydantic import BaseModel, Field
from typing import Literal, Optional


class RetrievedChunk(BaseModel):
    id: str
    score: float = Field(ge=0, le=1)
    text: str
    source: Literal["podcast", "EU_AI_Act"]
    section: str
    article: Optional[str] = None
    recital: Optional[str] = None
    chapter: Optional[str] = None
    annex: Optional[str] = None


class RerankedChunk(RetrievedChunk):
    llm_score: Optional[float] = None
    combined_score: Optional[float] = None
    rerank_score: Optional[float] = None


class LLMRelevanceScore(BaseModel):
    score: float = Field(ge=0, le=1)