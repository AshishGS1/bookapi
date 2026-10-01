from typing import Optional

from pydantic import BaseModel


class UploadRes(BaseModel):
    doc_id: str
    num_chunks: int


class QueryReq(BaseModel):
    query: str
    top_k: Optional[int] = None  # falls back to top_k


class SrcChunk(BaseModel):
    text: str
    score: float
    chunk_index: int
    doc_id: str

class QueryRes(BaseModel):
    answer: str
    sources: list[SrcChunk]
