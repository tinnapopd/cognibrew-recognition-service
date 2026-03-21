from typing import Optional

from pydantic import BaseModel, Field


class CreateRequest(BaseModel):
    embedding: list[float]
    username: str
    is_correct: bool = True
    point_id: Optional[str] = None


class CreateResponse(BaseModel):
    point_id: str


class BatchItem(BaseModel):
    embedding: list[float]
    username: str
    is_correct: bool = True
    point_id: Optional[str] = None


class CreateBatchRequest(BaseModel):
    items: list[BatchItem]


class CreateBatchResponse(BaseModel):
    point_ids: list[str]


class SearchRequest(BaseModel):
    embedding: list[float]
    top_k: int = Field(default=5, ge=1)
    score_threshold: Optional[float] = None


class SearchResult(BaseModel):
    id: str
    score: float
    username: Optional[str] = None
    is_correct: Optional[bool] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class GetResponse(BaseModel):
    id: str
    vector: Optional[list[float]] = None
    username: Optional[str] = None
    is_correct: Optional[bool] = None


class UpdateRequest(BaseModel):
    embedding: Optional[list[float]] = None
    username: Optional[str] = None
    is_correct: Optional[bool] = None


class DeleteRequest(BaseModel):
    point_ids: list[str]


class MessageResponse(BaseModel):
    message: str
