import uuid
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from core.config import settings
from core.logger import Logger

logger = Logger().get_logger()


class QdrantProcessor:
    def __init__(self) -> None:
        self.client = QdrantClient(
            host=settings.qdrant.HOST,
            port=settings.qdrant.PORT,
            prefer_grpc=True,
        )
        self.collection_name = settings.qdrant.COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [
            c.name for c in self.client.get_collections().collections
        ]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection '%s'", self.collection_name)
        else:
            logger.info("Collection '%s' already exists", self.collection_name)

    def _build_payload(self, username: str, is_correct: bool) -> dict:
        """Build a standardised payload dict."""
        return {"username": username, "is_correct": is_correct}

    def create(
        self,
        embedding: np.ndarray,
        username: str,
        is_correct: bool = True,
        point_id: Optional[str] = None,
    ) -> str:
        """Insert a single embedding. Returns the point ID."""
        pid = point_id or str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=pid,
                    vector=embedding.tolist(),
                    payload=self._build_payload(username, is_correct),
                )
            ],
        )
        logger.info("Created point %s for user '%s'", pid, username)
        return pid

    def create_batch(
        self,
        embeddings: list[np.ndarray],
        usernames: list[str],
        is_corrects: Optional[list[bool]] = None,
        point_ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Insert multiple embeddings in one call. Returns point IDs."""
        ids = point_ids or [str(uuid.uuid4()) for _ in embeddings]
        flags = is_corrects or [True] * len(embeddings)
        points = [
            PointStruct(
                id=pid,
                vector=emb.tolist(),
                payload=self._build_payload(uname, flag),
            )
            for pid, emb, uname, flag in zip(ids, embeddings, usernames, flags)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Created %d points", len(points))
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """Search for the most similar embeddings.

        Returns a list of dicts with keys:
        id, score, username, is_correct.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
        )
        return [
            {
                "id": r.id,
                "score": r.score,
                "username": r.payload.get("username") if r.payload else None,
                "is_correct": r.payload.get("is_correct")
                if r.payload
                else None,
            }
            for r in results.points
        ]

    def get(self, point_id: str) -> Optional[dict]:
        """Retrieve a single point by ID.

        Returns a dict with keys:
        id, vector, username, is_correct — or None.
        """
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True,
        )
        if not results:
            return None
        point = results[0]
        payload = point.payload or {}
        return {
            "id": point.id,
            "vector": point.vector,
            "username": payload.get("username"),
            "is_correct": payload.get("is_correct"),
        }

    def update(
        self,
        point_id: str,
        embedding: Optional[np.ndarray] = None,
        username: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ) -> None:
        """Update a point's vector and/or payload fields."""
        if embedding is not None:
            # Re-upsert the full point; fetch existing payload if needed.
            existing = self.get(point_id)
            uname = (
                username
                if username is not None
                else (existing["username"] if existing else "")
            )
            flag = (
                is_correct
                if is_correct is not None
                else (existing["is_correct"] if existing else True)
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=self._build_payload(uname, flag),
                    )
                ],
            )
        else:
            # Partial payload update
            partial: dict = {}
            if username is not None:
                partial["username"] = username
            if is_correct is not None:
                partial["is_correct"] = is_correct
            if partial:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=partial,
                    points=[point_id],
                )
        logger.info("Updated point %s", point_id)

    def delete(self, point_ids: list[str]) -> None:
        """Delete points by their IDs."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        logger.info("Deleted %d points", len(point_ids))
