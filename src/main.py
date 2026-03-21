import threading
import time
from typing import Optional, Dict, Any

import numpy as np

from core.config import settings
from core.logger import Logger
from core.message_queue import MessageQueue
from core.vectordb import QdrantProcessor

from schemas.proto.face_embedding_pb2 import FaceEmbedding  # type: ignore
from schemas.proto.face_result_pb2 import FaceRecognized  # type: ignore
from schemas.proto.face_update_pb2 import PersonUpdate  # type: ignore

logger = Logger().get_logger()


# Consumer 1: Face recognition (inference --> recognition)
class RecognitionProcessor:
    """Consumes FaceEmbedding protobuf messages from the inference service,
    looks up identities in Qdrant, and publishes match results as protobuf."""

    def __init__(self) -> None:
        self.threshold = settings.model.SIMILARITY_THRESHOLD
        self.qdrant = QdrantProcessor()

        # Inbound – consumes from cognibrew.inference exchange
        self.mq = MessageQueue()

        # Outbound – publishes results on the same exchange
        # (reuses the same connection after connect)

    def _identify(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        hits = self.qdrant.search(
            query_embedding=embedding,
            top_k=1,
            score_threshold=self.threshold,
        )
        if not hits:
            return None
        return hits[0]

    def _on_face_embedded(self, body: bytes) -> None:
        """Callback for each face.embedded protobuf message."""
        msg = FaceEmbedding()
        msg.ParseFromString(body)

        embedding = np.array(msg.embedding, dtype=np.float32)
        bbox = list(msg.bbox)
        det_score = msg.det_score

        t0 = time.perf_counter()
        match = self._identify(embedding)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        if match:
            result = FaceRecognized(
                username=match["username"],
                score=match["score"],
                bbox=bbox,
            )
            self.mq.publish(
                body=result.SerializeToString(),
                routing_key=settings.rabbitmq.FACE_RECOGNIZED_ROUTING_KEY,
            )
            logger.info(
                "face_recognized",
                extra={
                    "username": match["username"],
                    "score": round(match["score"], 2),
                    "det_score": round(det_score, 2),
                    "bbox": bbox,
                    "latency_ms": latency_ms,
                },
            )
        else:
            logger.info(
                "face_unknown",
                extra={
                    "det_score": round(det_score, 2),
                    "bbox": bbox,
                    "latency_ms": latency_ms,
                },
            )

    def start(self) -> None:
        logger.info("Recognition consumer started, waiting for embeddings…")
        self.mq.connect(
            binding_keys=[settings.rabbitmq.FACE_EMBEDDED_ROUTING_KEY],
        )
        try:
            self.mq.consume(callback=self._on_face_embedded)
        finally:
            self.mq.close()
            logger.info("Recognition consumer stopped")


# Consumer 2: Face updates (cloud --> vectordb)
class FaceUpdateProcessor:
    """Consumes PersonUpdate protobuf messages from the cloud and
    creates / updates / deletes person embeddings in Qdrant."""

    def __init__(self) -> None:
        self.qdrant = QdrantProcessor()
        self.mq = MessageQueue(
            exchange_name=settings.rabbitmq.FACE_UPDATE_EXCHANGE_NAME,
            queue_name=settings.rabbitmq.FACE_UPDATE_QUEUE_NAME,
        )

    def _on_face_updated(self, body: bytes) -> None:
        """Callback for each face.updated protobuf message."""
        msg = PersonUpdate()
        msg.ParseFromString(body)

        person_id = msg.person_id
        username = msg.username
        action = msg.action

        if action == PersonUpdate.CREATE:
            embedding = np.array(msg.embedding, dtype=np.float32)
            self.qdrant.create(
                embedding=embedding,
                username=username,
                point_id=person_id,
            )
            logger.info(
                "person_created",
                extra={"person_id": person_id, "username": username},
            )

        elif action == PersonUpdate.UPDATE:
            embedding = (
                np.array(msg.embedding, dtype=np.float32)
                if len(msg.embedding) > 0
                else None
            )
            self.qdrant.update(
                point_id=person_id,
                embedding=embedding,
                username=username if username else None,
            )
            logger.info(
                "person_updated",
                extra={"person_id": person_id, "username": username},
            )

        elif action == PersonUpdate.DELETE:
            self.qdrant.delete(point_ids=[person_id])
            logger.info(
                "person_deleted",
                extra={"person_id": person_id},
            )

        else:
            logger.warning(
                "unknown_person_action",
                extra={
                    "person_id": person_id,
                    "action": PersonUpdate.Action.Name(action),
                },
            )

    def start(self) -> None:
        logger.info("FaceUpdate consumer started, waiting for updates…")
        self.mq.connect(
            binding_keys=[settings.rabbitmq.FACE_UPDATE_ROUTING_KEY],
        )
        try:
            self.mq.consume(callback=self._on_face_updated)
        finally:
            self.mq.close()
            logger.info("FaceUpdate consumer stopped")


# Entrypoint: run both consumers on separate threads
def main() -> None:
    recognition = RecognitionProcessor()
    face_update = FaceUpdateProcessor()

    t1 = threading.Thread(
        target=recognition.start, name="recognition", daemon=True
    )
    t2 = threading.Thread(
        target=face_update.start, name="face-update", daemon=True
    )

    t1.start()
    t2.start()

    logger.info("All consumers running")

    # Block until either thread exits (or KeyboardInterrupt)
    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        logger.info("Shutting down…")


if __name__ == "__main__":
    main()
