"""Tests for RecognitionProcessor and PersonUpdateProcessor callbacks."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from schemas.proto.face_embedding_pb2 import FaceEmbedding  # type: ignore
from schemas.proto.face_result_pb2 import FaceRecognized, FaceUnknown  # type: ignore
from schemas.proto.person_update_pb2 import PersonUpdate  # type: ignore


# ---------------------------------------------------------------------------
# RecognitionProcessor
# ---------------------------------------------------------------------------
class TestRecognitionProcessor:
    @pytest.fixture()
    def processor(self):
        with (
            patch("main.QdrantProcessor") as MockQdrant,
            patch("main.MessageQueue") as MockMQ,
        ):
            from main import RecognitionProcessor

            proc = RecognitionProcessor()
            proc.qdrant = MockQdrant.return_value
            proc.mq = MockMQ.return_value
            yield proc

    def test_face_recognized(self, processor):
        """When Qdrant returns a match, publish FaceRecognized."""
        processor.qdrant.search.return_value = [
            {"id": "p1", "score": 0.92, "username": "alice", "is_correct": True}
        ]

        msg = FaceEmbedding(
            bbox=[10, 20, 100, 200],
            embedding=[0.1] * 512,
            det_score=0.95,
        )
        processor._on_face_embedded(msg.SerializeToString())

        processor.mq.publish.assert_called_once()
        call_kwargs = processor.mq.publish.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body") or call_kwargs[0][0]

        result = FaceRecognized()
        result.ParseFromString(body)
        assert result.username == "alice"
        assert abs(result.score - 0.92) < 1e-5
        assert list(result.bbox) == [10, 20, 100, 200]

    def test_face_unknown(self, processor):
        """When Qdrant returns no match, publish FaceUnknown."""
        processor.qdrant.search.return_value = []

        msg = FaceEmbedding(
            bbox=[30, 40, 150, 250],
            embedding=[0.2] * 512,
            det_score=0.80,
        )
        processor._on_face_embedded(msg.SerializeToString())

        processor.mq.publish.assert_called_once()
        call_kwargs = processor.mq.publish.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body") or call_kwargs[0][0]

        result = FaceUnknown()
        result.ParseFromString(body)
        assert list(result.bbox) == [30, 40, 150, 250]


# ---------------------------------------------------------------------------
# PersonUpdateProcessor
# ---------------------------------------------------------------------------
class TestPersonUpdateProcessor:
    @pytest.fixture()
    def processor(self):
        with (
            patch("main.QdrantProcessor") as MockQdrant,
            patch("main.MessageQueue") as MockMQ,
        ):
            from main import PersonUpdateProcessor

            proc = PersonUpdateProcessor()
            proc.qdrant = MockQdrant.return_value
            proc.mq = MockMQ.return_value
            yield proc

    def test_create_action(self, processor):
        msg = PersonUpdate(
            person_id="p-001",
            username="alice",
            embedding=[0.5] * 512,
            action=PersonUpdate.CREATE,
        )
        processor._on_person_updated(msg.SerializeToString())

        processor.qdrant.create.assert_called_once()
        call_kwargs = processor.qdrant.create.call_args
        assert call_kwargs.kwargs["username"] == "alice"
        assert call_kwargs.kwargs["point_id"] == "p-001"

    def test_update_action(self, processor):
        msg = PersonUpdate(
            person_id="p-001",
            username="alice-updated",
            embedding=[0.6] * 512,
            action=PersonUpdate.UPDATE,
        )
        processor._on_person_updated(msg.SerializeToString())

        processor.qdrant.update.assert_called_once()
        call_kwargs = processor.qdrant.update.call_args
        assert call_kwargs.kwargs["point_id"] == "p-001"
        assert call_kwargs.kwargs["username"] == "alice-updated"

    def test_delete_action(self, processor):
        msg = PersonUpdate(
            person_id="p-001",
            action=PersonUpdate.DELETE,
        )
        processor._on_person_updated(msg.SerializeToString())

        processor.qdrant.delete.assert_called_once_with(point_ids=["p-001"])

    def test_update_without_embedding(self, processor):
        """UPDATE with no embedding should pass None for embedding."""
        msg = PersonUpdate(
            person_id="p-001",
            username="renamed",
            action=PersonUpdate.UPDATE,
        )
        processor._on_person_updated(msg.SerializeToString())

        call_kwargs = processor.qdrant.update.call_args
        assert call_kwargs.kwargs["embedding"] is None
        assert call_kwargs.kwargs["username"] == "renamed"
