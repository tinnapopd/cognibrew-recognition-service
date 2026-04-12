"""Tests for RecognitionProcessor and FaceUpdateProcessor in main.py."""

from unittest.mock import MagicMock, patch

from schemas.proto.face_embedding_pb2 import FaceEmbedding  # type: ignore
from schemas.proto.face_result_pb2 import FaceRecognized  # type: ignore
from schemas.proto.face_update_pb2 import PersonUpdate  # type: ignore


class TestRecognitionProcessor:
    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_face_recognized_publishes_result(self, MockMQ, MockQdrant):
        """When Qdrant returns a match above threshold, a FaceRecognized
        message should be published."""
        from main import RecognitionProcessor

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = [{"username": "alice", "score": 0.9}]

        proc = RecognitionProcessor()
        proc.mq = MagicMock()

        embedding = FaceEmbedding(
            face_id="face-001",
            bbox=[10, 20, 100, 200],
            embedding=[0.1] * 512,
            det_score=0.95,
        )
        proc._on_face_embedded(embedding.SerializeToString())

        proc.mq.publish.assert_called_once()
        call_kwargs = proc.mq.publish.call_args.kwargs
        assert call_kwargs["routing_key"] == "face.recognized"

        result = FaceRecognized()
        result.ParseFromString(call_kwargs["body"])
        assert result.face_id == "face-001"
        assert result.username == "alice"

    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_face_unknown_publishes_result(self, MockMQ, MockQdrant):
        """When Qdrant returns no match, a FaceRecognized message with
        username='unknown' should be published."""
        from main import RecognitionProcessor

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = []

        proc = RecognitionProcessor()
        proc.mq = MagicMock()

        embedding = FaceEmbedding(
            face_id="face-002",
            bbox=[10, 20, 100, 200],
            embedding=[0.1] * 512,
            det_score=0.85,
        )
        proc._on_face_embedded(embedding.SerializeToString())

        proc.mq.publish.assert_called_once()
        call_kwargs = proc.mq.publish.call_args.kwargs

        result = FaceRecognized()
        result.ParseFromString(call_kwargs["body"])
        assert result.face_id == "face-002"
        assert result.username == "unknown"
        assert result.score == 0.0

    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_cooldown_suppresses_duplicate_publish(self, MockMQ, MockQdrant):
        """Second message for the same person within cooldown should not
        be published."""
        from main import RecognitionProcessor

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = [{"username": "alice", "score": 0.9}]

        proc = RecognitionProcessor()
        proc.mq = MagicMock()

        embedding = FaceEmbedding(
            face_id="face-003",
            bbox=[10, 20, 100, 200],
            embedding=[0.1] * 512,
            det_score=0.95,
        )
        body = embedding.SerializeToString()

        # First call publishes
        proc._on_face_embedded(body)
        assert proc.mq.publish.call_count == 1

        # Second call within cooldown is suppressed
        proc._on_face_embedded(body)
        assert proc.mq.publish.call_count == 1

    @patch("main.time.monotonic")
    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_cooldown_expires_allows_publish(
        self, MockMQ, MockQdrant, mock_monotonic
    ):
        """After cooldown expires, the same person should be published again."""
        from main import RecognitionProcessor

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = [{"username": "alice", "score": 0.9}]

        proc = RecognitionProcessor()
        proc.mq = MagicMock()

        embedding = FaceEmbedding(
            face_id="face-004",
            bbox=[10, 20, 100, 200],
            embedding=[0.1] * 512,
            det_score=0.95,
        )
        body = embedding.SerializeToString()

        # First call at t=0
        mock_monotonic.return_value = 0.0
        proc._on_face_embedded(body)
        assert proc.mq.publish.call_count == 1

        # Second call at t=10 (within cooldown) — suppressed
        mock_monotonic.return_value = 10.0
        proc._on_face_embedded(body)
        assert proc.mq.publish.call_count == 1

        # Third call at t=61 (after cooldown) — published
        mock_monotonic.return_value = 61.0
        proc._on_face_embedded(body)
        assert proc.mq.publish.call_count == 2


class TestFaceUpdateProcessor:
    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_face_update_new_person(self, MockMQ, MockQdrant):
        from main import FaceUpdateProcessor

        mock_qdrant_inst = MockQdrant.return_value
        mock_qdrant_inst.search.return_value = []  # No exact match found

        proc = FaceUpdateProcessor()
        proc.qdrant = mock_qdrant_inst

        msg = PersonUpdate(
            face_id="person-001",
            username="bob",
            embedding=[0.5] * 512,
        )
        proc._on_face_updated(msg.SerializeToString())

        mock_qdrant_inst.search.assert_called_once()
        mock_qdrant_inst.delete.assert_not_called()
        mock_qdrant_inst.create.assert_called_once()

        call_kwargs = mock_qdrant_inst.create.call_args.kwargs
        assert call_kwargs["username"] == "bob"
        assert call_kwargs["point_id"] == "person-001"

    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_face_update_existing_person(self, MockMQ, MockQdrant):
        from main import FaceUpdateProcessor

        mock_qdrant_inst = MockQdrant.return_value
        mock_qdrant_inst.search.return_value = [
            {"id": "old-uuid-999", "score": 0.995}
        ]

        proc = FaceUpdateProcessor()
        proc.qdrant = mock_qdrant_inst

        msg = PersonUpdate(
            face_id="person-002",
            username="charlie",
            embedding=[0.6] * 512,
        )
        proc._on_face_updated(msg.SerializeToString())

        mock_qdrant_inst.search.assert_called_once()
        mock_qdrant_inst.delete.assert_called_once_with(
            point_ids=["old-uuid-999"]
        )
        mock_qdrant_inst.create.assert_called_once()

        call_kwargs = mock_qdrant_inst.create.call_args.kwargs
        assert call_kwargs["username"] == "charlie"
        assert call_kwargs["point_id"] == "person-002"
