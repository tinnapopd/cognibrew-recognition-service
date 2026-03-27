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
        assert result.username == "alice"

    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_face_unknown_no_publish(self, MockMQ, MockQdrant):
        """When Qdrant returns no match, nothing should be published."""
        from main import RecognitionProcessor

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = []

        proc = RecognitionProcessor()
        proc.mq = MagicMock()

        embedding = FaceEmbedding(
            bbox=[10, 20, 100, 200],
            embedding=[0.1] * 512,
            det_score=0.85,
        )
        proc._on_face_embedded(embedding.SerializeToString())

        proc.mq.publish.assert_not_called()


class TestFaceUpdateProcessor:
    @patch("main.uuid.uuid4")
    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_face_update_new_person(self, MockMQ, MockQdrant, mock_uuid):
        from main import FaceUpdateProcessor

        mock_uuid.return_value = "new-uuid"
        mock_qdrant_inst = MockQdrant.return_value
        mock_qdrant_inst.search.return_value = []  # No exact match found

        proc = FaceUpdateProcessor()
        proc.qdrant = mock_qdrant_inst

        msg = PersonUpdate(
            username="bob",
            embedding=[0.5] * 512,
        )
        proc._on_face_updated(msg.SerializeToString())

        mock_qdrant_inst.search.assert_called_once()
        mock_qdrant_inst.delete.assert_not_called()
        mock_qdrant_inst.create.assert_called_once()

        call_kwargs = mock_qdrant_inst.create.call_args.kwargs
        assert call_kwargs["username"] == "bob"
        assert call_kwargs["point_id"] == "new-uuid"

    @patch("main.uuid.uuid4")
    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_face_update_existing_person(self, MockMQ, MockQdrant, mock_uuid):
        from main import FaceUpdateProcessor

        mock_uuid.return_value = "new-uuid"
        mock_qdrant_inst = MockQdrant.return_value
        mock_qdrant_inst.search.return_value = [{"id": "old-uuid-999", "score": 0.995}]

        proc = FaceUpdateProcessor()
        proc.qdrant = mock_qdrant_inst

        msg = PersonUpdate(
            username="charlie",
            embedding=[0.6] * 512,
        )
        proc._on_face_updated(msg.SerializeToString())

        mock_qdrant_inst.search.assert_called_once()
        mock_qdrant_inst.delete.assert_called_once_with(point_ids=["old-uuid-999"])
        mock_qdrant_inst.create.assert_called_once()

        call_kwargs = mock_qdrant_inst.create.call_args.kwargs
        assert call_kwargs["username"] == "charlie"
        assert call_kwargs["point_id"] == "new-uuid"
