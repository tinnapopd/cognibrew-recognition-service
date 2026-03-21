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
    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_create_action(self, MockMQ, MockQdrant):
        from main import FaceUpdateProcessor

        mock_qdrant = MockQdrant.return_value
        proc = FaceUpdateProcessor()
        proc.qdrant = mock_qdrant

        msg = PersonUpdate(
            person_id="id-1",
            username="bob",
            embedding=[0.5] * 512,
            action=PersonUpdate.CREATE,
        )
        proc._on_face_updated(msg.SerializeToString())

        mock_qdrant.create.assert_called_once()
        call_kwargs = mock_qdrant.create.call_args.kwargs
        assert call_kwargs["username"] == "bob"
        assert call_kwargs["point_id"] == "id-1"

    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_delete_action(self, MockMQ, MockQdrant):
        from main import FaceUpdateProcessor

        mock_qdrant = MockQdrant.return_value
        proc = FaceUpdateProcessor()
        proc.qdrant = mock_qdrant

        msg = PersonUpdate(
            person_id="id-2",
            username="charlie",
            action=PersonUpdate.DELETE,
        )
        proc._on_face_updated(msg.SerializeToString())

        mock_qdrant.delete.assert_called_once_with(point_ids=["id-2"])

    @patch("main.QdrantProcessor")
    @patch("main.MessageQueue")
    def test_update_action(self, MockMQ, MockQdrant):
        from main import FaceUpdateProcessor

        mock_qdrant = MockQdrant.return_value
        proc = FaceUpdateProcessor()
        proc.qdrant = mock_qdrant

        msg = PersonUpdate(
            person_id="id-3",
            username="diana",
            embedding=[0.3] * 512,
            action=PersonUpdate.UPDATE,
        )
        proc._on_face_updated(msg.SerializeToString())

        mock_qdrant.update.assert_called_once()
        call_kwargs = mock_qdrant.update.call_args.kwargs
        assert call_kwargs["point_id"] == "id-3"
        assert call_kwargs["username"] == "diana"
