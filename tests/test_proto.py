"""Tests for protobuf serialisation round-trips."""

from schemas.proto.face_embedding_pb2 import FaceEmbedding  # type: ignore
from schemas.proto.face_result_pb2 import FaceRecognized  # type: ignore
from schemas.proto.face_update_pb2 import PersonUpdate  # type: ignore


class TestFaceEmbeddingProto:
    def test_round_trip(self):
        original = FaceEmbedding(
            bbox=[10, 20, 100, 200],
            embedding=[0.1, 0.2, 0.3],
            det_score=0.95,
        )
        data = original.SerializeToString()
        restored = FaceEmbedding()
        restored.ParseFromString(data)

        assert list(restored.bbox) == [10, 20, 100, 200]
        assert restored.embedding[:3] == [
            pytest.approx(0.1, abs=1e-5),
            pytest.approx(0.2, abs=1e-5),
            pytest.approx(0.3, abs=1e-5),
        ]
        assert restored.det_score == pytest.approx(0.95, abs=1e-5)

    def test_empty_message(self):
        msg = FaceEmbedding()
        assert list(msg.bbox) == []
        assert list(msg.embedding) == []
        assert msg.det_score == 0.0


class TestFaceRecognizedProto:
    def test_round_trip(self):
        original = FaceRecognized(
            bbox=[50, 60, 200, 300],
            username="alice",
            score=0.92,
        )
        data = original.SerializeToString()
        restored = FaceRecognized()
        restored.ParseFromString(data)

        assert list(restored.bbox) == [50, 60, 200, 300]
        assert restored.username == "alice"
        assert restored.score == pytest.approx(0.92, abs=1e-5)


class TestPersonUpdateProto:
    def test_create_action(self):
        msg = PersonUpdate(
            person_id="abc-123",
            username="bob",
            embedding=[0.5] * 512,
            action=PersonUpdate.CREATE,
        )
        data = msg.SerializeToString()
        restored = PersonUpdate()
        restored.ParseFromString(data)

        assert restored.person_id == "abc-123"
        assert restored.username == "bob"
        assert restored.action == PersonUpdate.CREATE
        assert len(restored.embedding) == 512

    def test_delete_action_no_embedding(self):
        msg = PersonUpdate(
            person_id="xyz-789",
            username="charlie",
            action=PersonUpdate.DELETE,
        )
        data = msg.SerializeToString()
        restored = PersonUpdate()
        restored.ParseFromString(data)

        assert restored.action == PersonUpdate.DELETE
        assert len(restored.embedding) == 0

    def test_action_enum_values(self):
        assert PersonUpdate.CREATE == 0
        assert PersonUpdate.UPDATE == 1
        assert PersonUpdate.DELETE == 2


import pytest  # noqa: E402 – needed for pytest.approx above
