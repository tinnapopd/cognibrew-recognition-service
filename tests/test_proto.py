"""Tests for Protobuf serialisation round-trips."""

from schemas.proto.face_embedding_pb2 import FaceEmbedding  # type: ignore
from schemas.proto.face_result_pb2 import FaceRecognized, FaceUnknown  # type: ignore
from schemas.proto.person_update_pb2 import PersonUpdate  # type: ignore


class TestFaceEmbeddingProto:
    def test_round_trip(self) -> None:
        original = FaceEmbedding(
            bbox=[10, 20, 100, 200],
            embedding=[0.1] * 512,
            det_score=0.95,
        )
        data = original.SerializeToString()

        parsed = FaceEmbedding()
        parsed.ParseFromString(data)

        assert list(parsed.bbox) == [10, 20, 100, 200]
        assert len(parsed.embedding) == 512
        assert abs(parsed.det_score - 0.95) < 1e-5


class TestFaceResultProto:
    def test_face_recognized_round_trip(self) -> None:
        original = FaceRecognized(
            username="alice",
            score=0.87,
            bbox=[10, 20, 100, 200],
        )
        data = original.SerializeToString()

        parsed = FaceRecognized()
        parsed.ParseFromString(data)

        assert parsed.username == "alice"
        assert abs(parsed.score - 0.87) < 1e-5
        assert list(parsed.bbox) == [10, 20, 100, 200]

    def test_face_unknown_round_trip(self) -> None:
        original = FaceUnknown(bbox=[30, 40, 150, 250])
        data = original.SerializeToString()

        parsed = FaceUnknown()
        parsed.ParseFromString(data)

        assert list(parsed.bbox) == [30, 40, 150, 250]


class TestPersonUpdateProto:
    def test_create_round_trip(self) -> None:
        original = PersonUpdate(
            person_id="abc-123",
            username="bob",
            embedding=[0.5] * 512,
            action=PersonUpdate.CREATE,
        )
        data = original.SerializeToString()

        parsed = PersonUpdate()
        parsed.ParseFromString(data)

        assert parsed.person_id == "abc-123"
        assert parsed.username == "bob"
        assert len(parsed.embedding) == 512
        assert parsed.action == PersonUpdate.CREATE

    def test_update_round_trip(self) -> None:
        original = PersonUpdate(
            person_id="abc-123",
            username="bob-updated",
            embedding=[0.6] * 512,
            action=PersonUpdate.UPDATE,
        )
        data = original.SerializeToString()

        parsed = PersonUpdate()
        parsed.ParseFromString(data)

        assert parsed.action == PersonUpdate.UPDATE
        assert parsed.username == "bob-updated"

    def test_delete_round_trip(self) -> None:
        original = PersonUpdate(
            person_id="abc-123",
            action=PersonUpdate.DELETE,
        )
        data = original.SerializeToString()

        parsed = PersonUpdate()
        parsed.ParseFromString(data)

        assert parsed.action == PersonUpdate.DELETE
        assert parsed.person_id == "abc-123"
        assert parsed.username == ""  # default for unset string
        assert len(parsed.embedding) == 0
