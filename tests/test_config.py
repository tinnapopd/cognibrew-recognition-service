"""Tests for core.config - Settings loading and validation."""

import os
from unittest.mock import patch

import pytest

from core.config import ModelConfig, QdrantConfig, RabbitMQConfig, Settings


class TestRabbitMQConfig:
    def test_defaults(self):
        cfg = RabbitMQConfig()
        assert cfg.HOST == "localhost"
        assert cfg.PORT == 5672
        assert cfg.USERNAME == "guest"
        assert cfg.PASSWORD == "guest"
        assert cfg.INFERENCE_EXCHANGE_NAME == "cognibrew.inference"
        assert cfg.INFERENCE_QUEUE_NAME == "cognibrew.inference.face_embedded"
        assert cfg.FACE_EMBEDDED_ROUTING_KEY == "face.embedded"
        assert cfg.FACE_RECOGNIZED_ROUTING_KEY == "face.recognized"
        assert cfg.FACE_UPDATE_EXCHANGE_NAME == "cognibrew.vectordb"
        assert cfg.FACE_UPDATE_QUEUE_NAME == "cognibrew.vectordb.face_updated"
        assert cfg.FACE_UPDATE_ROUTING_KEY == "face.updated"

    def test_env_override(self):
        with patch.dict(os.environ, {"RABBITMQ_HOST": "broker.example.com"}):
            cfg = RabbitMQConfig()
            assert cfg.HOST == "broker.example.com"


class TestQdrantConfig:
    def test_defaults(self):
        cfg = QdrantConfig()
        assert cfg.HOST == "localhost"
        assert cfg.PORT == 6334
        assert cfg.COLLECTION_NAME == "face_embeddings"
        assert cfg.EMBEDDING_DIM == 512

    def test_env_override(self):
        with patch.dict(os.environ, {"QDRANT_PORT": "9999"}):
            cfg = QdrantConfig()
            assert cfg.PORT == 9999


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.SIMILARITY_THRESHOLD == 0.65

    def test_env_override(self):
        with patch.dict(os.environ, {"MODEL_SIMILARITY_THRESHOLD": "0.8"}):
            cfg = ModelConfig()
            assert cfg.SIMILARITY_THRESHOLD == 0.8

    def test_invalid_threshold_zero(self):
        with patch.dict(os.environ, {"MODEL_SIMILARITY_THRESHOLD": "0"}):
            with pytest.raises(ValueError, match="between 0 and 1"):
                ModelConfig()

    def test_invalid_threshold_one(self):
        with patch.dict(os.environ, {"MODEL_SIMILARITY_THRESHOLD": "1"}):
            with pytest.raises(ValueError, match="between 0 and 1"):
                ModelConfig()

    def test_invalid_threshold_negative(self):
        with patch.dict(os.environ, {"MODEL_SIMILARITY_THRESHOLD": "-0.5"}):
            with pytest.raises(ValueError, match="between 0 and 1"):
                ModelConfig()


class TestSettings:
    def test_creates_all_configs(self):
        s = Settings()
        assert isinstance(s.rabbitmq, RabbitMQConfig)
        assert isinstance(s.qdrant, QdrantConfig)
        assert isinstance(s.model, ModelConfig)
