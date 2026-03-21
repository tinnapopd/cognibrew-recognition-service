from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RabbitMQConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=False,
        env_prefix="RABBITMQ_",
        case_sensitive=False,
    )

    HOST: str = Field(default="localhost")
    PORT: int = Field(default=5672)
    USERNAME: str = Field(default="guest")
    PASSWORD: str = Field(default="guest")

    # Recognition (Inference Server -> Recognition Service)
    INFERENCE_EXCHANGE_NAME: str = Field(default="cognibrew.inference")
    INFERENCE_QUEUE_NAME: str = Field(
        default="cognibrew.inference.face_embedded"
    )

    # Routing keys – inbound
    FACE_EMBEDDED_ROUTING_KEY: str = Field(default="face.embedded")

    # Routing keys – outbound (recognition results)
    FACE_RECOGNIZED_ROUTING_KEY: str = Field(default="face.recognized")
    FACE_UNKNOWN_ROUTING_KEY: str = Field(default="face.unknown")

    # Person sync (cloud --> vectordb)
    PERSON_SYNC_EXCHANGE_NAME: str = Field(default="cognibrew.vectordb")
    PERSON_SYNC_QUEUE_NAME: str = Field(
        default="cognibrew.vectordb.person_updated"
    )
    PERSON_SYNC_ROUTING_KEY: str = Field(default="person.updated")


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=False,
        env_prefix="QDRANT_",
        case_sensitive=False,
    )

    HOST: str = Field(default="localhost")
    PORT: int = Field(default=6334)
    COLLECTION_NAME: str = Field(default="face_embeddings")
    EMBEDDING_DIM: int = Field(default=512)


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=False,
        env_prefix="MODEL_",
        case_sensitive=False,
    )

    SIMILARITY_THRESHOLD: float = Field(default=0.65)

    @field_validator("SIMILARITY_THRESHOLD")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError(
                f"SIMILARITY_THRESHOLD must be between 0 and 1 (exclusive), got {v}"
            )
        return v


class Settings:
    """Main configuration class aggregating all settings."""

    def __init__(self) -> None:
        self.rabbitmq = RabbitMQConfig()
        self.qdrant = QdrantConfig()
        self.model = ModelConfig()


# Module-level singleton instance
settings = Settings()
