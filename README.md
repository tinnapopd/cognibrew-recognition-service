# Cognibrew Recognition Service

Face-recognition microservice that consumes face-embedding vectors from RabbitMQ, matches them against a Qdrant vector database, and publishes recognition results back to RabbitMQ. All messages are serialised with **Protocol Buffers**.


## Prerequisites

- **Python 3.10+**
- **Docker** (for local RabbitMQ & Qdrant)
- **protoc** (Protocol Buffers compiler — only needed if regenerating stubs)

## Quick Start

### 1. Start Infrastructure

```bash
# Start RabbitMQ (port 5672 + management UI on 15672)
bash scripts/init_rabbitmq.sh

# Start Qdrant (REST 6333 + gRPC 6334)
bash scripts/init_qdrant.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Compile Protobuf (optional — pre-compiled stubs are committed)

```bash
bash scripts/compile_proto.sh
```

### 4. Configure

Copy the example env file and adjust values as needed:

```bash
cp .env.example .env
```

### 5. Run

```bash
PYTHONPATH=src python src/main.py
```

The service starts two consumer threads:

- **RecognitionProcessor** — listens on `face.embedded`, queries Qdrant, publishes `face.recognized`.
- **FaceUpdateProcessor** — listens on `face.updated`, creates / updates / deletes points in Qdrant.

## Docker

### Build

```bash
docker build -t cognibrew-recognition-service .
```

### Run

```bash
docker run --env-file .env cognibrew-recognition-service
```

> When running inside Docker, set `RABBITMQ_HOST=host.docker.internal` (macOS/Windows) or the appropriate network address to reach the RabbitMQ broker.

## Testing

### End-to-End with Mock Publishers

The `init_rabbitmq.sh` script starts RabbitMQ **and** two mock publishers that continuously send fake `FaceEmbedding` and `PersonUpdate` protobuf messages.

```bash
# Terminal 1 — start RabbitMQ + Qdrant + mock publishers
bash scripts/init_qdrant.sh
bash scripts/init_rabbitmq.sh
```

```bash
# Terminal 2 — run the service (native)
PYTHONPATH=src python src/main.py
```

Or via Docker:

```bash
docker build -t cognibrew-recognition-service .
docker run --rm --env-file .env \
  -e QDRANT_HOST=host.docker.internal \
  cognibrew-recognition-service
```

You should see JSON logs for `face_unknown`, `person_created`, `person_updated`, and `person_deleted` events flowing through.

The RabbitMQ management UI is available at [http://localhost:15672](http://localhost:15672) (guest/guest) to inspect queues and messages.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| **RabbitMQ** | | |
| `RABBITMQ_HOST` | `localhost` | Broker hostname |
| `RABBITMQ_PORT` | `5672` | AMQP port |
| `RABBITMQ_USERNAME` | `guest` | Auth username |
| `RABBITMQ_PASSWORD` | `guest` | Auth password |
| `RABBITMQ_INFERENCE_EXCHANGE_NAME` | `cognibrew.inference` | Exchange for recognition flow |
| `RABBITMQ_INFERENCE_QUEUE_NAME` | `cognibrew.inference.face_embedded` | Queue for inbound embeddings |
| `RABBITMQ_FACE_EMBEDDED_ROUTING_KEY` | `face.embedded` | Routing key — inbound embeddings |
| `RABBITMQ_FACE_RECOGNIZED_ROUTING_KEY` | `face.recognized` | Routing key — outbound results |
| `RABBITMQ_FACE_UPDATE_EXCHANGE_NAME` | `cognibrew.vectordb` | Exchange for vector-db sync |
| `RABBITMQ_FACE_UPDATE_QUEUE_NAME` | `cognibrew.vectordb.face_updated` | Queue for person updates |
| `RABBITMQ_FACE_UPDATE_ROUTING_KEY` | `face.updated` | Routing key — person updates |
| **Qdrant** | | |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6334` | Qdrant gRPC port |
| `QDRANT_COLLECTION_NAME` | `face_embeddings` | Collection name |
| `QDRANT_EMBEDDING_DIM` | `512` | Embedding vector dimension |
| **Model** | | |
| `MODEL_SIMILARITY_THRESHOLD` | `0.65` | Min cosine-similarity to consider a match (0–1 exclusive) |

## Project Structure

```
.
├── proto/                          # Protobuf schema definitions
│   ├── face_embedding.proto
│   ├── face_result.proto
│   └── face_update.proto
├── scripts/
│   ├── compile_proto.sh            # Compile .proto → Python
│   ├── init_rabbitmq.sh            # Launch RabbitMQ + mock publishers
│   └── init_qdrant.sh              # Launch Qdrant
├── src/
│   ├── main.py                     # Entrypoint — starts consumer threads
│   ├── core/
│   │   ├── config.py               # Pydantic Settings (env-based config)
│   │   ├── logger.py               # JSON-structured logging
│   │   ├── message_queue.py        # RabbitMQ connection wrapper
│   │   └── vectordb.py             # Qdrant CRUD operations
│   └── schemas/
│       ├── point.py                # Qdrant point data models
│       └── proto/                  # Compiled protobuf stubs (auto-generated)
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .dockerignore
```
