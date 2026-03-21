#!/usr/bin/env bash

set -x
set -eo pipefail

if ! [ -x "$(command -v docker)" ]; then
    echo >&2 "Error: Docker is not installed."
    exit 1
fi

# RabbitMQ configuration
RABBITMQ_PORT=${RABBITMQ_PORT:-5672}
RABBITMQ_MANAGEMENT_PORT=${RABBITMQ_MANAGEMENT_PORT:-15672}
RABBITMQ_DEFAULT_USER=${RABBITMQ_DEFAULT_USER:-guest}
RABBITMQ_DEFAULT_PASS=${RABBITMQ_DEFAULT_PASS:-guest}

# Recognition (inference server --> recognition service)
RABBITMQ_INFERENCE_EXCHANGE=${RABBITMQ_INFERENCE_EXCHANGE:-cognibrew.inference}
RABBITMQ_FACE_EMBEDDED_ROUTING_KEY=${RABBITMQ_FACE_EMBEDDED_ROUTING_KEY:-face.embedded}

# Person sync (cloud --> vectordb)
RABBITMQ_PERSON_SYNC_EXCHANGE=${RABBITMQ_PERSON_SYNC_EXCHANGE:-cognibrew.vectordb}
RABBITMQ_PERSON_SYNC_ROUTING_KEY=${RABBITMQ_PERSON_SYNC_ROUTING_KEY:-person.updated}


# Launch RabbitMQ using Docker
# Allow to skip docker if a dockerized RabbitMQ is already running
# Use: SKIP_DOCKER=1 ./scripts/init_rabbitmq.sh
if [[ -z "${SKIP_DOCKER}" ]]; then
    # Remove any previous RabbitMQ docker container
    docker rm -f rabbitmq || true
    docker run \
        --name rabbitmq \
        -p "${RABBITMQ_PORT}":5672 \
        -p "${RABBITMQ_MANAGEMENT_PORT}":15672 \
        -e RABBITMQ_DEFAULT_USER="${RABBITMQ_DEFAULT_USER}" \
        -e RABBITMQ_DEFAULT_PASS="${RABBITMQ_DEFAULT_PASS}" \
        -d rabbitmq:3-management
fi

# Keep pinging RabbitMQ until it's ready
until curl -sf "http://localhost:${RABBITMQ_MANAGEMENT_PORT}/api/healthchecks/node" -u "${RABBITMQ_DEFAULT_USER}:${RABBITMQ_DEFAULT_PASS}" > /dev/null 2>&1; do
    >&2 echo "RabbitMQ is still unavailable - sleeping"
    sleep 5
done

>&2 echo "RabbitMQ is up and running on port ${RABBITMQ_PORT} (AMQP), ${RABBITMQ_MANAGEMENT_PORT} (Management), ready to go!"

# Clean up background jobs and Docker container on Ctrl+C
trap 'kill $(jobs -p) 2>/dev/null; docker rm -f rabbitmq 2>/dev/null; echo "Stopped all mock publishers and removed RabbitMQ container."; exit 0' INT TERM

# Mock 1: FaceEmbedding (inference --> recognition)
>&2 echo "[mock] FaceEmbedding → ${RABBITMQ_INFERENCE_EXCHANGE}/${RABBITMQ_FACE_EMBEDDED_ROUTING_KEY}"

PYTHONPATH=src \
  MQ_PORT="${RABBITMQ_PORT}" \
  MQ_USER="${RABBITMQ_DEFAULT_USER}" \
  MQ_PASS="${RABBITMQ_DEFAULT_PASS}" \
  MQ_EXCHANGE="${RABBITMQ_INFERENCE_EXCHANGE}" \
  MQ_ROUTING_KEY="${RABBITMQ_FACE_EMBEDDED_ROUTING_KEY}" \
python3 - <<'FACE_MOCK' &
import os, random, sys, time, pika
from schemas.proto.face_embedding_pb2 import FaceEmbedding

port = int(os.environ.get("MQ_PORT", "5672"))
user = os.environ.get("MQ_USER", "guest")
passwd = os.environ.get("MQ_PASS", "guest")
exchange = os.environ.get("MQ_EXCHANGE", "cognibrew.inference")
routing_key = os.environ.get("MQ_ROUTING_KEY", "face.embedded")

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="localhost", port=port,
        credentials=pika.PlainCredentials(user, passwd))
)
channel = connection.channel()
channel.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
queue = os.environ.get("MQ_QUEUE", "cognibrew.inference.face_embedded")
channel.queue_declare(queue=queue, durable=True)
channel.queue_bind(queue=queue, exchange=exchange, routing_key=routing_key)

count = 0
while True:
    msg = FaceEmbedding(
        bbox=[random.randint(0, 500), random.randint(0, 500),
              random.randint(500, 1000), random.randint(500, 1000)],
        embedding=[random.uniform(-1, 1) for _ in range(512)],
        det_score=round(random.uniform(0.7, 1.0), 2),
    )
    channel.basic_publish(
        exchange=exchange, routing_key=routing_key,
        body=msg.SerializeToString(),
        properties=pika.BasicProperties(content_type="application/x-protobuf", delivery_mode=2),
    )
    count += 1
    bbox = "[" + ", ".join(f"{v:4d}" for v in msg.bbox) + "]"
    print(f"[FaceEmbedding  #{count:<4}] bbox={bbox}, det_score={msg.det_score:.2f}")
    time.sleep(1)
FACE_MOCK

# Mock 2: PersonUpdate (cloud --> vectordb)
>&2 echo "[mock] PersonUpdate → ${RABBITMQ_PERSON_SYNC_EXCHANGE}/${RABBITMQ_PERSON_SYNC_ROUTING_KEY}"

PYTHONPATH=src \
  MQ_PORT="${RABBITMQ_PORT}" \
  MQ_USER="${RABBITMQ_DEFAULT_USER}" \
  MQ_PASS="${RABBITMQ_DEFAULT_PASS}" \
  MQ_EXCHANGE="${RABBITMQ_PERSON_SYNC_EXCHANGE}" \
  MQ_ROUTING_KEY="${RABBITMQ_PERSON_SYNC_ROUTING_KEY}" \
python3 - <<'PERSON_MOCK' &
import os, random, string, sys, time, uuid, pika
from schemas.proto.person_update_pb2 import PersonUpdate

port = int(os.environ.get("MQ_PORT", "5672"))
user = os.environ.get("MQ_USER", "guest")
passwd = os.environ.get("MQ_PASS", "guest")
exchange = os.environ.get("MQ_EXCHANGE", "cognibrew.vectordb")
routing_key = os.environ.get("MQ_ROUTING_KEY", "person.updated")

MOCK_NAMES = ["alice", "bob", "charlie", "diana", "eve", "frank"]

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="localhost", port=port,
        credentials=pika.PlainCredentials(user, passwd))
)
channel = connection.channel()
channel.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
queue = os.environ.get("MQ_QUEUE", "cognibrew.vectordb.person_updated")
channel.queue_declare(queue=queue, durable=True)
channel.queue_bind(queue=queue, exchange=exchange, routing_key=routing_key)

count = 0
while True:
    action = random.choice([PersonUpdate.CREATE, PersonUpdate.UPDATE, PersonUpdate.DELETE])
    action_name = {0: "CREATE", 1: "UPDATE", 2: "DELETE"}[action]
    msg = PersonUpdate(
        person_id=str(uuid.uuid4()),
        username=random.choice(MOCK_NAMES),
        embedding=[random.uniform(-1, 1) for _ in range(512)] if action != PersonUpdate.DELETE else [],
        action=action,
    )
    channel.basic_publish(
        exchange=exchange, routing_key=routing_key,
        body=msg.SerializeToString(),
        properties=pika.BasicProperties(content_type="application/x-protobuf", delivery_mode=2),
    )
    count += 1
    print(f"[PersonUpdate   #{count:<4}] action={action_name:<6}, username={msg.username:<8}, id={msg.person_id[:8]}…")
    time.sleep(3)
PERSON_MOCK

>&2 echo "Both mock publishers running (Ctrl+C to stop)"
wait