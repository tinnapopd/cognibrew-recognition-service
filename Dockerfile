# Stage 1: compile protobuf
FROM python:3.10.19-slim AS builder

RUN pip install --no-cache-dir protobuf grpcio-tools

WORKDIR /build
COPY proto/ proto/
RUN mkdir -p src/schemas/proto \
    && python -m grpc_tools.protoc \
    --python_out=src/schemas/proto \
    --proto_path=proto \
    proto/face_embedding.proto \
    proto/face_update.proto \
    proto/face_result.proto \
    && touch src/schemas/proto/__init__.py

# Stage 2: runtime
FROM python:3.10.19-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/

# Copy compiled protobuf modules from builder
COPY --from=builder /build/src/schemas/proto/ src/schemas/proto/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/main.py"]
