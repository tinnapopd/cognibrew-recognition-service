#!/usr/bin/env bash

# Compile .proto files into Python modules.
# Usage: bash scripts/compile_proto.sh

set -euo pipefail

PROTO_DIR="proto"
OUT_DIR="src/schemas/proto"

mkdir -p "${OUT_DIR}"

protoc \
  --python_out="${OUT_DIR}" \
  --proto_path="${PROTO_DIR}" \
  "${PROTO_DIR}"/face_embedding.proto \
  "${PROTO_DIR}"/face_update.proto \
  "${PROTO_DIR}"/face_result.proto

# Create __init__.py so the directory is importable
touch "${OUT_DIR}/__init__.py"

echo "Proto compiled to ${OUT_DIR}/"
