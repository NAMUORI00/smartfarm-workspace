#!/usr/bin/env bash
set -euo pipefail

QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_URL="${QDRANT_URL:-http://${QDRANT_HOST}:${QDRANT_PORT}}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-smartfarm_chunks_v2}"
EMBED_DIM="${EMBED_DIM:-512}"

echo "[reindex] qdrant=${QDRANT_URL} collection=${QDRANT_COLLECTION} embed_dim=${EMBED_DIM}"
echo "[reindex] deleting collection (ignore if absent)"
curl -fsS -X DELETE "${QDRANT_URL}/collections/${QDRANT_COLLECTION}" >/dev/null 2>&1 || true

echo "[reindex] creating collection with named vectors: dense_text/dense_image + sparse"
curl -fsS -X PUT "${QDRANT_URL}/collections/${QDRANT_COLLECTION}" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "vectors": {
    "dense_text": {"size": ${EMBED_DIM}, "distance": "Cosine"},
    "dense_image": {"size": ${EMBED_DIM}, "distance": "Cosine"}
  },
  "sparse_vectors": {
    "sparse": {"modifier": "idf"}
  }
}
JSON

echo
echo "[reindex] collection info:"
curl -fsS "${QDRANT_URL}/collections/${QDRANT_COLLECTION}"
echo
