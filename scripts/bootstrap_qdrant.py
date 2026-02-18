from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap Qdrant collection for ERA-SmartFarm")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6333)
    p.add_argument("--collection", default="smartfarm_chunks")
    p.add_argument("--dim", type=int, default=512)
    args = p.parse_args()

    try:
        from qdrant_client import QdrantClient, models
    except Exception as exc:
        raise SystemExit(f"qdrant-client is required: {exc}")

    client = QdrantClient(url=f"http://{args.host}:{args.port}", timeout=10.0)

    if client.collection_exists(args.collection):
        print(f"[bootstrap-qdrant] collection already exists: {args.collection}")
        return 0

    client.create_collection(
        collection_name=args.collection,
        vectors_config={
            "dense_text": models.VectorParams(size=int(args.dim), distance=models.Distance.COSINE),
            "dense_image": models.VectorParams(size=int(args.dim), distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    print(f"[bootstrap-qdrant] created: {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
