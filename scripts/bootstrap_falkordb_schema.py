from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap FalkorDB graph schema")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--graph", default="smartfarm")
    p.add_argument("--schema", default="schema.cypher")
    args = p.parse_args()

    try:
        import redis
    except Exception as exc:
        raise SystemExit(f"redis package is required: {exc}")

    query = Path(args.schema).read_text(encoding="utf-8")
    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    r.execute_command("GRAPH.QUERY", args.graph, query, "--compact")
    print(f"[bootstrap-falkordb] applied schema to graph={args.graph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
