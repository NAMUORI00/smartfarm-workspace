from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "smartfarm-ingest"))
from pipeline.artifact_export import build_artifact_manifest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".venv",
            "output",
            "*.pyc",
            "*.pyo",
            ".git",
        ),
    )


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def write_bundle_manifest(bundle_root: Path, runtime_paths: list[Path]) -> Path:
    files = []
    for root in runtime_paths:
        if root.is_file():
            files.append(
                {
                    "path": str(root.relative_to(bundle_root)).replace("\\", "/"),
                    "size_bytes": int(root.stat().st_size),
                    "sha256": sha256(root),
                }
            )
            continue
        for path in iter_files(root):
            files.append(
                {
                    "path": str(path.relative_to(bundle_root)).replace("\\", "/"),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256(path),
                }
            )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_workspace": str(ROOT),
        "runtime_files": files,
    }
    out = bundle_root / "bundle_manifest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle_root / "bundle_manifest.sha256").write_text(f"{sha256(out)}  bundle_manifest.json\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Jetson RQ3 bundle")
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "jetson_bundle"))
    parser.add_argument("--qdrant-dir", default=str(ROOT / "data" / "index" / "qdrant"))
    parser.add_argument("--falkordb-dir", default=str(ROOT / "data" / "index" / "falkordb"))
    parser.add_argument("--weights-file", default=str(ROOT / "data" / "artifacts" / "fusion_weights.runtime.json"))
    parser.add_argument("--meta-file", default=str(ROOT / "data" / "artifacts" / "fusion_profile_meta.runtime.json"))
    parser.add_argument("--llm-model", default=str(ROOT / "smartfarm-llm-inference" / "models" / "Qwen3-4B-Q4_K_M.gguf"))
    parser.add_argument("--embed-dir", default=str(ROOT / "smartfarm-search" / "models" / "embeddings" / "BAAI__bge-m3"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    qdrant_dir = require(Path(args.qdrant_dir).resolve(), "Qdrant index")
    falkordb_dir = require(Path(args.falkordb_dir).resolve(), "FalkorDB index")
    weights_file = require(Path(args.weights_file).resolve(), "fusion weights artifact")
    meta_file = require(Path(args.meta_file).resolve(), "fusion profile metadata")
    llm_model = require(Path(args.llm_model).resolve(), "GGUF model")
    require(Path(args.embed_dir).resolve(), "embedding model directory")

    smartfarm_search = require(ROOT / "smartfarm-search", "smartfarm-search source")
    benchmarking = require(ROOT / "smartfarm-benchmarking", "smartfarm-benchmarking source")
    require(TEMPLATE_DIR, "deploy/jetson templates")

    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(TEMPLATE_DIR, out_dir / "deploy" / "jetson")
    copy_tree(smartfarm_search, out_dir / "smartfarm-search")
    copy_tree(benchmarking, out_dir / "smartfarm-benchmarking")

    model_dst = out_dir / "smartfarm-llm-inference" / "models"
    model_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(llm_model, model_dst / llm_model.name)

    qdrant_dst = out_dir / "data" / "index" / "qdrant"
    falkordb_dst = out_dir / "data" / "index" / "falkordb"
    copy_tree(qdrant_dir, qdrant_dst)
    copy_tree(falkordb_dir, falkordb_dst)

    artifacts_dst = out_dir / "data" / "artifacts"
    artifacts_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_file, artifacts_dst / "fusion_weights.runtime.json")
    shutil.copy2(meta_file, artifacts_dst / "fusion_profile_meta.runtime.json")

    export_dir = out_dir / "data" / "index" / "export"
    build_artifact_manifest(
        qdrant_dir=qdrant_dst,
        falkordb_dir=falkordb_dst,
        output_dir=export_dir,
        model_id="BAAI/bge-m3",
        graph_name="smartfarm",
    )

    manifest = write_bundle_manifest(
        out_dir,
        [
            qdrant_dst,
            falkordb_dst,
            artifacts_dst / "fusion_weights.runtime.json",
            artifacts_dst / "fusion_profile_meta.runtime.json",
            model_dst / llm_model.name,
            out_dir / "smartfarm-search" / "models" / "embeddings" / "BAAI__bge-m3",
        ],
    )

    print(f"[jetson-bundle] wrote {out_dir}")
    print(f"[jetson-bundle] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"[jetson-bundle] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
