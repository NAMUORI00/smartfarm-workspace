"""Pre-flight check: verify ALL ontology-refactored modules import cleanly."""
import sys, os

print("=" * 60)
print("PRE-FLIGHT CHECK: Ontology-Driven Module Import Validation")
print("=" * 60)

errors = []

# ── smartfarm-search modules ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "smartfarm-search"))

print("\n[smartfarm-search]")
modules_search = [
    ("core.Config.ontology", "Ontology Loader"),
    ("core.retrieval.fusion_profile", "DAT Fusion Profile"),
    ("core.retrieval.falkordb_client", "FalkorDB Client"),
    ("core.retrieval.private_extractor", "Private Extractor"),
    ("core.contracts.storage", "Storage Contract"),
    ("core.retrieval.fusion_policy", "Fusion Policy"),
    ("core.retrieval.qdrant_client", "Qdrant Client"),
]

for mod_name, label in modules_search:
    try:
        __import__(mod_name)
        print(f"  ✓ {label} ({mod_name})")
    except Exception as e:
        print(f"  ✗ {label} ({mod_name}): {e}")
        errors.append(f"search:{mod_name}")

# ── smartfarm-ingest modules ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "smartfarm-ingest"))

print("\n[smartfarm-ingest]")
modules_ingest = [
    ("pipeline.ontology", "Ontology Loader (ingest)"),
    ("pipeline.ontology_builder", "Ontology Builder"),
    ("pipeline.llm_extractor", "LLM Extractor"),
    ("pipeline.kg_writer", "KG Writer"),
    ("pipeline.public_ingest_runner", "Public Ingest Runner"),
]

for mod_name, label in modules_ingest:
    try:
        __import__(mod_name)
        print(f"  ✓ {label} ({mod_name})")
    except Exception as e:
        print(f"  ✗ {label} ({mod_name}): {e}")
        errors.append(f"ingest:{mod_name}")

# ── ontology data validation ─────────────────────────────────────────
print("\n[domain_ontology.json]")
from core.Config.ontology import (
    RELATION_HINT_TOKENS, ALLOWED_ENTITY_LABELS,
    ALLOWED_RELATION_TYPES, ALLOWED_MODALITIES, SENSOR_THRESHOLDS,
    ALLOWED_ENTITY_LABELS_CSV, ALLOWED_RELATION_TYPES_CSV,
)
checks = [
    ("RELATION_HINT_TOKENS", RELATION_HINT_TOKENS, 5),
    ("ALLOWED_ENTITY_LABELS", ALLOWED_ENTITY_LABELS, 3),
    ("ALLOWED_RELATION_TYPES", ALLOWED_RELATION_TYPES, 3),
    ("ALLOWED_MODALITIES", ALLOWED_MODALITIES, 2),
    ("SENSOR_THRESHOLDS", SENSOR_THRESHOLDS, 2),
]
for name, val, min_count in checks:
    count = len(val)
    ok = count >= min_count
    sym = "✓" if ok else "✗"
    print(f"  {sym} {name}: {count} items")
    if not ok:
        errors.append(f"ontology:{name}")

print(f"\n  CSV Entity Labels:   {ALLOWED_ENTITY_LABELS_CSV}")
print(f"  CSV Relation Types:  {ALLOWED_RELATION_TYPES_CSV}")

# ── fusion weights runtime check ─────────────────────────────────────
print("\n[fusion_weights.runtime.json]")
import json
fw_path = os.path.join(os.path.dirname(__file__), "data", "artifacts", "fusion_weights.runtime.json")
try:
    fw = json.loads(open(fw_path, encoding="utf-8").read())
    n_seg = len(fw.get("segments", []))
    print(f"  ✓ Loaded: {n_seg} segments, schema={fw.get('schema_version')}")
except Exception as e:
    print(f"  ✗ Load failed: {e}")
    errors.append("fusion_weights")

# ── summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"PREFLIGHT FAILED — {len(errors)} error(s): {errors}")
    sys.exit(1)
else:
    print("ALL PREFLIGHT CHECKS PASSED ✓")
    print("System is ready for experiment execution.")
    sys.exit(0)
