# Gold Data Schema Alignment Fix Report

**Date**: 2026-01-28
**Status**: ✓ COMPLETED

## Summary

Fixed schema alignment issues in gold data files to match `CausalSchema.py` definitions.

## Files Modified

1. `era-smartfarm-rag/benchmarking/data/causal_extraction_gold.jsonl` (5 lines)
2. `era-smartfarm-rag/benchmarking/data/multihop_gold.jsonl` (5 lines)

## Changes Made

### 1. Entity Type Mappings

Fixed entity types to match CausalSchema.py EntityType enum:

| Old Type | New Type | Count |
|----------|----------|-------|
| `environmental_factor` | `environment` | Multiple |
| `disorder` | `disease` | Multiple |
| `nutrient_disorder` | `nutrient` | Multiple |
| `physiological_effect` | `symptom` | Multiple |
| `secondary_effect` | `symptom` | Multiple |
| `pathogen` | `cause` | 1 |
| `fungal_infection` | `disease` | (if any) |
| `temperature_stress` | `cause` | (if any) |
| `osmotic_pressure` | `cause` | (if any) |

**Valid Types**: crop, disease, symptom, cause, solution, environment, nutrient, practice

### 2. Relation Type Mappings

Fixed relation types to match CausalSchema.py RelationType enum:

| Old Type | New Type | Count |
|----------|----------|-------|
| `manifests_as` | `causes` | Multiple |
| `leads_to` | `causes` | Multiple |
| `increases_risk_of` | `causes` | 1 |
| `promotes` | `causes` | 1 |

**Valid Types**: causes, prevents, treats, requires, associated_with

### 3. Relation Structure Fix

**Before** (canonical_id reference):
```json
{
  "relations": [{
    "source": "ENV_TEMP_LOW",
    "target": "DIS_COLD_DAMAGE",
    "type": "causes",
    "evidence": "..."
  }]
}
```

**After** (full entity object):
```json
{
  "relations": [{
    "source": {
      "text": "저온",
      "type": "environment",
      "canonical_id": "ENV_TEMP_LOW"
    },
    "target": {
      "text": "저온 장해",
      "type": "disease",
      "canonical_id": "DIS_COLD_DAMAGE"
    },
    "type": "causes",
    "evidence_text": "..."
  }]
}
```

### 4. Multihop Field Rename

Fixed `multihop_gold.jsonl` to use 0-indexed `hop_index`:

**Before**:
```json
{"hop_num": 1, "query": "...", "expected_info": "..."}
```

**After**:
```json
{"hop_index": 0, "query": "...", "expected_info": "..."}
```

## Validation Results

All schema checks passed:

```
=== Checking causal_extraction_gold.jsonl ===
Causal extraction: OK

=== Checking multihop_gold.jsonl ===
Multihop: OK

All schema checks passed!
```

### Validated Aspects

1. ✓ Entity types match CausalSchema.py EntityType enum
2. ✓ Relation types match CausalSchema.py RelationType enum
3. ✓ Relations use full entity objects (not canonical_id strings)
4. ✓ Field renamed from `evidence` to `evidence_text`
5. ✓ Multihop uses `hop_index` (0-indexed) instead of `hop_num`

## Example Fixed Data

### Causal Extraction Example (Line 1)

**Entities**:
- 저온 (environment) → ENV_TEMP_LOW
- 저온 장해 (disease) → DIS_COLD_DAMAGE
- 잎 황화 (symptom) → SYM_LEAF_YELLOWING
- 생육 지연 (symptom) → SYM_GROWTH_DELAY
- 기형과 (symptom) → SYM_FRUIT_DEFORMITY

**Relations**:
- 저온 --[causes]--> 저온 장해
- 저온 장해 --[causes]--> 잎 황화
- 저온 장해 --[causes]--> 생육 지연
- 저온 --[causes]--> 기형과

### Multihop Example (Line 2)

```json
{
  "question_id": "mh_002",
  "question_type": "bridge",
  "num_hops": 2,
  "gold_hops": [
    {"hop_index": 0, "query": "...", "expected_info": "..."},
    {"hop_index": 1, "query": "...", "expected_info": "..."}
  ]
}
```

## Implementation

Script: `fix_gold_data_schema.py`

```python
# Key transformations:
1. Entity type mapping via ENTITY_TYPE_MAPPING dict
2. Relation type mapping via RELATION_TYPE_MAPPING dict
3. Relation source/target expansion using entity_map lookup
4. Field rename: evidence → evidence_text
5. Field rename: hop_num → hop_index (with 0-indexing)
```

## Next Steps

- ✓ Schema alignment complete
- [ ] Run benchmarking suite to verify compatibility
- [ ] Update any code that expects old schema format (if any)

## Files

- Fix script: `c:\Users\yskim\Project\smartfarm-workspace-1\fix_gold_data_schema.py`
- Modified data:
  - `era-smartfarm-rag\benchmarking\data\causal_extraction_gold.jsonl`
  - `era-smartfarm-rag\benchmarking\data\multihop_gold.jsonl`
