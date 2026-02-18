// ERA-SmartFarm canonical graph schema bootstrap

// Entity roots (labels are created lazily in FalkorDB).

// Optional seed constraints can be managed externally.

// Relationship type reference:
// CAUSES, TREATED_BY, REQUIRES, SUSCEPTIBLE_TO, AFFECTS, MENTIONS, PART_OF, BELONGS_TO, HAS_CHUNK

// Seed node to verify connectivity.
MERGE (r:SystemRoot {name:'smartfarm', version:'v1'})
SET r.updated_at=datetime();
