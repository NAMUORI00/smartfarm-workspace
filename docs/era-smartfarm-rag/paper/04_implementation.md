# 4. 구현 (Implementation)

본 장에서는 EdgeKG v3.2의 구현을 구성 요소(서비스), 저장 구조(SQLite SoT + 파생 캐시), 그리고 런타임 업데이트/검색 경로 중심으로 정리한다. 구현은 “외부지식 유입(Ingress)은 허용하되, 온프레미스 내부 지식의 외부 유출(Egress)은 0”이라는 운영 제약을 기본으로 한다.

## 4.1 시스템 구성 및 배포 스택

- **API 서버**: FastAPI 기반 `smartfarm-search` (질의/업데이트 엔드포인트 제공)
- **비동기 워커**: ingest-worker (업로드/센서 입력 처리 → 오버레이 인덱스 빌드)
- **로컬 LLM**: llama.cpp server (생성 + KB 업데이트 구조화 추출에 사용)
- **Ingress 서비스(선택)**: base-sync (외부 Base 번들 다운로드 → inbox 적재)
- **오케스트레이션**: Docker Compose

구현은 Trust Zone을 두 평면으로 분리한다.

- **Public Ingress Plane**: 외부 네트워크 접근 가능. `inbox/`에만 쓰기 가능(RW). Overlay/Private DB 마운트 금지.
- **Private Reasoning Plane**: API/worker/llama 및 SQLite를 포함. 외부 egress를 기본적으로 차단(내부 llama 호출만).

## 4.2 저장 구조: `base.sqlite + overlay.sqlite` SoT 통일

### 4.2.1 Base SoT (`base.sqlite`)

- 공개/외부 지식 기반의 KB를 SQLite로 패키징한 결과물이다.
- Ingress Plane이 `inbox/base_updates/<version>/base.sqlite`로 유입하고, Private Plane이 이를 내부 `base.sqlite`로 적용한다.

### 4.2.2 Overlay SoT (`overlay.sqlite`)

온프레미스에서 발생하는 민감 입력을 저장한다.

- `chunks`: chunk-level 텍스트, 메타데이터, 해시, `sensitivity(public|private)`, `owner_id`
- `extractions/entities/relations`: 로컬 LLM(=llama.cpp) 기반 구조화 추출 결과(facts)
- (옵션) 센서 원시 시계열은 별도 SQLite(`overlay_sensors.sqlite`)에 저장 후 요약 chunk로 KB 반영

### 4.2.3 파생 캐시(버전 디렉터리)

SQLite SoT로부터 재생성 가능한 파생 아티팩트를 디스크에 저장한다.

- Base bundle: `base_bundles/versions/<v>/...` + `CURRENT`
- Overlay bundle: `overlay_uploads/versions/<v>/...` + `CURRENT`

각 버전 디렉터리는 다음을 포함한다.

- Dense: `dense.faiss`, `dense_docs.jsonl`
- Sparse: `sparse_bm25.pkl` 또는 `sparse_tfidf.pkl`
- Tri-Graph(선택): `trigraph_edge/` (mmap-friendly `.npy` + CSR `.npz`)
- TagHash: `tags/` (`tag_alias.json`, `tag2chunk.csr.npz`, `chunk_ids.json` 등)
- CausalGraph: `graph/` (`entity_registry.json`, `entity2chunk.csr.npz`, `edges_*.csr.npz` 등)
- `manifest.json` (아티팩트 해시/메타데이터)
- Overlay의 경우, 버전 디렉터리에 `overlay.sqlite` 스냅샷을 함께 저장한다(백업/복구 용도).

## 4.3 Base Ingress 및 적용(Private Egress 0)

Ingress Plane은 외부 Base 번들을 다운로드하여 `inbox/`에 적재한다. Private Plane은 `inbox/`를 **read-only mount**로 읽고, 다음을 수행한다.

1. `base.sqlite`를 내부 경로로 원자적 교체(파일 복사/rename)
2. `base.sqlite`로부터 파생 캐시 번들(`base_bundles`)을 재생성하고 `CURRENT`를 갱신
3. API는 `CURRENT` 변화를 감지하여 base 인덱스를 hot reload

이 과정은 네트워크 호출 없이 파일 IO만 수행되므로, Private Plane의 egress 0 요구를 위반하지 않는다.

## 4.4 온프레미스 Overlay 업데이트 파이프라인

### 4.4.1 입력 소스 및 비동기 잡

- 텍스트/파일 업로드: `/ingest`, `/ingest_file`
- 센서 입력: `/sensor/ingest` (원시 저장 + rollup job enqueue)

업데이트는 JobStore(SQLite)에 잡으로 큐잉되고, ingest-worker가 백그라운드에서 처리한다.

### 4.4.2 로컬 LLM 기반 KB Update Extractor

업데이트 입력(청크)은 로컬 llama.cpp에 의해 KB Update 스키마로 구조화된다.

- JSON schema 강제 출력(entities/relations)
- canonical_id는 `lower_snake_case` 규칙을 따르며, Base 번들의 태그/레지스트리 후보를 프롬프트에 제공하여 정규화를 안정화한다.

추출 결과는 `overlay.sqlite`의 `extractions/entities/relations` 테이블에 저장된다.

### 4.4.3 오버레이 번들 빌드 및 게시

ingest-worker는 `overlay.sqlite`에서 활성 청크/팩트를 읽어 파생 아티팩트를 생성하고, `overlay_uploads/versions/<v>/`로 게시한다.

- 게시 시점에 `overlay.sqlite` 스냅샷을 버전 디렉터리에 저장한다.
- `CURRENT` 포인터를 원자적으로 갱신하여 API가 안전하게 로드하도록 한다.

## 4.5 런타임 검색 및 프라이버시 필터링

### 4.5.1 다채널 검색 + 융합

API는 Base+Overlay를 함께 검색하며, 채널별 결과를 weighted RRF로 융합한다.

- Dense(FAISS) + Sparse(BM25) + Tri-Graph + TagHash + CausalGraph
- 질의 유형(수치/단위 포함, 인과 키워드 포함 등)에 따라 채널 가중치를 간단 규칙으로 조정한다.

### 4.5.2 Private scope 필터링

컨텍스트 구성 시 `sensitivity=private` 문서는 요청의 `owner_id`가 일치할 때만 포함된다. 이는 모드 토글이 아니라 기본 규칙으로 항상 적용된다.

## 4.6 Egress 방지(방어적 구현)

- **LLMLITE_HOST allowlist**: llama.cpp는 `localhost/llama` 등 로컬 호스트만 허용한다.
- **런타임 다운로드 금지**: 임베딩/리랭커 모델은 `local_files_only` 로딩을 사용하며, 없으면 해당 채널은 실패-폐쇄 또는 비활성화된다.
- **외부 ingest 경로 제거**: 런타임에서 HuggingFace/URL 기반 문서 수집은 제거하고, Base 번들 유입은 Ingress Plane으로만 분리한다.
