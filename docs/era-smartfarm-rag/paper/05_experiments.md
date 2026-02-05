# 5. 실험 및 평가 (Experiments & Evaluation)

본 장은 EdgeKG v3.2의 (i) 검색/생성 품질, (ii) 온프레미스 업데이트 실용성, (iii) 프라이버시 경계(Private Egress 0) 준수 관점에서 평가 프로토콜을 제시한다. 본 절은 “모델 교체”가 아니라 **구현/아키텍처 변화**(Base/Overlay 분리, 업데이트, 다채널 검색)의 효과를 측정하는 데 초점을 둔다.

## 5.1 평가 시나리오

### S1. 온프레미스 민감 업데이트(메모)
1) 사용자 메모 ingest → 2) 로컬 LLM 구조화 추출 → 3) overlay bundle publish → 4) 질의 시 근거/답변 반영

### S2. 온프레미스 민감 업데이트(센서)
1) 센서 배치 ingest → 2) 로컬 rollup으로 요약 chunk 생성 → 3) 구조화 추출 → 4) 질의 시 반영

### S3. 외부지식 Ingress(Base 갱신)
1) Ingress Plane이 base 번들 다운로드 → 2) Private Plane 적용 + base bundle rebuild → 3) API hot reload로 즉시 반영

## 5.2 메트릭

### 5.2.1 Retrieval 품질(채널/융합)
- Precision@K, Recall@K, MRR, NDCG@K
- 채널별 ablation: Dense-only → +Sparse → +Tri-Graph → +TagHash → +CausalGraph

### 5.2.2 Generation 품질(reference-free)
- RAGAS 기반 Faithfulness, Answer Relevancy, Context Precision/Recall [22]
- 폴백 비율: LLM 실패 시 템플릿/검색 전용 전환 빈도

### 5.2.3 Edge 실용성(지연/메모리)
- Query latency (p50/p95), End-to-End latency (검색+생성)
- RSS/피크 메모리, 인덱스 로드 시간, mmap 동작 여부

### 5.2.4 업데이트 실용성
- Update latency: ingest → overlay publish까지의 p50/p95
- Overlay rebuild 시간, facts 추출 실패율(로컬 LLM)

### 5.2.5 프라이버시/운영 경계
- **Owner isolation**: `private` chunk가 owner 불일치 질의에서 컨텍스트에 0건 포함(회귀 테스트)
- **Egress 0 관찰**: Private Plane 경로에서 외부 네트워크 요청이 발생하지 않음을 확인(네트워크 차단/로그 기반)

## 5.3 실험 설계(권장 최소 세트)

1. **E2E 업데이트 테스트(S1/S2)**: 메모/센서 입력이 overlay에 반영되고, 질의에서 근거로 검색되는지 확인
2. **Base 갱신 반영(S3)**: base.sqlite 교체 후 API가 재기동 없이(또는 안전 재로딩으로) 검색 결과가 바뀌는지 확인
3. **채널 ablation**: TagHash/CausalGraph 추가가 품질/지연에 미치는 영향(특히 인과 질의 및 정확 매칭 질의)
4. **프라이버시 회귀**: owner_id를 바꿔 질의했을 때 private 근거가 포함되지 않는지 검증

본 연구는 위 지표를 통해 “엣지에서의 검색/생성 성능”뿐 아니라, **온프레미스 운영에서 중요한 업데이트/보안 요구**를 정량·정성적으로 함께 보고한다.
