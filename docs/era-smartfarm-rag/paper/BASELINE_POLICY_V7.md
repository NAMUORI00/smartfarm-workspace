# Baseline Policy v7 (RAGAS 고정, 구조 비교 중심)

## 목적
- 평가 로직(RAGAS 공식 라이브러리)은 고정하고, RAG 구조 개선 효과를 비교군 중심으로 검증한다.
- 엣지 경량 LLM+RAG의 실효성을 `IR 고전 기준선 + GraphRAG 계열 기준선`과 함께 입증한다.

## 공식 비교군 (게이팅)
- `bm25_only`
- `dense_only`
- `rrf`
- `lightrag_mix`
- `ours_structural`

## 비게이팅 항목
- `trigraph_only`는 성능 주장용이 아니라 원인분해(ablation) 용도로만 사용한다.

## 선정 근거
- `bm25_only`: BEIR에서 강한 전통 기준선으로 반복 확인됨.  
  Ref[BEIR]: https://arxiv.org/abs/2104.08663
- `dense_only`: 신경망 기반 단일 채널 naive RAG 대응 기준선.
- `rrf`: 전통 하이브리드 융합 기준선.  
  Ref[RRF]: https://ir.webis.de/anthology/2009.sigirconf_conference-2009.146/
- `lightrag_mix`: 그래프 다층 검색 계열 직접 비교군.  
  Ref[LightRAG]: https://arxiv.org/abs/2410.05779
- `ours_structural`: 제안 구조.

## 데이터셋 범위 (v7)
- `agxqa(test)`: 도메인 QA
- `2wiki(validation)`: multi-hop
- `scifact(BEIR-style)`: 일반 도메인 retrieval

## 평가 고정 원칙
- RAGAS metric 계산식/파이프라인은 수정하지 않는다.
- 구조 변경은 `smartfarm-search` retrieval/ingest에만 적용한다.
- 파라미터 스윕/반복 관측 루프를 금지한다.

