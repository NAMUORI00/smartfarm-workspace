# 6. 결론 (Conclusion)

(작성 예정)

---

## 6.1 연구 요약

본 연구는 스마트팜 도메인에 특화된 온디바이스 하이브리드 RAG 시스템을 제안하였다. 

**주요 기여:**

1. **3채널 검색 융합**: Dense-Sparse-PathRAG 검색 채널을 동적 가중치로 결합하여 질의 특성에 따른 최적 검색 수행
2. **도메인 온톨로지**: 작물-환경-병해-영양소-생육단계-재배실천 6개 개념 유형으로 농업 도메인 특화 검색 품질 향상
3. **인과관계 그래프**: 규칙 기반 패턴 매칭으로 원인→결과→해결책 인과 체인 구축 (LLM 비용 $0)
4. **엣지 배포 최적화**: Q4_K_M 양자화와 FAISS mmap으로 8GB RAM 엣지 디바이스에서 실시간 추론 지원

---

## 6.2 향후 연구

1. **다중 도메인 확장**: 와사비 외 토마토, 딸기, 파프리카 등 다양한 작물 도메인으로 일반화 검증
2. **PathRAG 고도화**: 현재 BFS 기반 2-hop 탐색에서 원본 PathRAG의 relational path pruning 기법 통합
3. **현장 검증**: 실제 스마트팜 농가에서의 배포 및 사용자 연구를 통한 실용성 검증
4. **멀티모달 확장**: 잎/줄기 이미지 기반 병해 진단과 RAG 검색 결합

---

## Acknowledgments

(감사의 글 작성 예정)

---

## AI Usage Statement

This paper was prepared with assistance from AI language models (Claude by Anthropic) for:
- Code implementation and debugging
- Documentation drafting
- Experimental design suggestions

All scientific claims, experimental results, and conclusions were verified and validated by the human authors.
