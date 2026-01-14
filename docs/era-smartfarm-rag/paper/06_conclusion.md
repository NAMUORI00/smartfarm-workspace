# 6. 결론 (Conclusion)

---

## 6.1 연구 요약

본 연구는 스마트팜 도메인에 특화된 온디바이스 하이브리드 RAG 시스템을 제안하였다. 

**주요 기여:**

1. **3채널 검색 융합**: Dense-Sparse-PathRAG 검색 채널을 동적 가중치로 결합하여 질의 특성에 따른 최적 검색 수행
2. **도메인 온톨로지**: 작물-환경-병해-영양소-생육단계-재배실천 6개 개념 유형으로 농업 도메인 특화 검색 품질 향상
3. **인과관계 그래프**: 규칙 기반 패턴 매칭으로 원인→결과→해결책 인과 체인 구축 (LLM 비용 $0)
4. **엣지 배포 최적화**: Q4_K_M 양자화와 FAISS mmap으로 8GB RAM 엣지 디바이스에서 실시간 추론 지원

**파일럿 스터디 성과:**

본 연구는 220개 QA 쌍을 활용한 파일럿 스터디로서, 소규모 데이터셋의 통계적 한계(MDE ~4-5%)를 인지하면서도 시스템 아키텍처의 타당성과 엣지 배포 가능성을 검증하였다. 특히 MiniLM 임베딩 모델 적용 시 25-40x 속도 향상을 달성하여 8GB RAM 환경에서의 실용적 운영 가능성을 확인하였다.

---

## 6.2 향후 연구

본 파일럿 스터디의 한계를 해결하기 위한 후속 연구 방향을 제시한다.

### 6.2.1 단기 과제 (Short-term)

1. **데이터셋 확장**: QA 500개 이상 확보, 다중 작물(토마토, 딸기, 파프리카) 추가하여 통계적 검정력 강화
2. **전문가 검증**: 50-100개 샘플에 대한 농업 전문가 품질 평가로 합성 데이터 한계 보완
3. **베이스라인 강화**: pyserini BM25, BGE-M3, E5-large 등 공개 모델과의 비교로 일반화 가능성 검증

### 6.2.2 중기 과제 (Medium-term)

4. **PathRAG 고도화**: 현재 BFS 기반 2-hop 탐색에서 원본 PathRAG의 relational path pruning 기법 통합
5. **다국어 확장**: 영어-한국어 교차 언어 검색 지원
6. **생성 품질 평가**: RAG 전체 파이프라인 (검색 + 생성) 통합 평가, Human evaluation 포함

### 6.2.3 장기 과제 (Long-term)

7. **현장 배포**: 실제 스마트팜 농가에서의 사용자 연구를 통한 실용성 검증
8. **연속 학습**: 사용자 피드백 기반 온라인 학습 메커니즘 개발
9. **멀티모달 확장**: 잎/줄기 이미지 기반 병해 진단과 RAG 검색 결합, 센서 데이터 통합

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
