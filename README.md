# ERA-SmartFarm-RAG

Dense, Sparse, Graph **3채널 검색을 통합하는 엣지 네이티브 RAG 시스템**의 벤치마킹 워크스페이스입니다.

## 연구 목적

기존 2채널(Dense+Sparse) 하이브리드 검색에 **Graph 검색을 3번째 채널로 추가**할 때 발생할 수 있는 **Pull-down 현상**(그래프 채널이 오히려 성능을 저하시키는 문제)을 방어하면서, 질의 구조에 따라 Graph 채널을 **억제하거나 활용하는 적응형 3채널 RAG 시스템**을 제안합니다.

### 학술적 기여

| ID | 기여 | 설명 |
|:--:|------|------|
| **C1** | **질의 적응형 3채널 융합 (DAT)** | 신경망 리랭커 없이 채널 가중치를 질의 유형에 따라 자동 조율하는 DAT(Data-driven Adaptive Tuning) 알고리즘. 오프라인 학습 프로파일을 런타임에서 O(1)으로 참조하여 그래프 인과 구조를 보존하면서 엣지 환경에서 추가 지연 없이 동작 |
| **C2** | **소버린 데이터 격리** | 공공 지식과 농장 고유 사적 지식의 아키텍처 수준 격리를 통해 민감 데이터 유출을 원천 차단하는 복합키 기반 소버린 아키텍처 |
| **C3** | **엔드투엔드 엣지 배포** | 멀티모달 문서 파싱 → 지식 그래프·벡터 인덱스 구축 → 경량 LLM 추론 → 자동화 품질 평가를 하나의 재현 가능한 파이프라인으로 통합. 8GB급 엣지 디바이스에서 오프라인 동작 |

### 연구 질문

| RQ | 질문 | 검증 방법 |
|:--:|------|-----------|
| **RQ1** | 3채널 DAT 융합이 **고정 3채널의 Pull-down을 방어**하고, Graph가 유효한 질의에서는 **2채널 대비 조건부 이득**을 제공하는가? | IR 메트릭 비교 (Table 1) + 대표 RAGAS 평가 (Table 2) + 도메인 적합성 (Table 3) |
| **RQ2** | DAT가 질의 유형별로 채널 가중치를 적응적으로 조정하여, **Graph가 해로운 환경에서는 자동 억제**하고 **유익한 환경에서는 선택적으로 활용**하는가? | Ablation 연구 (Table 4) + Case Study 설계 (Table 5) |
| **RQ3** | 전체 시스템이 엣지 디바이스의 메모리·지연 예산 내에서 실용적으로 동작하는가? | 엣지 프로파일링 (Table 6) |

---

## 실험 설계

### 벤치마크 데이터셋

| Dataset | Hop 복잡도 | 역할 |
|---------|:----------:|------|
| AgXQA | 1-hop | 농업 도메인 QA. Graph 억제 필요 환경에서 DAT 방어력 검증 |
| HotpotQA | 2-hop | 위키 기반 QA. 단축 추론 존재 시 Graph 기여 한계 관찰 |
| MuSiQue | 2–4-hop | **진정한 다중홉 벤치마크**. Graph 엔티티 체인 탐색의 핵심 가치 입증 |
| 2WikiMHQA | 2-hop | 보조 검증. MuSiQue 결과의 일반화 확인 |

### 비교 방법 (Compared Methods)

모든 통제 변인(프롬프트, 임베딩, KG 인덱스)을 동일하게 고정하고 **융합 전략만 변경**하여 비교합니다.

| 방법 | 채널 | 역할 |
|------|:----:|------|
| no\_rag | — | 하한선 |
| bm25\_only | S | Sparse 기준점 |
| dense\_only | D | Dense 상한선 |
| graph\_only | G | Graph 한계 참조 |
| RRF (D+S) | D+S | **메인 기준선** |
| SG (Sparse+Graph) | S+G | Graph 계열 참조 |
| RRF (D+S+G, 1:1:1) | D+S+G | Pull-down 대조군 |
| **ours\_structural (DAT)** | **D+S+G** | **제안 방식** |

---

## 실험 결과

> 본 README의 수치는 **표별 프로토콜 기준**에 따라 정리합니다.  
> Table 1은 `Q_Test` 기반 메인 검색 평가 요약, Table 2는 AgXQA 생성 품질의 **대표 실행 예시**, Table 3은 실제 수집한 와사비 문서 코퍼스 기반 **보조 도메인 검증의 예비 상태**, Table 4–5는 **설계형 TBD 표**, Table 6은 엣지 프로파일링 설계입니다.

### Table 1. 메인 검색 성능 비교 (RQ1)

> 아래 표에서 **볼드체는 단순 열 최대값이 아니라**, 본 연구의 핵심 주장과 직접 연결되는 비교 포인트를 표시합니다. 즉, 각 데이터셋에서 **2채널 기준점(RRF D+S)**, **나이브 3채널의 Pull-down**, **DAT의 회복 또는 조건부 이득**을 우선적으로 강조합니다.

| Dataset | Method | nDCG@10 | Δ nDCG vs RRF (D+S) | MRR | Recall@10 |
|---------|--------|:-------:|:-------------------:|:---:|:---------:|
| **AgXQA** | dense\_only | 0.881 | -0.019 | 0.846 | 0.985 |
| *(1-hop)* | graph\_only | 0.201 | -0.699 | 0.143 | 0.396 |
| | **RRF (D+S)** | **0.900** | **0.000** | **0.872** | **0.981** |
| | **RRF (D+S+G, 1:1:1)** | **0.721** | **-0.179** | **0.655** | 0.929 |
| | **ours\_structural** | **0.809** | **-0.091** | **0.751** | **0.981** |
| **2WikiMHQA** | dense\_only | 0.811 | +0.040 | 0.988 | 0.801 |
| *(2-hop)* | graph\_only | 0.244 | -0.527 | 0.262 | 0.352 |
| | **RRF (D+S)** | **0.771** | **0.000** | **0.951** | 0.781 |
| | **RRF (D+S+G, 1:1:1)** | **0.700** | **-0.071** | **0.849** | 0.750 |
| | **ours\_structural** | **0.788** | **+0.017** | **0.955** | **0.805** |
| **MuSiQue** | dense\_only | 0.655 | +0.087 | 0.859 | 0.675 |
| *(2–4-hop)* | graph\_only | 0.009 | -0.559 | 0.009 | 0.018 |
| | **RRF (D+S)** | **0.568** | **0.000** | **0.747** | 0.617 |
| | **RRF (D+S+G, 1:1:1)** | **0.512** | **-0.056** | **0.664** | 0.574 |
| | **ours\_structural** | **0.614** | **+0.046** | **0.789** | **0.669** |

**핵심 관찰:**
- **AgXQA (1-hop, Graph 억제 시나리오):** 고정 3채널(1:1:1)은 2채널(D+S) 대비 nDCG가 `0.900 -> 0.721`로 급락하며, **-0.179의 Pull-down**이 발생한다. DAT는 Graph를 자동 억제해 `0.809`까지 회복하며 손실을 **-0.091**로 제한한다. 즉, 나이브 3채널이 만든 하락폭의 약 **49.2%**를 줄인다.
- **2WikiMHQA (2-hop, 보조 검증):** DAT는 2채널 D+S(0.771) 대비 `+0.017`의 소폭 이득을 보이며, AgXQA에서 관찰된 억제 메커니즘이 단순 방어에만 머무르지 않고 중간 난도의 멀티홉 질의에서도 안정적으로 작동함을 시사한다.
- **MuSiQue (2–4-hop, Graph 활용 시나리오):** DAT(0.614)가 2채널 D+S(0.568) 대비 nDCG를 **+0.046 역전**한다. 진정한 다중홉 추론에서 Graph 채널의 조건부 가치가 입증된다.
- 즉, 제안 방식의 1차 목표는 **무조건적인 우월성**이 아니라 **3채널 추가로 인한 하방 위험의 통제**와 **Graph가 유효한 환경에서의 조건부 이득**에 있다.

### Table 2. RAGAS 생성 품질 평가 (RQ1)

> AgXQA 기준 · Reference-free · Judge: gpt-oss-120B (Vertex AI) · Answer: Qwen3-4B
> 이 표 역시 **열 최대값 경쟁**이 아니라, AgXQA의 Graph 억제 시나리오에서 `RRF (D+S)`를 기준선으로 두었을 때 **나이브 3채널이 만든 생성 품질 하락을 DAT가 얼마나 회복하는지**를 읽는 것이 중요합니다.

| Method | Faithfulness | Δ vs D+S | Answer Rel. | Δ vs D+S | Context Prec. | Δ vs D+S | Context Recall | Δ vs D+S |
|--------|:------------:|:--------:|:-----------:|:--------:|:-------------:|:--------:|:--------------:|:--------:|
| **RRF (D+S)** | **0.414** | **0.000** | **0.756** | **0.000** | **0.877** | **0.000** | **0.990** | **0.000** |
| **RRF (D+S+G, 1:1:1)** | **0.444** | **+0.030** | **0.726** | **-0.030** | **0.683** | **-0.194** | **0.989** | **-0.001** |
| **ours\_structural** | **0.443** | **+0.029** | **0.766** | **+0.010** | **0.787** | **-0.090** | **1.000** | **+0.010** |

> **진단용 참조군(표 외):** dense\_only = (0.395, 0.756, 0.879, 0.980), bm25\_only = (0.404, 0.722, 0.840, 0.970), SG = (1.000, 0.757, 0.491, 0.800), graph\_only = (0.241, 0.395, 0.208, 0.333)

| Recovery View | Pull-down by Naive 3ch | Recovery by DAT | Recovery Rate |
|---------------|:----------------------:|:---------------:|:-------------:|
| Answer Relevancy | -0.030 | +0.040 | 133.3% |
| Context Precision | -0.194 | +0.104 | 53.6% |
| Context Recall | -0.001 | +0.011 | +0.010 (Surpassed D+S) |

**핵심 관찰:**
- **표의 주 메시지:** RAGAS에서도 IR Table 1과 같은 방향의 하락이 관찰된다. 즉, 나이브 3채널은 `Answer Relevancy`와 `Context Precision`을 떨어뜨리고, DAT는 이를 회복한다.
- **Context Precision Pull-down 방어:** 나이브 3채널은 `0.877 → 0.683`으로 **-0.194** 하락한다. DAT는 이를 `0.787`까지 회복하여 하락폭의 **53.6%**를 방어한다. 이는 Table 1의 IR 지표(nDCG) 회복 패턴과 방향이 일치한다.
- **Answer Relevancy 조건부 이득:** 나이브 3채널이 `0.756 → 0.726`으로 하락시킬 때, DAT는 `0.766`으로 회복해 **2채널 기준선까지 초과**한다.
- **Faithfulness 해석 주의:** Faithfulness는 `D+S (0.414)`보다 나이브 3채널과 DAT가 모두 높지만, `naive 3ch (0.444)`와 `ours (0.443)`의 차이는 사실상 미미하다. 따라서 이 표의 초점은 **Faithfulness 우월성**보다 **관련성/정밀도 회복**에 둔다.
- **Context Recall 보존:** DAT의 `1.000`은 Graph를 억제하더라도 검색 완전성을 유지했음을 보여준다.

### Table 3. 와사비 도메인 적합성 평가 — Preliminary Domain Validation (§4.3.1)

> 실제 수집 와사비 재배 코퍼스(564건) · Reference-free · Judge: gpt-oss-120B (Vertex AI) · Answer: Qwen3-4B
> 이 표는 공개 벤치의 승부표가 아니라, **실제 수집 도메인 코퍼스에서 어떤 비교축이 이미 준비되었는지**를 보여주는 보조 검증 현황입니다.

| Method | Retrieval | Generation | Role | Status |
|--------|:---------:|:----------:|------|--------|
| RRF (D+S) | Preliminary | Preliminary | 비-그래프 실사용 기준선 | Preliminary run exists |
| RRF (D+S+G, 1:1:1) | Preliminary | Preliminary | fixed 3-channel 대조군 | Preliminary run exists |
| **ours\_structural (DAT)** | **Preliminary** | **Preliminary** | **adaptive 3-channel 제안 방식** | **Preliminary run exists** |

> 현재 저장 결과는 `RRF (D+S)`, `RRF (D+S+G, 1:1:1)`, `DAT` 모두 존재하지만, retrieval 지표와 context 계열 지표가 아직 안정적이지 않습니다. 따라서 Table 3의 역할은 **우월성 주장**이 아니라, 실제 타겟 도메인에서의 **외적 타당성 검증 축**을 명시하는 데 둡니다.

### Table 4. Ablation 연구 (RQ2)

> Q_Test (80%) 기준 · 설계형 TBD · DAT의 어떤 구성요소가 실제로 필요한지 분해해서 읽는 표입니다.

| ID | Variant | Category | Purpose | Expected Pattern | nDCG@10 | Δ nDCG | MRR | Δ MRR |
|:--:|---------|:--------:|---------|------------------|:-------:|:------:|:---:|:-----:|
| A0 | Full DAT | — | 전체 제안 방식 기준점 | 기준 성능 | TBD | — | TBD | — |
| A1 | -Graph | Channel Contribution | Graph 제거 효과 분리 | MuSiQue 계열에서 하락 예상 | TBD | TBD | TBD | TBD |
| A2 | -Sparse | Channel Contribution | Sparse 제거 효과 분리 | 수치/정확매칭 질의에서 하락 예상 | TBD | TBD | TBD | TBD |
| A3 | -Dense | Channel Contribution | Dense 제거 효과 분리 | 의미 매칭 질의에서 하락 예상 | TBD | TBD | TBD | TBD |
| A4 | Single-channel only | Channel Contribution | 융합 자체의 가치 확인 | 전반적 하락 예상 | TBD | TBD | TBD | TBD |
| A5 | Fixed 1:1:1 (no DAT) | DAT Core | DAT 존재 이유를 직접 검증 | 가장 큰 pull-down 대조군 | TBD | TBD | TBD | TBD |
| A6 | DAT global only | DAT Core | 세그먼트 적응성 제거 효과 확인 | A0 대비 완만한 하락 예상 | TBD | TBD | TBD | TBD |
| A7 | No evidence adjustment | DAT Core | 증거 기반 보정의 기여 분리 | graph 빈 결과 시 품질 저하 예상 | TBD | TBD | TBD | TBD |
| A8 | No guardrail | Safety / Guardrail | 가중치 안전장치 제거 효과 확인 | tail risk 증가 예상 | TBD | TBD | TBD | TBD |
| A9 | No quality gate | Safety / Guardrail | fallback 품질 통제 제거 효과 확인 | 저품질 프로파일 노출 위험 증가 | TBD | TBD | TBD | TBD |

> 핵심 대조군은 `A5`입니다. `A5 -> A0` 차이가 곧 "왜 fixed 3-channel 대신 DAT가 필요한가"를 보여주는 직접 증거가 됩니다.

### Table 5. Qualitative Case Study 설계 (RQ2)

> 정성 사례 분석은 Ablation과 다른 역할을 맡습니다. Ablation이 구성요소 기여도를 묻는다면, Case Study는 **왜 그런 결과가 나왔는지**를 질의 단위로 해석합니다.

| Case ID | Dataset | Question Type | Failure / Gain | Fixed fusion behavior | DAT decision | Interpretation | Status |
|:-------:|---------|---------------|----------------|-----------------------|--------------|----------------|--------|
| C1 | AgXQA 또는 Wasabi | single-hop / definition | Graph harmful case | 불필요한 graph 근거가 context를 오염 | graph 가중치 억제 | pull-down을 막는 adaptive suppression 사례 | Design-only / TBD |
| C2 | MuSiQue | genuine multi-hop | Graph helpful case | 2채널만으로는 evidence chain 복원 한계 | graph 가중치 선택적 활성화 | multi-hop evidence chain 복원 사례 | Design-only / TBD |

### Table 6. 엣지 디바이스 프로파일 (RQ3)

| Metric | Value |
|--------|:-----:|
| Retrieval p50 (ms) | TBD |
| Retrieval p95 (ms) | TBD |
| Retrieval p99 (ms) | TBD |
| TTFT p50 (ms) | TBD |
| TTFT p95 (ms) | TBD |
| RSS Peak (MB) | TBD |
| QPS | TBD |
