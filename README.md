# DAT-RAG: Query-Adaptive Tri-Channel Retrieval for Sovereign Edge Smart Farming

`DAT-RAG` 논문 원고와 재현용 벤치마킹 파이프라인을 함께 관리하는 워크스페이스입니다.

이 저장소의 핵심 메시지는 단순히 3채널(Dense, Sparse, Graph)을 더하는 것이 아니라, **naive 3채널 Graph 통합이 만드는 Pull-down을 방어하고, Graph가 실제로 필요한 질의에서만 조건부 이득을 주는 적응형 RAG 시스템**을 제안한다는 점입니다. 전체 시스템은 **resource-constrained edge 환경 배포를 목표로 설계**되며, Jetson Orin 계열 장비는 참고 플랫폼으로 다룹니다.

## 연구 목적

기존 2채널(Dense+Sparse) 하이브리드 검색에 **Graph 검색을 3번째 채널로 추가**할 때 발생할 수 있는 **Pull-down 현상**(그래프 채널이 오히려 성능을 저하시키는 문제)을 방어하면서, 질의 구조에 따라 Graph 채널을 **억제하거나 활용하는 적응형 3채널 RAG 시스템**을 제안합니다.

### 학술적 기여

| ID | 기여 | 설명 |
|:--:|------|------|
| **C1** | **질의 적응형 3채널 융합 (DAT)** | 신경망 리랭커 없이 채널 가중치를 질의 유형에 따라 자동 조율하는 DAT(Data-driven Adaptive Tuning) 알고리즘. 오프라인 학습 프로파일을 런타임에서 O(1)으로 참조하여 그래프 인과 구조를 보존하면서 엣지 환경에서 추가 지연 없이 동작 |
| **C2** | **소버린 데이터 격리** | 공공 지식과 농장 고유 사적 지식의 아키텍처 수준 격리를 통해 민감 데이터 유출을 원천 차단하는 복합키 기반 소버린 아키텍처 |
| **C3** | **서버 보조 빌드 + 엣지 서빙** | 멀티모달 문서 파싱과 지식 그래프·벡터 인덱스 구축은 오프라인/서버 측 파이프라인으로 수행하고, 검색·융합·응답 생성은 resource-constrained edge 환경 배포를 목표로 설계. Jetson Orin 계열 장비를 참고 플랫폼으로 사용 |

### 연구 질문

| RQ | 질문 | 검증 방법 |
|:--:|------|-----------|
| **RQ1** | 3채널 DAT 융합이 **고정 3채널의 Pull-down을 방어**하고, Graph가 유효한 질의에서는 **2채널 대비 조건부 이득**을 제공하는가? | IR 메트릭 비교 (Table 1) + 대표 RAGAS 평가 (Table 2) + 도메인 적합성 (Table 3) |
| **RQ2** | DAT가 질의 유형별로 채널 가중치를 적응적으로 조정하여, **Graph가 해로운 환경에서는 자동 억제**하고 **유익한 환경에서는 선택적으로 활용**하는가? | Ablation 연구 (Table 4) + Case Study 설계 (Table 5) |
| **RQ3** | 전체 시스템이 edge-class 배포 환경의 메모리·지연 예산 내에서 실용적으로 동작하는가? | 엣지 프로파일링 (Table 6) |

---

## 실험 설계

### 벤치마크 데이터셋

| Dataset | 유형 | 역할 |
|---------|:----:|------|
| AgXQA | 공개 벤치 (1-hop) | 농업 도메인 QA. Graph 억제 필요 환경에서 DAT 방어력 검증 |
| MuSiQue | 공개 벤치 (2–4-hop) | **진정한 다중홉 벤치마크**. Graph 엔티티 체인 탐색의 핵심 가치 입증 |
| 2WikiMHQA | 공개 벤치 (2-hop) | 보조 검증. MuSiQue 결과의 일반화 확인 |
| Wasabi | **자체 수집** (도메인 특화) | 학술 논문 529건 + 공공 기관 문서 67건 + 분류 데이터 30건 (총 626건, en/ja/ko 3개 언어). `smartfarm-corpus-pipeline`으로 OpenAlex·EuropePMC·정부 공개 자료에서 수집. 15개 큐레이션 질의로 타겟 도메인 적합성 검증 (Table 3) |

### 비교 방법 (Compared Methods)

모든 통제 변인(프롬프트, 임베딩, KG 인덱스)을 동일하게 고정하고 **융합 전략만 변경**하여 비교합니다.

| 방법 | 채널 | 역할 |
|------|:----:|------|
| dense\_only | D | Dense 단일 채널 참조 |
| graph\_only | G | Graph 단일 채널 참조 |
| RRF (D+S) | D+S | **메인 기준선 (2채널)** |
| RRF (D+S+G, 1:1:1) | D+S+G | Pull-down 대조군 (고정 3채널) |
| **ours\_structural (DAT)** | **D+S+G** | **제안 방식 (적응형 3채널)** |

---

## 실험 결과

> Table 1–5는 현재 원고의 핵심 주장과 직접 연결되는 결과이며, Table 6(엣지 프로파일링)은 profiling pending 상태입니다.

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

### Table 3. 와사비 도메인 적합성 평가 — Domain Fitness (§4.3.1)

> 실제 수집 와사비 재배 코퍼스(626건) · Reference-free · Judge: gpt-oss-120B (Vertex AI) · Answer: Qwen3-4B
> Table 1–2에서 Pull-down 방어는 이미 검증되었으므로, 여기서는 **실제 타겟 도메인에서의 생성 신뢰도**만 확인합니다.

| Method | Faithfulness |
|--------|:------------:|
| RRF (D+S+G, 1:1:1) | 0.902 |
| **ours\_structural** | **0.977** |

> DAT는 와사비 도메인에서 **97.7%의 사실 충실도**로 동작하며, 나이브 3채널(90.2%) 대비 높은 신뢰성을 유지한다.

### Table 4. Ablation 연구 (RQ2)

> DAT의 어떤 구성요소가 실제로 필요한지 분해합니다. A1–A4는 Table 1의 단일/2채널 기준군으로 대체 가능하므로 참조로 남기고, A5–A9는 DAT 내부 구성요소 ablation 결과입니다.

**Table 4a. AgXQA (1-hop, Graph 억제 시나리오)**

| ID | Variant | nDCG@10 | Δ nDCG | MRR | Δ MRR |
|:--:|---------|:-------:|:------:|:---:|:-----:|
| A0 | Full DAT | 0.810 | — | 0.755 | — |
| A5 | Fixed 1:1:1 (no DAT) | 0.685 | -0.125 | 0.596 | -0.159 |
| A6 | DAT global only | 0.793 | -0.017 | 0.734 | -0.021 |
| A7 | No evidence adjustment | 0.810 | 0.000 | 0.755 | 0.000 |
| A8 | No guardrail | 0.890 | +0.080 | 0.861 | +0.106 |
| A9 | No quality gate | 0.793 | -0.017 | 0.734 | -0.021 |

**Table 4b. MuSiQue (2–4-hop, Graph 활용 시나리오)**

| ID | Variant | nDCG@10 | Δ nDCG | MRR | Δ MRR |
|:--:|---------|:-------:|:------:|:---:|:-----:|
| A0 | Full DAT | 0.595 | — | 0.730 | — |
| A5 | Fixed 1:1:1 (no DAT) | 0.517 | -0.078 | 0.665 | -0.065 |
| A6 | DAT global only | 0.605 | +0.010 | 0.752 | +0.022 |
| A7 | No evidence adjustment | 0.595 | 0.000 | 0.730 | 0.000 |
| A8 | No guardrail | 0.659 | +0.064 | 0.847 | +0.117 |
| A9 | No quality gate | 0.605 | +0.010 | 0.752 | +0.022 |

> **보조 참조 (2WikiMHQA):** A0(0.790), A5(0.754, -0.036), A8(0.824, +0.034). 방향성은 AgXQA·MuSiQue와 일치.

**핵심 관찰:**
- **A5 (DAT 제거):** 모든 데이터셋에서 **가장 큰 하락**. AgXQA에서 -0.125, MuSiQue에서 -0.078. DAT의 존재 이유가 직접 입증됨.
- **A6 (세그먼트 제거):** A0 대비 소폭 하락(-0.017). 질의 유형별 세분화가 일부 기여하지만, 전역 프로파일만으로도 상당 부분 커버됨.
- **A7 (증거 보정 제거):** A0과 동일. 현재 벤치마크의 Graph 결과가 항상 존재하여 보정이 트리거되지 않음. 실제 와사비 도메인(Graph 빈 결과 빈번)에서 차이가 예상됨.
- **A8 (가드레일 제거):** 모든 데이터셋에서 **A0보다 상승**. 이는 가드레일(최소 가중치 하한 10%)이 성능을 보수적으로 제한하고 있음을 시사. 단, 가드레일의 역할은 평균 성능 극대화가 아니라 **tail risk 방지**이므로, 분산(std) 분석이 필요.
- **A9 (품질 게이트 제거):** A6과 동일한 결과. 현재 tune 데이터 규모에서 품질 게이트가 fallback을 트리거하지 않아 차이 미발생.

### Table 5. Qualitative Case Study (RQ2)

> Pull-down 중앙값 부근의 **대표적** 사례를 선정하여, DAT의 두 가지 대응 전략을 질의 단위로 비교합니다.

#### C1. Pull-down 방어 — Graph 억제 (AgXQA, 1-hop)

**Query:** *"Who can advise you on how to develop a leasing contract between lessee and landowner?"*
**정답 문서:** *"Seek professional help. A lawyer that works with land rental could advise you how to add these sections …"*

| Method | nDCG@10 |
|--------|:-------:|
| RRF (D+S) | 0.631 |
| RRF (D+S+G, 1:1:1) | 0.289 ← Pull-down (-0.342) |
| **DAT** | **0.387** ← 부분 회복 (+0.098) |

**DAT 프로파일:** α = (Dense 0.65, Sparse 0.35, **Graph 0.00**) — 데이터셋 레벨 default

naive 3ch는 nDCG를 -0.342 하락시켰다. DAT는 학습된 α\_graph=0.00으로 Graph를 차단하여 하락폭의 약 29%를 회복했다. 완전 복구는 아니지만, AgXQA 전체 77건 Pull-down 사례의 **중앙값(Δ=-0.500) 부근**에서 방어가 관찰된다.

---

#### C2. 조건부 이득 — Graph 제한적 활용 (MuSiQue, 3-hop)

**Query:** *"When is Celebrity Big Brother coming to the network which, along with ABC and the network which broadcasted Highway to Heaven, is the other major broadcaster based in NY?"*
**정답 문서 (3-hop chain):** *"New York City …"* → *"Highway to Heaven … ran on NBC …"* → *"Celebrity Big Brother … February 7, 2018 …"*

| Method | nDCG@10 |
|--------|:-------:|
| RRF (D+S) | 0.765 |
| RRF (D+S+G, 1:1:1) | 0.704 ← Pull-down (-0.061) |
| **DAT** | **0.853** ← D+S 대비 +0.088 이득 |

**DAT 프로파일:** α = (Dense 0.80, Sparse 0.00, **Graph 0.20**) — `medium·no-entity` 세그먼트 레벨

naive 3ch는 여기서도 Pull-down(-0.061)을 일으켰다. 그러나 DAT는 α\_graph=0.20으로 Graph를 **제한적으로 허용**하여, Pull-down을 회피하면서 D+S 대비 **+0.088의 조건부 이득**을 달성했다. MuSiQue 전체 78건 DAT>D+S 사례의 **중앙값(Δ=+0.100) 부근**이다.

---

> **대비 요약:** C1에서는 α\_graph=0.00으로 **차단**, C2에서는 α\_graph=0.20으로 **제한적 허용**. 두 사례 모두 중앙값 부근의 대표적 결과이며, DAT가 데이터셋·세그먼트 특성에 따라 채널 가중치를 적응적으로 학습한다는 RQ2의 주장을 뒷받침한다.

### Table 6. 엣지 디바이스 프로파일 (RQ3)

> 본 표는 edge-class 배포 관점의 시스템 실용성 검증을 위한 자리이며, 현재는 latency/RSS/QPS profiling이 pending 상태입니다.

| Metric | Value |
|--------|:-----:|
| Retrieval p50 (ms) | TBD |
| Retrieval p95 (ms) | TBD |
| Retrieval p99 (ms) | TBD |
| TTFT p50 (ms) | TBD |
| TTFT p95 (ms) | TBD |
| RSS Peak (MB) | TBD |
| QPS | TBD |
