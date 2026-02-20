# 3. 제안 방법론 (Proposed Methodology)

본 장에서는 제안 시스템의 핵심 기여인 **Data-driven Adaptive Tuning (DAT)**의 수학적 정의를 기술한다. DAT는 3채널 하이브리드 검색(Dense, Sparse, Graph)의 융합 가중치를 휴리스틱 없이 데이터로부터 자동 결정하는 방법이다. 시스템 아키텍처와 각 채널의 구현 세부사항은 4장에서 다룬다.

## 3.1 문제 정의 (Problem Formulation)

### 3.1.1 기호 정의

| 기호 | 정의 |
|------|------|
| $\mathcal{C} = \{dense, sparse, graph\}$ | 검색 채널 집합 |
| $q$ | 사용자 질의 |
| $d$ | 문서(청크), $d \in \mathcal{D}$ |
| $\mathbf{w} = (w_d, w_s, w_g)$ | 채널 가중치 벡터, $\sum w_c = 1, \; w_c \geq 0$ |
| $\pi_c(d \mid q)$ | 채널 $c$에서 질의 $q$에 대한 문서 $d$의 순위 (1-indexed) |
| $K$ | 최종 반환 문서 수 |
| $\mathcal{R}(q)$ | 질의 $q$의 정답 문서 집합 |

### 3.1.2 채널 융합 문제

3채널 검색의 결과를 단일 순위로 통합하기 위해, Cormack et al. [12]의 Reciprocal Rank Fusion(RRF)을 채널 가중치로 확장한 **Weighted RRF (WRRF)**를 정의한다:

$$
\operatorname{WRRF}(d \mid q, \mathbf{w}) = \sum_{c \in \mathcal{C}} w_c \cdot \frac{1}{k + \pi_c(d \mid q)}
$$

여기서 $k = 60$은 순위 평활 상수이다.

**핵심 문제**: 기존 연구는 $\mathbf{w}$를 수동 설정하거나 도메인 휴리스틱에 의존한다. 본 연구는 $\mathbf{w}^*$를 데이터로부터 자동으로 학습하는 DAT를 제안한다.

---

## 3.2 데이터 분할 프로토콜 (Dataset Splitting Protocol)

DAT의 가중치 학습이 최종 평가에 혼입되지 않도록, 2단계 분할을 적용한다.

### 3.2.1 1단계: DAT 학습 영역 분리

주어진 데이터셋 $\mathcal{Q}$에 대해 고정 시드 $s$로 결정론적 셔플 후, 먼저 DAT 학습용 부분집합 $\mathcal{Q}_{DAT}$와 본 평가용 $\mathcal{Q}_{eval}$로 분리한다:

$$
\mathcal{Q} \xrightarrow{shuffle(s)} \underbrace{\mathcal{Q}_{DAT}}_{\text{20\%}} \;|\; \underbrace{\mathcal{Q}_{eval}}_{\text{80\%}}
$$

$\mathcal{Q}_{eval}$은 DAT 가중치 학습에 **일체 사용하지 않으며**, 논문 결과표의 모든 메트릭은 $\mathcal{Q}_{eval}$에서만 보고한다.

### 3.2.2 2단계: DAT 내부 분할

$\mathcal{Q}_{DAT}$ 내부를 6:2:2 비율로 추가 분할한다:

$$
\mathcal{Q}_{DAT} \;\rightarrow\; \underbrace{\mathcal{Q}_{train}}_{\text{60\%}} \;|\; \underbrace{\mathcal{Q}_{val}}_{\text{20\%}} \;|\; \underbrace{\mathcal{Q}_{test}^{DAT}}_{\text{20\%}}
$$

| 분할 | 전체 대비 비율 | 용도 |
|------|:---:|------|
| $\mathcal{Q}_{train}$ | 12% | DAT 가중치 탐색 (grid search + $F$-fold CV) |
| $\mathcal{Q}_{val}$ | 4% | 후보 깊이·하이퍼파라미터 선택 |
| $\mathcal{Q}_{test}^{DAT}$ | 4% | DAT 자체 효과 확인 |
| $\mathcal{Q}_{eval}$ | **80%** | **본 평가 — Baseline 비교 및 최종 성능 보고** |

### 3.2.3 데이터 격리 보장

- **$\mathcal{Q}_{eval}$(80%)**: DAT 가중치 탐색·하이퍼파라미터 선택 **어디에도** 사용되지 않는다. 모든 방법(Baseline 포함)의 최종 성능을 이 분할에서만 보고한다.
- **$\mathcal{Q}_{train}$ 내부**: $F$-fold CV를 추가 적용하여 과적합을 방지한다.
- **Multi-seed 검증**: 시드 $s \in \{42, 52, 62\}$로 전체 절차를 반복하여 재현성을 확보한다.

---

## 3.3 DAT: 가중치 탐색 (Weight Optimization)

### 3.3.1 탐색 공간

가중치 탐색은 3차원 심플렉스 위의 이산 격자에서 수행된다. 스텝 크기 $\Delta$에 대해:

$$
\mathcal{W} = \left\{ \left(\frac{a}{n}, \frac{b}{n}, \frac{n-a-b}{n}\right) \;\middle|\; a \in [0, n], \; b \in [0, n-a] \right\}, \quad n = \left\lceil \frac{1}{\Delta} \right\rceil
$$

$\Delta = 0.05$일 때 $n = 20$이므로 $|\mathcal{W}| = \binom{22}{2} = 231$개 후보가 생성된다.

후보 깊이(각 채널의 검색 범위) $m$도 함께 탐색한다:

$$
\mathcal{M} = \{2K, \; 4K, \; 8K, \; \max(K, 32)\}
$$

### 3.3.2 목적함수

$\mathcal{Q}_{train}$을 $F = 3$ fold로 나누어 교차 검증을 수행한다. fold $f$에서 가중치 $\mathbf{w}$, 깊이 $m$의 평균 nDCG@K를 $\mu_f(\mathbf{w}, m)$로 표기하면:

$$
J(\mathbf{w}, m) = \underbrace{\frac{1}{F} \sum_{f=1}^{F} \mu_f(\mathbf{w}, m)}_{\text{mean performance}} - \underbrace{\lambda \cdot \sqrt{\frac{1}{F} \sum_{f=1}^{F} \left(\mu_f - \bar{\mu}\right)^2}}_{\text{stability penalty}}
$$

여기서 $\lambda = 0.25$이다. 이 형태는 mean-variance 프레임워크에서 착안한 것으로, 평균 성능이 높으면서 fold 간 분산이 작은 가중치를 선호한다.

**nDCG@K 정의**:

$$
\operatorname{nDCG@K} = \frac{\sum_{i=1}^{K} \frac{2^{\operatorname{rel}(d_i)} - 1}{\log_2(i+1)}}{\operatorname{IDCG@K}}
$$

여기서 $\operatorname{IDCG@K}$는 이상적 순위의 DCG이다.

### 3.3.3 최적 프로파일 선택

$$
(\mathbf{w}^*, m^*) = \underset{(\mathbf{w}, m) \in \mathcal{W} \times \mathcal{M}}{\operatorname{argmax}} \; J(\mathbf{w}, m)
$$

동점 시 $(\bar{\mu} \uparrow, \; \sigma \downarrow, \; m \downarrow)$ 순으로 우선한다.

$\mathbf{w}^*$가 결정되면, $\mathcal{Q}_{val}$에서 검증하고, 최종 성능은 $\mathcal{Q}_{test}$에서만 보고한다.

---

## 3.4 쿼리 적응형 세그먼트 (Query-Adaptive Segments)

단일 글로벌 가중치가 아닌, 쿼리 유형별 최적 가중치를 학습한다.

### 3.4.1 쿼리 특성 추출

질의 $q$에서 4차원 이산 특성 벡터를 추출한다:

$$
\mathbf{f}(q) = \big( f_{mod}, \; f_{len}, \; f_{rel}, \; f_{ent} \big)
$$

| 특성 | 정의 | 값 |
|------|------|----|
| $f_{mod}$ | 검색 모달리티 | $\{text, image, table\}$ |
| $f_{len}$ | 질의 길이 버킷 | $\leq 6$: short, $\leq 14$: medium, $>14$: long |
| $f_{rel}$ | 인과/관계 키워드 존재 | $\{0, 1\}$, 키워드: why, cause, 원인, 비교 등 |
| $f_{ent}$ | 수치 토큰 존재 | $\{0, 1\}$ |

### 3.4.2 세그먼트 매칭 및 튜닝

특성 벡터로 $\mathcal{Q}_{train}$을 세그먼트로 분할한 후, 각 세그먼트 $s$에 대해 §3.3과 동일한 절차로 $\mathbf{w}_s^*$를 독립 탐색한다. 런타임에서 새 질의 $q$가 입력되면, 특성 일치 수가 가장 높은 세그먼트의 가중치를 적용한다:

$$
s^*(q) = \underset{s \in \mathcal{S}}{\operatorname{argmax}} \; \left| \{ k : f_k(q) = f_k(s) \} \right|
$$

세그먼트 샘플 수가 3건 미만이면 글로벌 $\mathbf{w}^*$로 폴백한다.

---

## 3.5 런타임 안전 장치 (Runtime Guardrails)

학습된 가중치의 안전한 런타임 적용을 위해 세 가지 제약을 부과한다.

### 3.5.1 가중치 경계 및 변화량 제한

$$
w_c \in [w_{min}, \; w_{max}], \qquad |w_c - w_c^{prev}| \leq \delta_{max}
$$

기본값: $w_{min} = 0.10$, $w_{max} = 0.80$, $\delta_{max} = 0.15$.

이는 특정 채널이 완전히 비활성화되거나 급격히 변동하는 것을 방지한다. 제약 적용 후 L1 재정규화를 수행한다.

### 3.5.2 품질 게이트

프로파일 활성화 조건:

$$
\operatorname{active}(\mathcal{P}) = \mathbb{1}\left[ n_{\mathcal{P}} \geq 300 \;\land\; \operatorname{age}(\mathcal{P}) \leq 168h \right]
$$

미충족 시 기본 균등 가중치 $\mathbf{w}^{default} = (0.34, 0.33, 0.33)$으로 폴백한다.

### 3.5.3 채널 증거 보정

런타임 검색에서 특정 채널이 유효 결과를 반환하지 못하면 해당 가중치를 동적 억제한다:

$$
w_c' = \begin{cases}
0 & \text{if } |H_c| = 0 \\
0.5 \cdot w_c & \text{if } |H_c| < 2 \\
w_c & \text{otherwise}
\end{cases}
$$

여기서 $H_c$는 채널 $c$의 유효 히트 집합이다. 이 메커니즘은 그래프 인덱스가 불완전한 엣지 환경에서 그래프 채널이 성능을 저하시키는 것을 자동으로 방지한다.

---

## 3.6 요약

DAT의 핵심 설계 원칙을 정리하면:

1. **휴리스틱 배제**: 채널 가중치를 수동 규칙이 아닌 데이터(nDCG@K)로부터 결정
2. **격리된 평가**: train/val/test 3-way 분할로 데이터 유출 차단
3. **쿼리 적응**: 질의 유형별 세그먼트 프로파일로 상황에 맞는 가중치 적용
4. **안전한 배포**: 경계 제약·품질 게이트·증거 보정으로 런타임 안정성 보장

각 채널(Dense, Sparse, Graph)의 검색 메커니즘, 시스템 아키텍처, Private Store 설계는 4장에서, DAT의 효과를 검증하는 ablation study와 엣지 효율성 평가는 5장에서 기술한다.